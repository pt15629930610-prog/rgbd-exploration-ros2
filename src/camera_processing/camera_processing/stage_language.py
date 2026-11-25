"""Bind an observation to the lifetime of one movement, preserving quoted constraints."""

import json
import re

from . import online_language as lang


MARKER = (r'途中|中途|期间|同时|顺便|顺带|一路留意|沿途|路上|过程中|一旦|一(?=遇到|看到|发现)'
          r'|一?边(?=(?:注意|留意|看|记|监测|观察))|时(?=看到|遇到|发现)')
EVENT = r'遇到|看见|看到|发现|碰到|经过|见到'
SEQUENTIAL = r'然后|接着|随后|之后|巡完后|巡完再|最后|最终'
PARTS = ('主任务', '期间条件与响应', '后续')
PROPERTIES = {**{k: {'type': 'string'} for k in PARTS}, '目标原话': {'type':'string'},
              '响应': {'enum':['记录并继续','停车','左转','右转','不确定']}}
SCHEMA = dict(type='object', properties=PROPERTIES, required=list(PROPERTIES), additionalProperties=False)
PROMPT = ('划分机器人指令的时间作用域。按顺序逐字摘抄三个连续片段，覆盖原句。主任务含速度、'
          '起始位置和行驶或巡检要求。期间条件与响应含途中观测哪个目标及发现后做什么，'
          '例如“途中留意右门，看见就记下来”。后续含巡完后的要求，没有就填空。不要改写。'
          '目标原话逐字摘出方位、序号和地标，例如“第一个十字路口”“右边的门”。'
          '从响应中选发现目标时的行为，含糊或有多个不同响应选不确定。')
SEP = ' ，,。；;！!'


def split_rule(text, answer=None):
    """Ground the requested response and leave physical quantities in the target quote."""
    text = text.strip(SEP)
    if re.search(r'不要|不能|不得|除非|没有|没看|别|未见|忽略', text):
        raise ValueError('监测条件包含否定或例外，请明确触发要求')
    trigger = re.sub('^(?:'+MARKER+r'|一边|并且|并|如果|若|注意观察|注意|留意|观察|'+EVENT+r')+', '', text)
    match = re.fullmatch(
        r'(?P<target>[^，,；;。]+?)[，, ]*(?:(?:一旦|一)?(?:'+EVENT+r')(?:它|了)?|时|的时候|后|了)*'
        r'(?:就|便|则|立即|马上|立刻|请)*'
        r'(?P<effect>记(?:录|下)(?:下来|来|位置|它的位置)?|记录下(?:来|位置)|停(?:下来|车|下|住)?|'
        r'(?:向|往)?[左右](?:转弯|转|拐))', trigger)
    proposed = (answer or {}).get('目标原话', '')
    response = (answer or {}).get('响应')
    grounded_effect = {'记录并继续':r'记|留意|留心|留个位置|标.{0,3}位置',
                       '停车':r'停|刹车', '左转':r'左', '右转':r'右'}
    if match:
        target, effect = match['target'], match['effect']
    elif (proposed and proposed in trigger and response in grounded_effect
          and re.search(grounded_effect[response], text.replace(proposed, '', 1))
          and not re.search(r'再|然后|必须|拍照|米|秒|第', trigger.replace(proposed, '', 1))):
        target = proposed
        # Reject an ambiguous response instead of silently dropping a requested action.
        remainder = trigger.replace(proposed,'',1)
        if sum(bool(re.search(expr, text)) for expr in ('门','十字|四岔','草莓|生菜|盆栽|植株')) > 1:
            raise ValueError('一个运动阶段请指定一个监测目标')
        if re.search('门|十字|四岔|这株|那株|左边|右边|左侧|右侧|对面', remainder):
            raise ValueError('没有完整绑定监测目标或它的方位')
        if sum(bool(re.search(expr,remainder)) for expr in (r'停|刹车',r'记|标记',r'左|右')) > 1:
            raise ValueError('同一监测条件有多个响应，请拆成独立阶段')
        effect = {'记录并继续':'记录','停车':'停','左转':'左转','右转':'右转'}[response]
    else:
        # “途中留意右门” requests a record without changing motion.
        if not re.search(r'留意|顺便观察|沿途观察', text):
            raise ValueError('请明确途中发现目标后要记录、停车还是转弯')
        target, effect = trigger, '记录'
    token = re.fullmatch(r'(?:第'+lang.NUMBER+r'(?:个|处|扇|株|棵))?'
                         r'(?P<side>左|右|对面|前|后)?(?:手边|侧|边|方|面)?的?(?:那扇)?'
                         r'(?P<label>草莓|生菜|盆栽|番茄|黄瓜|香草)?'
                         r'(?P<kind>房门|大门|门|十字路口|十字口|四岔口|植株|作物|植物)?', target)
    if not token or not (token['label'] or token['kind']):
        raise ValueError('暂不能核实途中观测目标：'+target)
    kind = ('door' if '门' in target else 'crossroad' if re.search('十字|四岔',target) else 'crop')
    if token['label'] and kind != 'crop':
        raise ValueError('目标种类和名称不一致')
    side = {'对面':'front', **lang.SIDES}.get(token['side'])
    if effect.startswith('记'):
        action = 'record'
    elif effect.startswith('停'):
        action = 'stop'
    else:
        if kind != 'crossroad':
            raise ValueError('途中转弯需要指定路口')
        action = 'turn'
    rule = dict(target=kind, effect=action, **lang.measurements(target))
    if side:
        rule['side'] = side
    if token['label']:
        rule['label'] = token['label']
    if action == 'turn':
        rule['turn_side'] = 'left' if '左' in effect else 'right'
    return rule


def scoped_plan(server, text, diagnostics):
    # An explicitly sequential connective leaves the following junction to the next skill.
    marker = re.search(MARKER, text)
    if not marker:
        if not re.search(r'沿.{1,25}(?:走|开|行驶|前进|穿行|巡检)', text):
            return None
        marker = re.search(r'(?:'+EVENT+r')(?=[^，,；;]{1,16}(?:就|便|立即|立刻))', text)
    if not marker or not marker.start() or re.search('(?:'+SEQUENTIAL+r')[，, ]*$', text[:marker.start()]):
        return None
    # Preserve the existing blocked-row branch, which has its own evidence and recovery.
    if re.match(r'(?:一旦|遇到)(?:行不通|走不通|封堵|堵住)', text[marker.start():]):
        return None
    prefix, remaining = text[:marker.start()].strip(SEP), text[marker.start():]
    if not prefix:
        raise ValueError('请明确在哪段运动中启用监测')
    tail = re.search(SEQUENTIAL+r'|[，,；;]\s*(?=(?:如果|若|要是|如果行末))', remaining)
    chunks = (prefix, remaining[:tail.start()] if tail else remaining, remaining[tail.start():] if tail else '')
    answer = lang.ask(server, PROMPT, text, SCHEMA, 520)
    quoted = tuple(str(answer.get(k, '')).strip(SEP) for k in PARTS)
    # Exact coverage prevents a small model from dropping a speed, recovery or destination.
    if (all(isinstance(answer.get(k), str) for k in PARTS)
            and lang._compact(''.join(quoted)) == lang._compact(text)
            and quoted[0] and quoted[1]
            and lang._compact(quoted[0]) == lang._compact(prefix)
            and (not quoted[2] or re.match(SEQUENTIAL+'|如果|要是|若', quoted[2]))
            and not re.search(SEQUENTIAL, quoted[1])):
        chunks = quoted
    main, condition, after = (s.strip(SEP) for s in chunks)
    rule = split_rule(condition, answer)
    main = re.sub(r'^一边|^在(?=沿|穿过|巡检)', '', main)
    main = re.sub(r'(草莓|生菜|盆栽|番茄|黄瓜|香草)行', r'\1作物行', main)
    speed = re.match(r'^(?:以|按|用)?'+lang.NUMBER+r'\s*(?:米|m)\s*(?:/\s*(?:秒|s)|每秒)'
                     r'(?:的?速度)?(?:行驶)?[，, ]*(?=沿)', main)
    setup = []
    if speed:
        setup.append({'do':'speed', 'speed':lang.measurements(speed[0])['speed']})
        main = main[speed.end():]
    # Normalize bare along-row driving to the same row-traversal skill as “穿过”.
    along = re.search(r'沿(?:着)?(?P<row>(?:左边|右边|左侧|右侧)?的?(?:第'+lang.NUMBER+
                      r'(?:条|排))?(?:草莓|生菜|盆栽)?(?:作物行|种植行|行作物))(?:向前|往前|直)?(?:走|开|行驶|前进)$', main)
    if along:
        main = main[:along.start()]+'穿过'+along['row']
    # Patrol already implements a measured blocked-end return; keep that condition with the host.
    recovery = re.match(r'(?:如果|若|要是)?(?:行末|行尾|尽头)?(?:堵了|堵住了?|封堵|走不通)'
                        r'(?:就|则)?(?:原路|沿原路)?(?:退回|返回)(?:来|行口)?(?=$|[，,；;。]|巡完|然后)', after)
    if recovery:
        after = after[recovery.end():].strip(SEP)
    base = lang.interpret_online(server, main, diagnostics)
    if (not base or base[-1]['do'] not in ('traverse_row','patrol','loop','advance')
            or 'watch' in base[-1] or 'if_blocked' in base[-1]):
        raise ValueError('途中监测需要绑定到一段明确的穿行、巡检或前进任务')
    if recovery and base[-1]['do'] != 'patrol':
        raise ValueError('封堵返回条件需要绑定到本行巡检')
    base[-1] = {**base[-1], 'watch':rule}
    after = re.sub('^(?:'+SEQUENTIAL+r')[，, ]*','',after)
    if rule['effect']=='stop' and after:
        raise ValueError('发现目标后停车与后续行驶同时存在，请说明何时恢复')
    if diagnostics is not None:
        diagnostics.append('阶段划分模型原答 '+json.dumps(answer,ensure_ascii=False))
        diagnostics.append(f'在「{main}」期间启用「{condition}」，规则 {rule}')
    return setup + base + (lang.interpret_online(server, after, diagnostics) if after else [])
