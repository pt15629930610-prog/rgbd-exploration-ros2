"""Ground whole-instruction model output in the user's words, then expand it into robot skills."""

import json
import math
import os
import re
from urllib.request import ProxyHandler, Request, build_opener

from .route_language import MODEL


VERBS = ['转向', '寻找', '靠近', '进入', '绕行', '巡检', '穿过', '返回入口', '等待', '行驶', '观察', '拍照', '调速', '遇阻退回', '询问', '未支持']
TARGETS = {'十字路口': 'crossroad', '通道开口': 'opening', '作物行': 'row',
           '门': 'door', '作物': 'crop', '其他物体': 'object'}
SIDES = {'左': 'left', '右': 'right', '前': 'front', '后': 'back'}
FIELDS = {'原文': {'type': 'string'}, '动作': {'enum': VERBS},
          '目标': {'enum': list(TARGETS)}, '方位': {'enum': list(SIDES)},
          '物体名称': {'type': 'string'}, '转弯时机': {'enum': ['通过路口后']}}
ITEM = {'type': 'object', 'properties': FIELDS, 'required': ['原文', '动作'],
        'additionalProperties': False}
SCHEMA = {'type': 'object', 'properties': {'动作序列': {'type': 'array', 'items': ITEM}},
          'required': ['动作序列'], 'additionalProperties': False}
PROMPT = """你是温室小车的中文语义解析器。理解整段原话，把用户要求按顺序翻译成 JSON 动作序列。
每项的原文必须连续逐字复制。相邻项覆盖整句，包括附加条件。先读完全文再输出。
方位以入口为准。只有明确说出的目标、方位和具体物体名称才填，其余键省略。距离、等待时长、行号直接从原话读取。
“找”只需观测；“去…前停”要靠近；“穿门出”要穿过。作物品种、具体物体名填物体名称。
单说左拐立即转。若指定在十字路口左拐，转向需目标十字路口。只有要求已经通过该路口再转，才加转弯时机。
绕一圈包括找到作物行、进去、绕回行口，用一个绕行动作表达。遇到第一条可走的岔道进去用进入。
沿一条作物行巡看、走到尽头折返选巡检。要求绕完整一圈选绕行。“走不通就退回来”指退回本行口。
单独的速度约束选调速。巡检中遇到封堵退回本行口选遇阻退回。追加次数要保留在原文。未给次数的“多几圈”或未指定哪侧的“隔壁”选询问。
“接着前面遇到”表示沿当前行进方向继续观察，不是向入口正前方转向。
拍照保存现场图，观察描述现场可见情况。禁止、未实现的条件分支、病害诊断、采摘、无从确定的动作选未支持。不能删除条件后执行动作。
例如“往左转，歇两秒再回起点”输出 {"动作序列":[{"原文":"往左转","动作":"转向","方位":"左"},{"原文":"歇两秒","动作":"等待"},{"原文":"再回起点","动作":"返回入口"}]}。
例如“去右手边的门口停下”输出 {"动作序列":[{"原文":"去右手边的门口停下","动作":"靠近","目标":"门","方位":"右"}]}。
例如“第一个十字口右拐，绕遇到的第二排种植槽一圈”选择转向（目标十字路口、方位右），然后绕行（目标作物行）。
只输出 JSON。"""


def _compact(text):
    return re.sub(r'[\s，,、。；;！!？?：:“”‘’（）()]', '', text)


def _significant(text):
    return bool(re.sub(r'然后|接下来|接着|随后|最后|之后|完了|请你|帮我|麻烦|首先|[先再请呀啊吧]', '', text))


def _blocked(reason, *, dependent=True):
    return {'do': 'unsupported', 'reason': reason, 'dependent': dependent}


NUMBER = r'(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百]+(?:点[零一二三四五六七八九]+)?|半)'
SIDE_WORDS = {'left': r'左', 'right': r'右', 'front': r'对面|对侧|正前|[左右往向朝]前|前门|前头|前(?:方|面)(?=的|那|这|有没)',
              'back': r'身后|后门|[左右往向朝]后|后(?:方|面)的'}
CONDITION = re.compile(r'除非|如果|要是|否则|不然|倘若|假如|(?:若|一旦).+|只有.+才|'
                       r'(?<!没)有(?!'+NUMBER+r'\s*(?:厘米|毫米|米|cm|mm|m)(?:时)?就停)[^，。]{1,14}就|'
                       r'没有就|没找到就|不要|(?<!能)不能|禁止|不得|不许|不准|别(?:再|去|往|转|拐|进|穿|出|动)')
LANDMARKS = {'crossroad': r'十字|四岔|四向交叉', 'row': r'作物行|种植行|行间|种植槽|种植床|排作物|排种植|第\s*'+NUMBER+r'\s*(?:排|行)',
             'opening': r'开口|岔口|岔路|支路|丁字|路口|交叉口|道路|过道|小路|走道|通道', 'door': r'门|出口|出棚',
             'crop': r'作物|植株|植物|盆栽|草莓|生菜|香草|番茄|黄瓜'}
COLOR = re.compile(r'(?:深|浅)?(?:棕|褐|红|黄|金|绿|蓝|白|黑|灰|紫|橙|粉)色')


def quantity(text):
    if text == '半':
        return .5
    if text[0].isdigit():
        return float(text)
    digits = '零一二三四五六七八九'
    integer, _, fraction = text.replace('两', '二').replace('〇', '零').partition('点')
    total = value = 0
    for char in integer:
        if char in '十百':
            total += (value or 1) * (10 if char == '十' else 100)
            value = 0
        else:
            value = value*10 + digits.index(char)
    decimal = ''.join(str(digits.index(c)) for c in fraction)
    return total + value + (float('0.'+decimal) if decimal else 0)


def bounded(value, minimum, maximum, name):
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f'{name}支持 {minimum} 至 {maximum}')
    return value


def measurements(quote):
    """Read physical quantities with their units. A model cannot supply a number absent from the instruction."""
    result = {}
    if re.search(r'[-负]\s*'+NUMBER, quote):
        raise ValueError('未支持负的距离、速度或时长')
    length = r'(厘米|公分|cm|毫米|mm|米|m)'
    scales = {'厘米': .01, '公分': .01, 'cm': .01, '毫米': .001, 'mm': .001}
    num = '(' + NUMBER + ')'
    def save_speed(m):
        value = bounded(quantity(m[1])*scales.get(m[2], 1.), .05, 1., '速度（米/秒）')
        if 'speed' in result and result['speed'] != value:
            raise ValueError('同一步包含多个速度')
        result['speed'] = value
        return ' '
    phrase = re.sub(num+r'\s*'+length+r'\s*(?:/\s*(?:秒|s)|每秒)', save_speed, quote)
    phrase = re.sub(r'每秒\s*'+num+r'\s*'+length, save_speed, phrase)
    groups = [('distance', num+r'\s*'+length, scales, .05, 50, '距离（米）'),
              ('seconds', num+r'\s*(分钟|秒钟|秒)', {'分钟': 60}, .1, 3600, '等待时长（秒）')]
    for key, pattern, factors, low, high, name in groups:
        matches = list(re.finditer(pattern, phrase))
        if len(matches) > 1:
            raise ValueError('一个步骤包含多个'+name)
        if matches:
            m = matches[0]
            result[key] = bounded(quantity(m[1])*factors.get(m[2], 1.), low, high, name)
    order = re.search(r'第\s*'+num, quote)
    if order:
        n = bounded(quantity(order[1]), 1, 30, '序号')
        if int(n) != n:
            raise ValueError('地标序号需要整数')
        result['ordinal'] = int(n)
    elif re.search(r'首个|首条|头一个|头一条|头一排', quote):
        result['ordinal'] = 1
    return result


def expand(item, slots):
    """Bind a model-selected intent to evidence in its own source span."""
    quote, verb = item['原文'], item['动作']
    if CONDITION.search(quote):
        return [_blocked('尚不能判定这一条件或禁止项：' + quote)]
    if verb == '未支持':
        return [_blocked('尚不能执行：' + quote, dependent=False)]
    if re.search(NUMBER + r'\s*(?:度|°)', quote):
        return [_blocked('目前能核实入口基准的方位，尚不能按任意转角执行：' + quote)]

    target = TARGETS.get(item.get('目标'))
    hint = next((name for name, pattern in LANDMARKS.items() if re.search(pattern, quote)), None)
    if hint:
        target = hint
    if verb in ('等待', '返回入口') or (not hint and not item.get('物体名称')
                                   and verb in ('行驶', '转向', '拍照', '观察')):
        target = None
    sides = [side for side, pattern in SIDE_WORDS.items() if re.search(pattern, quote)]
    if 'left' in sides and 'right' in sides:
        raise ValueError('同一步涉及左右两个方位，无法绑定到单一动作')
    side = next((s for s in sides if s in ('left', 'right')), None)
    if side is None and sides:
        side = SIDES.get(item.get('方位'))
        if side not in sides:
            side = sides[0]

    # A spoken completion condition also applies when the model only names the
    # first operation (e.g. entering a row with an explicit requirement to circle it).
    circuit = re.search(r'(?:(' + NUMBER + r')\s*圈|一周|闭环)', quote)
    if circuit and target == 'row':
        if circuit[1] and quantity(circuit[1]) != 1:
            raise ValueError('单次行间绕行支持一圈')
        verb = '绕行'
    if verb == '靠近' and 'distance' in slots and not hint and not item.get('物体名称'):
        verb, target = '行驶', None
    if verb == '靠近' and re.search(r'找|寻', quote) and not re.search(r'停|靠近|跟前', quote):
        verb = '寻找'
    if (verb in ('寻找', '靠近', '转向') and target in ('row', 'opening') and
            re.search(r'进去|驶入|进入', quote) and not re.search(r'能否|能不能|是否|可否|能进去吗', quote)):
        verb = '进入'

    seek = {'do': 'seek', 'target': target}
    if side:
        seek['side'] = side
        axial = next((value for value in sides if value in ('front', 'back')), None)
        if axial and side in ('left', 'right') and verb != '转向':
            seek['region'] = [axial, side]
        if (target in ('row', 'opening') and re.search(r'半区|半边|区域', quote)
                or target == 'row' and re.search(r'[左右](?:手边|边|侧)', quote)):
            seek['area'] = True
    if 'ordinal' in slots:
        seek['ordinal'] = slots['ordinal']
    label = item.get('物体名称', '').strip()
    if label not in quote:
        label = ''
    if target in ('row', 'crop') and label in ('作物行', '行间道', '种植行', '作物', '植物', '植株', '种植槽'):
        label = ''
    # Keep explicit crop names even when the small model only outputs its broader class.
    crop_name = re.search(r'草莓|生菜|盆栽|香草|番茄|西红柿|黄瓜', quote)
    if crop_name and target in ('row', 'crop'):
        label = crop_name[0]
    color = COLOR.search(quote)
    if color and target in ('row', 'crop', 'door'):
        if target == 'door':
            seek['label'] = color[0]
        else:
            label = color[0] + '的' + (crop_name[0] if crop_name else '植物')
    if label and target in ('row', 'crop', 'object'):
        seek['label'] = label
    if target == 'object' and 'label' not in seek:
        raise ValueError('模型没有给出可在原话中找到的物体名称')
    if 'distance' in slots and verb not in ('靠近', '行驶'):
        raise ValueError('距离约束尚未绑定到移动或停车动作')
    if 'seconds' in slots and verb != '等待' and not re.search(r'停|待|歇|等|留', quote):
        raise ValueError('时间条件缺少可验证的等待动作')
    if 'ordinal' in slots and not target:
        raise ValueError('序号缺少对应的地标')

    if verb == '转向':
        if not side:
            raise ValueError('缺少转向方位')
        seek.pop('side', None)  # Left/right describes the turn after the trigger.
        prefix = [seek] if target else []
        after = re.search(r'(?:过了|通过|穿过|过完|过后|之后|以后).*(?:再|随后|然后)', quote)
        if after and target in ('crossroad', 'opening'):
            return prefix + [{'do': 'pass'}, {'do': 'seek', 'target': 'opening', 'side': side}, {'do': 'enter'}]
        return prefix + [{'do': 'turn', 'side': side}]
    if verb == '等待':
        if 'seconds' not in slots and not re.search(r'等|停|待|歇|休息|候|片刻', quote):
            return [_blocked('这句话未表达等待，已跳过：' + quote, dependent=False)]
        return [{'do': 'wait', 'seconds': slots.get('seconds', 2.)}]
    if verb == '返回入口':
        return [{'do': 'home'}]
    if verb == '行驶':
        step = {'do': 'advance', 'distance': slots.get('distance', 1.)}
        if side:
            step['side'] = side
        return [step]
    if verb in ('拍照', '观察'):
        shot = {'do': 'photo' if verb == '拍照' else 'inspect'}
        if target:
            shot['face_target'] = True
            if label:
                shot['label'] = label
            return [seek, shot]
        return [{'do': 'turn', 'side': side}, shot] if side else [shot]
    if verb == '绕行':
        if target not in (None, 'row', 'crop'):
            raise ValueError('可验证的绕行目标为作物行')
        if target is None and re.search(r'原地|车身', quote):
            raise ValueError('原地旋转与绕作物行一圈是不同的动作')
        if target is None:
            return [{'do': 'loop'}]
        seek['target'] = 'row'
        return [seek, {'do': 'enter'}, {'do': 'loop'}]
    if verb == '进入':
        if target is None and re.search(r'这|那|进去', quote):
            return [{'do': 'enter'}]
        seek['target'] = target or 'opening'
        if seek['target'] not in ('row', 'opening'):
            raise ValueError('当前能驶入已确认的行间道或通道开口')
        return [seek, {'do': 'enter'}]
    referenced = not side and 'ordinal' not in slots and bool(re.search(r'刚(?:才|找到|看见)|这扇|那扇|这株|那株|它', quote))
    if verb == '穿过' and referenced:
        return [{'do': 'exit'}]
    if not target:
        raise ValueError('缺少可观测的地标')
    if verb == '寻找':
        return [seek]
    if verb == '靠近':
        if target in ('crossroad', 'opening', 'row'):
            if 'distance' in slots:
                return [seek, {'do': 'landmark_stop', 'distance': slots['distance']}]
            return [seek]  # The junction finder already reaches its measured centre.
        approach = {'do': 'approach'}
        if 'label' in seek:
            approach['label'] = seek['label']
        if 'distance' in slots:
            approach['distance'] = slots['distance']
        return [approach] if referenced else [seek, approach]
    if verb == '穿过' and target in ('door', 'opening', 'crossroad'):
        if target == 'opening' and re.search(r'道路|过道|小路|走道|通道', quote):
            return [seek, {'do': 'enter'}, {'do': 'seek', 'target': 'opening'}]
        return [seek, {'do': 'exit' if target == 'door' else 'pass'}]
    raise ValueError('当前无法验证此动作的完成条件')


ROW_RETURN = re.compile(r'(?:如果|要是|若|如果发现)?(?:走不通|前方不通|遇到封堵|遇到死胡同|前方被堵)(?:时|了)?(?:就|则)?(?:沿原路|原路)?(?:退回|返回|回到|退到)(?:来|去|作物行入口|本行行口|行口|该行入口)')
PATROL = re.compile(r'巡(?:检|查|视|一趟|一遍)|沿(?:这|该)行看看')


def row_steps(item, slots, previous):
    quote = item['原文']
    neighbor = re.search(r'隔壁|相邻|紧挨着', quote)
    if re.search(r'几(?:圈|遍|次|趟)', quote):
        return [_blocked('请指定巡检次数，例如“这条行再巡两趟”')]
    if neighbor and not re.search(r'[左右]', quote):
        return [_blocked('请指定上次行口左侧还是右侧的相邻作物行')]
    if item['动作'] == '询问':
        return [_blocked('请补充目标或次数：'+quote)]
    retreat = ROW_RETURN.search(quote)
    if retreat:
        if retreat.end() != len(quote):
            return [_blocked('退回行口后的动作需要单独解析：' + quote)]
        if retreat.start() == 0:
            if previous and previous[-1]['do'] == 'patrol':
                return []  # The patrol already owns the return to its measured gate.
            return [_blocked('退回条件缺少对应的作物行巡检：' + quote)]
        quote = quote[:retreat.start()].rstrip('，,')
        item = dict(item, 原文=quote)
    if CONDITION.search(quote):
        return expand(item, slots)
    context = list(previous)
    while context and context[-1]['do'] in ('inspect', 'photo', 'wait'):
        context.pop()
    entered = bool(context and context[-1]['do'] == 'enter')
    if entered:
        context.pop()
    known_row = bool(context and context[-1].get('target') == 'row')
    named_row = bool(re.search(LANDMARKS['row'], quote))
    circuit = re.search(r'绕|闭环|一周', quote)
    reuse = re.search(r'这条行|这行|同一行|刚才那条行|原来的行|加巡', quote)
    requires_circuit = circuit and ('巡' not in quote or re.search(r'闭环|必须绕|务必绕', quote))
    patrol = (PATROL.search(quote) or '巡' in quote or item['动作'] == '巡检') and (
        known_row or named_row or reuse or neighbor) and not requires_circuit
    if requires_circuit and known_row and not named_row and item['动作'] == '巡检':
        item = dict(item, 动作='绕行', 目标='作物行')
    if not retreat and re.search(r'走不通|遇到封堵|前方不通|遇到死胡同', quote):
        return [_blocked('尚不能确认封堵时要求的动作：' + quote)]
    if retreat and not patrol:
        return [_blocked('无法绑定退回条件：' + item['原文'])]
    if patrol:
        if any(k in slots for k in ('distance', 'seconds')):
            return [_blocked('巡检距离或时长尚不能核实：' + quote)]
        count = re.search('('+NUMBER+r')\s*(圈|遍|次|趟)', quote)
        times = quantity(count[1]) if count else 1
        if int(times) != times or not 1 <= times <= 20:
            return [_blocked('每次指令支持 1 至 20 次巡检')]
        steps = expand(dict(item, 原文=quote.replace('圈', '趟'), 动作='进入', 目标='作物行'), slots)
        if reuse:
            steps = [{'do': 'resume_row'}]
        if neighbor:
            steps = [{'do': 'neighbor', 'side': 'left' if '左' in quote else 'right'}, {'do': 'enter'}]
        if steps[-1]['do'] == 'unsupported':
            return steps
        leg = {'do': 'patrol'}
        if times > 1:
            leg['count'] = int(times)
        if reuse and re.search(r'一共|总共|总计|补到', quote):
            leg['total'] = True
        steps.append(leg)
    else:
        steps = expand(item, slots)
    if known_row and not named_row and re.search(r'进去|进入|绕|巡', quote) and not re.search(r'另|下一|再找|第', quote):
        if steps[0].get('target') == 'row' and len(steps[0]) == 2:
            steps = steps[1:]
        if entered and steps and steps[0]['do'] == 'enter':
            steps = steps[1:]
    return steps


def compile_semantics(answer, instruction, diagnostics=None, prior=()):
    if not isinstance(answer, dict) or set(answer) != {'动作序列'}:
        raise ValueError('模型回复缺少动作序列')
    items = answer['动作序列']
    if not isinstance(items, list) or not 1 <= len(items) <= 24:
        raise ValueError('模型动作数量异常')
    original = _compact(instruction)
    cursor = 0
    result = []
    for i, item in enumerate(items):
        if (not isinstance(item, dict) or set(item) - FIELDS.keys()
                or not isinstance(item.get('原文'), str) or item.get('动作') not in VERBS
                or not isinstance(item.get('物体名称', ''), str)):
            raise ValueError('模型动作字段无效')
        quote = _compact(item['原文'])
        start = original.find(quote, cursor)
        if not quote or start < 0 or _significant(original[cursor:start]):
            raise ValueError('模型忽略或改写了原话，请重新拆解并逐字引用')
        gap = original[cursor:start]
        cursor = start + len(quote)
        try:
            slots = measurements(quote)
            if ('seconds' in slots and item['动作'] in ('等待', '寻找')
                    and re.search(r'去|到|靠近|门前|门口', quote)
                    and any(re.search(pattern, quote) for pattern in LANDMARKS.values())):
                item = dict(item, 动作='靠近')
            if (set(slots) == {'speed'} and re.fullmatch(
                    r'(?:请|以|按|用|保持|速度|限速|别超过|设为|改为|为|的|每秒|厘米|公分|毫米|米|秒|行驶|开车|行车|开|[\d.s/cm零〇一二两三四五六七八九十百点半])+', quote)):
                steps = [{'do': 'speed'}]
            else:
                steps = row_steps(item, slots, [*prior, *result])
            if not steps:
                if diagnostics is not None:
                    diagnostics.append(f'{quote} → 沿用上一阶段确认的作物行及其退回条件')
                continue
            if ('seconds' in slots and steps[-1]['do'] not in ('unsupported', 'wait')
                    and re.search(r'停|待|歇|等|留', quote)):
                steps.append({'do': 'wait', 'seconds': slots['seconds']})
            if (i + 1 < len(items) and steps[-1].get('target') in ('crossroad', 'opening')
                    and re.search(r'过了|通过|穿过|过完', quote)
                    and isinstance(items[i+1], dict) and items[i+1].get('动作') == '转向'):
                steps.append({'do': 'pass'})
            if (result and result[-1]['do'] == 'pass' and len(steps) == 1 and steps[0]['do'] == 'turn'
                    and re.search(r'再|接着|然后|随后', gap+item['原文'])):
                steps = [{'do': 'seek', 'target': 'opening', 'side': steps[0]['side']}, {'do': 'enter'}]
            speed = slots.get('speed')
            if speed is not None:
                for step in steps:
                    step['speed'] = speed
        except ValueError as exc:
            steps = [_blocked(f'{exc}：{item["原文"]}')]
        if diagnostics is not None:
            diagnostics.append(f'{item["原文"]} → 执行 {json.dumps(steps, ensure_ascii=False)}')
        result.extend(steps)
    if _significant(original[cursor:]):
        raise ValueError('模型遗漏了指令结尾：' + original[cursor:])
    return result


def ask(server, prompt, message, schema, tokens):
    payload = {'model': MODEL, 'temperature': 0, 'max_tokens': tokens,
               'response_format': {'type': 'json_schema', 'json_schema': {'name': 'result', 'schema': schema}},
               'messages': [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': message}]}
    if prompt.startswith(PROMPT) and os.getenv('GREENHOUSE_PATROL_LORA', '').lower() in ('1', 'true'):
        gain = bounded(float(os.environ.get('GREENHOUSE_PATROL_LORA_SCALE', '0.75')), 0., 1.,
                       '巡检适配器强度')
        payload['lora'] = [{'id': 0, 'scale': gain}]
    request = Request(server.rstrip('/') + '/v1/chat/completions', json.dumps(payload).encode(),
                      {'Content-Type': 'application/json'})
    with build_opener(ProxyHandler({})).open(request, timeout=45) as response:
        return json.loads(json.load(response)['choices'][0]['message']['content'])


def ground_objects(server, items):
    """Ask for the visible referent in the user's own words, independently of a scene vocabulary."""
    schema = {'type': 'object', 'properties': {'名称': {'type': 'string'}}, 'required': ['名称'],
              'additionalProperties': False}
    for item in items:
        quote, label = item['原文'], item.get('物体名称', '')
        if (item['动作'] == '拍照' and re.fullmatch(
                r'(?:请|帮我|再|先|然后|接着|原地|就地|现在|麻烦|给我|来)*(?:照|拍)(?:一|个|张|个照)*[相照](?:片)?[吧啊。！! ]*',
                quote.strip())):
            item.pop('目标', None)
            item.pop('物体名称', None)
            continue
        if (item['动作'] not in ('寻找', '观察', '靠近', '拍照')
                or CONDITION.search(quote) or any(re.search(p, quote) for p in LANDMARKS.values())
                or item['动作'] == '靠近' and 'distance' in measurements(quote)):
            continue
        if not label or label not in quote:
            prompt = ('抽出用户要寻找或查看的具体物体的名称，必须是原话里的连续文字。'
                      '如有名称只写物体，不写方向、动作和问句。泛指周围场景、原地拍照或没有具体物体时填空字符串。')
            label = ask(server, prompt, quote, schema, 50).get('名称')
        if isinstance(label, str) and label and label in quote and label not in ('周围', '周边', '四周', '附近', '现场', '相', '照片', '照'):
            item['物体名称'] = label
            item['目标'] = '其他物体'


def interpret_span(server, instruction, diagnostics, prior=()):
    prompt = PROMPT
    schema = SCHEMA
    for attempt in range(2):
        try:
            answer = ask(server, prompt, instruction, schema, 1100)
            compile_semantics(answer, instruction, prior=prior)
            trace = ['模型原答 ' + json.dumps(answer, ensure_ascii=False)]
            ground_objects(server, answer['动作序列'])
            plan = compile_semantics(answer, instruction, trace, prior)
        except ValueError as exc:
            if attempt:
                raise
            prompt = PROMPT + '\n上次输出校验失败，请修正：' + str(exc)
            # A short isolated request can be copied exactly by the response grammar.
            if len(instruction) <= 24 and not re.search(r'[,，。；;、再后并又顺]|接着|同时|末了|随后', instruction):
                item = dict(ITEM, properties={**FIELDS, '原文': {'enum': [instruction]}})
                schema = dict(SCHEMA, properties={'动作序列': {'type': 'array', 'items': item,
                                                           'minItems': 1, 'maxItems': 1}})
            continue
        if diagnostics is not None:
            diagnostics.extend(trace)
        return plan


def interpret_online(server, instruction, diagnostics=None):
    if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 600:
        raise ValueError('请输入 1～600 字的路线指令')
    # "停留两秒后返航" gives an explicit temporal boundary even without punctuation.
    parts = re.split(r'(?<=[秒钟])后(?=[回返])', instruction)
    if len(parts) > 1 and not CONDITION.search(instruction):
        return [step for part in parts for step in interpret_span(server, part, diagnostics)]
    try:
        return interpret_span(server, instruction, diagnostics)
    except ValueError:
        # A missing or invented quote invalidates the whole plan. Retry independent
        # clauses individually, without separating a condition from its consequence.
        parts = [s.strip() for s in re.split(r'[。；;]|[，,]\s*(?=再|回|先|最后|末了)|[，,]?\s*(?:然后|接着|随后)', instruction) if s.strip()]
        if not 2 <= len(parts) <= 8 or CONDITION.search(instruction):
            raise
        steps = []
        for part in parts:
            try:
                steps.extend(interpret_span(server, part, diagnostics, steps))
            except ValueError as exc:
                steps.append(_blocked(f'无法确认原话「{part}」：{exc}'))
                break
        return steps
