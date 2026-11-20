"""Translate a greenhouse instruction once, then validate the ordered actions."""

import json
import re
from urllib.request import ProxyHandler, Request, build_opener


MODEL = 'Qwen3-VL-2B-Instruct'
ACTIONS = {'A': '转弯', 'B': '绕行一圈', 'C': '穿门出棚', 'D': '返回入口', 'E': '等待', 'X': '未确定'}
PROMPT = """给中文小车指令的每个编号短句选一个动作类别。每个编号只选一次。
A 转弯进入横道；B 绕一块种植槽走一圈；C 穿门去棚外；D 返回原入口；E 原地等待；X 其他或不清楚。
只到门口而不出门选 X。数字、左右和前后由程序从原文另行读取。按编号输出 JSON。"""
NUM = r'(?:[0-9]+|[零一二两三四五六七八九十]+)'
TURN = r'[左右](?:手边|边|侧)?[拐转]|[拐转].{0,3}[左右]|(?:[路叉]口|十字口).{0,5}[左右]'
HOME = r'(?:回|返).*(?:入口|起点|出发点|原处)|返航|原路返回|回来'
EXIT = r'出棚|出温室|出大棚|出去|离棚|离开(?:温室|大棚)|门.{0,3}出(?!口)|穿过.{0,7}门|出.{0,3}门|到棚外'
WAIT = r'停留|等候|等待|待上|待在|(?:停|等)(?:下|上|个|候)?\s*' + NUM


def number(text):
    digits = '零一二三四五六七八九'
    text = text.replace('两', '二')
    if text.isdecimal():
        value = int(text)
    elif len(text) == 1 and text in digits:
        value = digits.index(text)
    elif re.fullmatch('[一二三四五六七八九]?十[一二三四五六七八九]?', text):
        tens, ones = text.split('十', 1)
        value = (digits.index(tens) if tens else 1)*10 + (digits.index(ones) if ones else 0)
    else:
        value = 0
    if not 1 <= value <= 60:
        raise ValueError('序号和等待秒数须为 1 至 60 的整数')
    return value


def split_instruction(text):
    if re.search(r'如果|要是|除非|否则|或者|还是|不要|禁止|别(?:去|往|出|回|拐|转)|不(?:许|能|要|出|转|回|去)|'
                 r'原地转|拍照|拍摄|病虫|采摘|速度|米/秒|米每秒|m/s', text):
        raise ValueError('这条指令含有尚未实现的条件或动作，请拆开说明')
    if re.search(r'\d+[.点]\d+|[一二三四五六七八九零]点[一二三四五六七八九零]|半秒|分钟', text):
        raise ValueError('等待时长请使用整秒')
    parts = re.split(r'[，,。；;！？!?\n]+|然后|接着|随后|最后|之后|后(?=[沿从往向等停回返])|'
                     r'(?:再|并且|并)(?=[从去往向走等停绕返回出进])|(?<=圈)(?=从|返回|回到)|'
                     r'(?<=[转拐])(?=沿|进入|进第|走第)', text)
    clauses, prefix = [], ''
    for part in parts:
        part = part.strip()
        if not part:
            continue
        part = prefix + part
        prefix = ''
        # Keep an ordinal/landmark with its verb across spoken punctuation.
        if (re.fullmatch(r'(?:请|先)?(?:进棚|从入口|进去)?(?:一直|往前|向前)?(?:走|直走|出发)', part) or
                re.search(r'(?:路口|十字口|交叉口|作物行|种植行|行间道|行间通道)$', part)
                and not re.search(r'绕|兜|左拐|右拐|左转|右转', part)):
            prefix = part
        else:
            clauses.append(part)
    if prefix:
        clauses.append(prefix)
    if not 1 <= len(clauses) <= 10:
        raise ValueError('请给出一至十个按顺序执行的动作')
    return clauses


def literal_action(clause):
    """Independent evidence for an explicit verb; also constrains the small model's vocabulary."""
    if re.search(r'门(?:口|前).*停|停.*门(?:口|前)', clause):
        return 'X'
    patterns = {'A': TURN, 'B': r'圈|一周|闭环', 'C': EXIT, 'D': HOME, 'E': WAIT}
    matches = [code for code, pattern in patterns.items() if re.search(pattern, clause)]
    if len(matches) > 1:
        raise ValueError('请用“然后”分开这几个动作：' + clause)
    return matches[0] if matches else None


def extract_step(op, clause):
    if op in ('A', 'C'):
        sides = [v for ch, v in [('左', 'left'), ('右', 'right')] if ch in clause]
        if op == 'C':
            sides += [v for ch, v in [('前门', 'front'), ('对面', 'front'), ('后门', 'back')]
                      if ch in clause]
        if len(set(sides)) != 1:
            raise ValueError('请明确方位：' + clause)
        if op == 'C':
            if not re.search(r'门|出口', clause) or not re.search(EXIT, clause):
                raise ValueError('请明确要穿过的门')
            return {'do': 'exit', 'side': sides[0]}
        if not re.search(r'路口|十字口|交叉口|拐|转', clause):
            raise ValueError('请明确要在哪个路口转弯：' + clause)
        order = re.search(r'第(' + NUM + r')(?=(?:个|处)?(?:十字|交叉)?(?:路口|口))', clause)
        if '第' in clause and order is None:
            raise ValueError('请明确十字路口的序号')
        return {'do': 'turn', 'side': sides[0], 'at': number(order[1]) if order else 1}
    if op == 'B':
        if not re.search(r'行|排|种植槽|种植床', clause):
            raise ValueError('请说明要绕行的作物行')
        if not re.search(r'绕|兜|环|圈|一周', clause):
            raise ValueError('请说明是否要绕作物行一圈')
        if re.search(r'草莓|生菜|盆栽|香草|番茄|西红柿|黄瓜', clause):
            raise ValueError('这版按通道序号选行；请用第几条作物行指定位置')
        count = re.search('(' + NUM + r')\s*(?:圈|周)', clause)
        if (count and number(count[1]) != 1) or re.search(r'半圈|几圈|多圈|数圈', clause):
            raise ValueError('当前一次绕行支持一圈')
        order = re.search(r'第(' + NUM + r')', clause)
        if order is None and not re.search(r'第一|首|头一|最先|头排|最初', clause):
            raise ValueError('请说明要绕第几条行间通道')
        front = re.search(r'(?:朝|向|往)(?:入口的?|棚的?|大棚的?)?前', clause)
        back = re.search(r'(?:朝|向|往)(?:入口的?|棚的?|大棚的?)?后', clause)
        if front and back:
            raise ValueError('请指定一个进入作物行的方向')
        toward = 'auto'
        if front:
            toward = 'front'
        if back:
            toward = 'back'
        return {'do': 'loop', 'row': number(order[1]) if order else 1, 'toward': toward}
    if op == 'D':
        if not re.search(HOME, clause):
            raise ValueError('请说明这一步的目标位置：' + clause)
        return {'do': 'home'}
    if op == 'E' and re.search(r'停|等|待|歇', clause):
        duration = re.search(r'(' + NUM + r')\s*秒', clause)
        if duration:
            return {'do': 'wait', 'seconds': number(duration[1])}
        raise ValueError('请说明停留多少秒')
    raise ValueError('请说明这一步要转弯、绕行、穿门、等候还是返航：' + clause)


def interpret(server, instruction, diagnostics=None):
    clauses = split_instruction(instruction)
    evidence = [literal_action(c) for c in clauses]
    # Fixed keys prevent dropping or inventing a stage. For a clearly grounded verb
    # the grammar permits that action or an abstention, never an unrelated command.
    fields = {}
    for i, op in enumerate(evidence, 1):
        allowed = list('ABCDEX') if op is None else [op]
        if 'X' not in allowed:
            allowed.append('X')
        fields[str(i)] = {'enum': allowed}
    schema = {'type': 'object', 'properties': fields, 'required': list(fields), 'additionalProperties': False}
    numbered = '\n'.join(f'{i}. {clause}' for i, clause in enumerate(clauses, 1))
    payload = {'model': MODEL, 'temperature': 0, 'max_tokens': 160,
               'response_format': {'type': 'json_schema', 'json_schema': {'name': 'actions', 'schema': schema}},
               'messages': [{'role': 'system', 'content': PROMPT}, {'role': 'user', 'content': numbered}]}
    req = Request(server.rstrip('/') + '/v1/chat/completions',
                  json.dumps(payload).encode(), {'Content-Type': 'application/json'})
    with build_opener(ProxyHandler({})).open(req, timeout=35) as response:
        answer = json.loads(json.load(response)['choices'][0]['message']['content'])
    if not isinstance(answer, dict) or set(answer) != set(fields):
        raise ValueError('模型遗漏或添加了动作编号')
    steps = []
    for i, (clause, explicit) in enumerate(zip(clauses, evidence), 1):
        code = answer[str(i)]
        if code not in fields[str(i)]['enum']:
            raise ValueError('模型动作与原文不一致：' + clause)
        if diagnostics is not None:
            grounded = ACTIONS[explicit] if explicit else '未显式指定'
            diagnostics.append(f'{i}. {clause} → Qwen {code}（{ACTIONS[code]}），原文动作：{grounded}')
        steps.append(extract_step(explicit or code, clause))
    return steps
