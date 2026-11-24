"""Build the second patrol-language curriculum and a separate behavioral exam."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src/camera_processing'))
from camera_processing.online_language import PROMPT, SCHEMA
from make_data import dump, item

PROMPT_V2 = PROMPT + """
补充巡检范围标注。要求遍历所有符合条件的行时，在巡检动作填“范围":"全部”；仅指定一行省略范围。
指定遍历的第一站用独立动作“指定起始行”，目标作物行，方位和序号来自原文。这是整次巡检的起点参数，按原文位置标注。
范围上的方位和作物是筛选条件；起点方位仅写在指定起始行。每秒多少米的约束与巡检动作分开引用。
巡检次数是追加还是总计保留原话。缺少参照时明确询问。"""
SCHEMA_V2 = deepcopy(SCHEMA)
fields = SCHEMA_V2['properties']['动作序列']['items']['properties']
fields['动作']['enum'].append('指定起始行')
fields['范围'] = {'enum': ['全部']}


def patrol(text, side=None, crop=None, every=False):
    action = item(text, '巡检', '作物行', side, crop)
    if every:
        action['范围'] = '全部'
    return action


def start(text, side):
    return item(text, '指定起始行', '作物行', side)


def record(family, actions, separator='，', pair=None):
    q = separator.join(a['原文'] for a in actions)
    return dict(id=hashlib.sha256(q.encode()).hexdigest()[:16], family=family, pair=pair,
                messages=[dict(role='system', content=PROMPT_V2), dict(role='user', content=q),
                          dict(role='assistant', content=dump({'动作序列': actions}))])


def curriculum(rng, split):
    train = split == 'train'
    result = {}
    limit = 640 if train else 52
    while len(result) < limit:
        side, door = rng.choices(['左', '右'], k=2)
        crop = rng.choice(['草莓', '生菜', '盆栽', '番茄', '黄瓜'])
        n = rng.choice(['一', '二', '三'])
        count = rng.choice(['两', '三', '四'])
        speed = rng.choice(['0.15', '0.2', '0.35', '0.5', '0.75', '1'])
        dist = rng.choice(['半', '一', '两', '五'])
        place = f'{side}边第{n}条作物行'
        first = start(rng.choice([f'从{place}开始', f'起点选{place}', f'打头的就选{place}'])
                      if train else f'{place}排在最先', side)
        all_rows = patrol(rng.choice(['巡检完每一条作物行', '各条作物行挨个检查', '把棚里的种植行都巡一遍',
                                      '作物行一条也别落下，逐行巡查', '全部作物行都进去看看'])
                          if train else '棚内各排作物逐一巡完', every=True)
        selected = patrol((f'把{side}边的{crop}行全部巡完' if train else f'{side}侧每排{crop}都巡查'),
                          side, crop, True)
        single = patrol((f'只巡{place}' if train else f'单独检查{place}'), side)
        velocity = item(f'以{speed}米/秒的速度' if train else f'限速每秒{speed}米', '调速')
        turn = item(f'先往{side}拐' if train else f'起步朝{side}转弯', '转向', side=side)
        cross = item(f'到第{n}个十字口{side}拐' if train else f'第{n}处四岔口向{side}转',
                     '转向', '十字路口', side)
        park = item(f'在{door}门前{dist}米停' if train else f'离{door}门还有{dist}米停车',
                    '靠近', '门', door)
        exit_door = item(f'从{door}门出去' if train else f'穿过{door}门出棚', '穿过', '门', door)
        home = item('回原来的入口' if train else '返回进棚处', '返回入口')
        wait = item(f'等{count}秒' if train else f'歇{count}秒钟', '等待')
        repeat = patrol(rng.choice([f'同一行再巡{count}遍', f'这行一共巡{count}遍'])
                        if train else f'这条行追加{count}趟')
        neighbor = patrol(f'巡检{side}侧隔壁的作物行' if train else f'巡查{side}手边相邻的那行', side)
        blocked = item('走不通就退回来' if train else '前方被堵就退回本行行口', '遇阻退回')
        clarify = item('隔壁那条也去一趟' if train else '旁边那一排也检查下', '询问')
        shot = item('拍张照' if train else '原地拍照', '拍照')
        family, actions = rng.choice([
            ('start', [velocity, all_rows, first]), ('start', [first, all_rows, exit_door]),
            ('start', [velocity, first, all_rows]), ('scope', [velocity, selected, park]),
            ('scope', [velocity, single, exit_door]), ('scope', [all_rows, park]),
            ('order', [turn, cross, single]), ('order', [cross, turn, single]),
            ('count', [single, repeat, home]), ('neighbor', [single, neighbor, park]),
            ('retreat', [single, blocked, exit_door]), ('clarify', [single, clarify]),
            ('stop', [velocity, park, wait, home]), ('inspection', [selected, shot, home]),
        ])
        row = record(family, actions, rng.choice(['，', '；', '。']))
        # Punctuation-only variants contribute no new examples.
        key = tuple(a['原文'] for a in actions)
        result[key] = row
    return list(result.values())


def exam():
    rows = []
    def pair(family, a, b):
        key = len(rows) // 2 + 1
        rows.extend(record(family, actions, pair=key) for actions in (a, b))
    pair('范围', [patrol('左边每条作物行都替我看看', '左', every=True)],
         [patrol('左边这条作物行替我看看', '左')])
    pair('范围', [patrol('所有草莓行各巡一趟', crop='草莓', every=True)],
         [patrol('第二条草莓行巡一趟', crop='草莓')])
    pair('范围', [patrol('一排一排地巡，整个棚的作物行都要顾到', every=True)],
         [patrol('就巡眼前第一排作物')])
    pair('起点', [start('头一站安排在右边第二条作物行','右'), patrol('全棚各行都跑一遍', every=True)],
         [start('头一站安排在左边第二条作物行','左'), patrol('全棚各行都跑一遍', every=True)])
    pair('起点', [patrol('所有行都要进去巡查', every=True), start('左手头一排作物打头阵','左')],
         [patrol('左手所有行都要进去巡查','左',every=True)])
    pair('起点', [item('每秒半米','调速'), patrol('棚里各排都查一遍',every=True),
                 start('第一站是右边第三排','右')],
         [item('每秒半米','调速'), patrol('棚里各排都查一遍',every=True),start('第一站是右边第二排','右')])
    pair('顺序', [item('先拍照','拍照'),item('再回入口','返回入口')],
         [item('先回入口','返回入口'),item('再拍照','拍照')])
    pair('顺序', [item('出发就左转','转向',side='左'),item('第二个十字口右转','转向','十字路口','右')],
         [item('出发就右转','转向',side='右'),item('第二个十字口左转','转向','十字路口','左')])
    pair('顺序', [item('在头一个十字口往左拐','转向','十字路口','左')],
         [dict(item('通过头一个十字口后再往左拐','转向','十字路口','左'),转弯时机='通过路口后')])
    pair('次数与对象', [patrol('刚才那条行再查两趟')], [patrol('刚才那条行总共查两趟')])
    pair('次数与对象', [patrol('第二排生菜进去巡三次',crop='生菜')], [patrol('第三排生菜进去巡两次',crop='生菜')])
    pair('次数与对象', [patrol('左手边相邻的作物行也巡视一下','左')], [item('相邻的作物行也巡视一下','询问')])
    pair('门与停靠', [item('到右侧门口前两米刹停','靠近','门','右')], [item('从右侧门口开出去','穿过','门','右')])
    pair('门与停靠', [item('离左门还有半米就停车','靠近','门','左')], [item('离左门还有两米就停车','靠近','门','左')])
    pair('门与停靠', [item('去对面的门口停下','靠近','门','前'),item('等三秒','等待')],
         [item('等三秒','等待'),item('去对面的门口停下','靠近','门','前')])
    pair('条件', [patrol('头一排草莓进去查一趟',crop='草莓'),item('走不通就原路退回行口','遇阻退回')],
         [item('只有发现病叶才原路退回行口','未支持')])
    pair('条件', [item('不要从左门出去','未支持')], [item('请从左门出去','穿过','门','左')])
    return rows


def incident():
    return record('历史原话', [item('以1米/秒的速度','调速'), patrol('巡检完每一条作物行',every=True),
                             start('从左边第一条作物行开始','左')], '')


def paraphrases():
    """Three independently worded alternatives for each operator intention."""
    rows = []
    for speed,begin,coverage in [
        ('每秒一米开','先拿左边第一排当起点','大棚里面有几排作物就巡几排'),
        ('车速给到1米/秒','打左手第一条种植行起步','把棚里的作物行挨排查完'),
        ('按1米每秒行驶','起始行定在左边头一排','所有种植行都走访到'),
    ]:
        actions=[item(speed,'调速'),start(begin,'左'),patrol(coverage,every=True)]
        if len(rows)%2:
            actions[1:]=actions[1:][::-1]
        rows.append(record('全覆盖倒装',actions))
    for turn,row,condition,exit_ in [
        ('开局朝左拐','看到第一条作物行就进去巡视','走不通就退回行口','完事从右门出去'),
        ('先左转一下','碰到头一条种植行进去巡一遍','遇到死胡同就原路返回行口','然后经右门出棚'),
        ('一开始往左边开个弯','遇上的第一排作物进去走查一次','前方被堵就退回本行行口','最后穿过右门'),
    ]:
        rows.append(record('巡检退回出门',[item(turn,'转向',side='左'),patrol(row),
                                          item(condition,'遇阻退回'),item(exit_,'穿过','门','右')]))
    for speed,park,wait,home in [
        ('车速每秒三十厘米','停到左门前两米的位置','在那等两秒','随后返回最初的入口'),
        ('速度用0.3米/秒','离左门还剩2米就刹停','停留2秒钟','再回进棚的位置'),
        ('按每秒0.3米走','在左边大门前方二米停下','歇上两秒','开回出发的入口'),
    ]:
        rows.append(record('停靠等待返航',[item(speed,'调速'),item(park,'靠近','门','左'),
                                         item(wait,'等待'),item(home,'返回入口')]))
    return rows


def main(root, experiment):
    dest = experiment / 'data'
    dest.mkdir(parents=True, exist_ok=True)
    rng = random.Random(3189)
    original = incident()
    original['messages'][1]['content'] = '以1米/秒的速度巡检完每一条作物行，从左边第一条作物行开始'
    sets = {'test': exam(), 'paraphrase': paraphrases(), 'validation': curriculum(rng,'validation'),
            'train': [original,*curriculum(rng,'train')]}
    seen = set()
    import re
    def key(row):
        return re.sub(r'[\s，。；、,;！!]', '', row['messages'][-2]['content'])
    legacy = {split: [json.loads(line) for line in (root/'data'/f'{split}.jsonl').read_text('utf8').splitlines()]
              for split in ('test','validation','train')}
    reserved = {key(row) for values in legacy.values() for row in values}
    for split, rows in sets.items():
        unique = []
        for row in rows:
            k = key(row)
            if k not in seen and k not in reserved:
                unique.append(row)
                seen.add(k)
        if split in ('train','validation'):
            unique += rng.sample(legacy[split], 160 if split == 'train' else 16)
        (dest/f'{split}.jsonl').write_text(''.join(dump(row)+'\n' for row in unique),encoding='utf8')
        print(split, len(unique))
    (dest/'incident.jsonl').write_text(dump(original)+'\n',encoding='utf8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--experiment', type=Path, required=True)
    a = p.parse_args()
    main(a.root, a.experiment)
