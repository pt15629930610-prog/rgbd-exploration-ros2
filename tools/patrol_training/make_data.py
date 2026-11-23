"""Generate auditable task labels and keep paraphrase families separated by split."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def item(text, action, target=None, side=None, crop=None):
    result = {'原文': text, '动作': action}
    for key, value in [('目标', target), ('方位', side), ('物体名称', crop)]:
        if value:
            result[key] = value
    return result


def phrases(split, rng):
    direction = rng.choice(['左', '右'])
    crop = rng.choice(['草莓', '生菜', '番茄', '黄瓜', '盆栽'])
    ordinal = rng.choice(['第一', '第二', '第三'])
    n = rng.choice(['一', '两', '三', '四'])
    speed = rng.choice(['0.05', '0.1', '0.25', '0.3', '0.55', '0.8', '1'])
    distance = rng.choice(['0.8', '1', '1.5', '2', '3.5', '5', '8', '12', '25', '50'])
    side = rng.choice(['左', '右'])
    door = rng.choice(['左', '右', '对面'])
    ds = '前' if door == '对面' else door
    # Each column has different surface forms. Values may recur across splits.
    col = ['train', 'validation', 'test'].index(split)

    def pick(*forms):
        return forms[col]

    speed_q = pick(f'以{speed}米/秒的速度行驶', f'速度设为每秒{speed}米', f'限速{speed}米每秒')
    turn_q = pick(f'先朝{direction}边拐', f'起步向{direction}转', f'先把车头转向{direction}侧')
    cross_q = pick(
        f'遇到{ordinal}个十字路口{direction}拐',
        f'在{ordinal}处十字口向{direction}转',
        f'沿路到{ordinal}个四岔口再往{direction}拐',
    )
    open_q = pick(
        f'遇到{ordinal}个路口{direction}转',
        f'在{ordinal}个岔口向{direction}拐',
        f'见到{ordinal}条支路时朝{direction}拐',
    )
    row_q = pick(
        f'遇到{ordinal}条{crop}作物行进去巡{n}趟',
        f'巡检{ordinal}排{crop}，一共{n}遍',
        f'{ordinal}条{crop}种植行到了就进去巡查{n}次',
    )
    add_q = pick(f'这条行再巡{n}圈', f'在刚才那条行追加{n}趟巡检', f'原来的行加巡{n}次')
    adjacent_q = pick(
        f'去{side}边隔壁的作物行巡一趟',
        f'换到{side}侧相邻的作物行巡检一次',
        f'到{side}手边紧挨着的下一行巡一遍',
    )
    blocked_q = pick('走不通就退回来', '如果遇到封堵就原路退回行口', '前方被堵就退回本行行口')
    park_q = pick(
        f'在{door}门前{distance}米停下',
        f'到{door}的门前{distance}米处停车',
        f'离{door}的门还有{distance}米时就停',
    )
    if split == 'train':
        speed_q = rng.choice([speed_q, f'速度别超过{speed}米每秒', f'每秒{speed}米行驶'])
        turn_q = rng.choice([turn_q, f'先{direction}转', f'一出发就往{direction}拐'])
        row_q = rng.choice(
            [row_q, f'去{ordinal}排{crop}巡查{n}遍', f'沿{ordinal}条{crop}作物行巡检{n}圈']
        )
        add_q = rng.choice([add_q, f'同一行多巡{n}遍', f'这行一共巡{n}趟'])
        park_q = rng.choice(
            [park_q, f'停在{door}门前方{distance}米', f'在离{door}门{distance}米的位置停车']
        )
    return {
        'speed': item(speed_q, '调速'),
        'turn': item(turn_q, '转向', side=direction),
        'cross': item(cross_q, '转向', '十字路口', direction),
        'opening': item(open_q, '转向', '通道开口', direction),
        'row': item(row_q, '巡检', '作物行', crop=crop),
        'add': item(add_q, '巡检', '作物行'),
        'adjacent': item(adjacent_q, '巡检', '作物行', side),
        'blocked': item(blocked_q, '遇阻退回'),
        'park': item(park_q, '靠近', '门', ds),
        'junctionstop': item(
            pick(
                f'在{ordinal}个十字口前{distance}米停车',
                f'停在{ordinal}个十字路口前方{distance}米',
                f'距{ordinal}个四岔口{distance}米处停下',
            ),
            '靠近',
            '十字路口',
        ),
        'cropstop': item(
            pick(
                f'在{crop}前{distance}米停车',
                f'靠近{crop}，相隔{distance}米停下',
                f'离那株{crop}还有{distance}米就停',
            ),
            '靠近',
            '作物',
            crop=crop,
        ),
        'home': item(pick('回大棚入口', '返回最早的入口', '返航到原来的出发点'), '返回入口'),
        'exit': item(
            pick(f'从{door}的门出去', f'穿过{door}的门出棚', f'最后经{door}门驶出温室'), '穿过', '门', ds
        ),
        'vague': item(pick('隔壁那行也巡一下', '换到旁边那条行转转', '还要巡紧挨着的那一行'), '询问'),
        'many': item(pick('再多巡几圈', '这行追加几趟', '多巡几遍吧'), '询问'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo / 'src/camera_processing'))
    from camera_processing.online_language import PROMPT, compile_semantics

    dest = args.root / 'data'
    dest.mkdir(parents=True, exist_ok=True)
    all_text = set()
    for index, (split, count) in enumerate([('test', 48), ('validation', 32), ('train', 320)]):
        rng = random.Random(771 + index)
        patterns = [
            'speed turn row blocked exit',
            'cross row park',
            'speed opening row home',
            'row adjacent park',
            'row add exit',
            'speed row blocked cropstop',
            'row park',
            'speed park',
            'turn row',
            'row many',
            'row vague',
            'row',
            'turn',
            'park',
            'cropstop',
            'cross row exit',
            'speed junctionstop',
        ]
        rows = []
        for pattern in itertools.cycle(patterns):
            pool = phrases(split, rng)
            labels = [pool[k] for k in pattern.split()]
            turns = []
            if 'add' in pattern or 'adjacent' in pattern or 'many' in pattern or 'vague' in pattern:
                # Explicit past exchange anchors the referenced row in the training dialogue.
                first = labels.pop(0)
                turns = [
                    {'role': 'user', 'content': first['原文']},
                    {'role': 'assistant', 'content': dump({'动作序列': [first]})},
                ]
            query = '，'.join(a['原文'] for a in labels)
            key = dump(turns) + query
            if key in all_text:
                continue
            all_text.add(key)
            messages = [
                {'role': 'system', 'content': PROMPT},
                *turns,
                {'role': 'user', 'content': query},
                {'role': 'assistant', 'content': dump({'动作序列': labels})},
            ]
            rows.append(
                {
                    'id': hashlib.sha256(key.encode()).hexdigest()[:16],
                    'family': pattern,
                    'messages': messages,
                }
            )
            if len(rows) == count:
                break
        (dest / (split + '.jsonl')).write_text(''.join(dump(r) + '\n' for r in rows), encoding='utf8')
        print(split, len(rows))
    source = dest / 'my_examples.jsonl'
    if source.exists():
        own = []
        reserved = {
            json.loads(line)['messages'][-2]['content']
            for split in ('test', 'validation', 'train')
            for line in (dest / (split + '.jsonl')).read_text('utf8').splitlines()
        }
        for line, value in enumerate(source.read_text('utf-8-sig').splitlines(), 1):
            if not value.strip():
                continue
            record = json.loads(value)
            query = record['指令']
            if query in reserved:
                raise ValueError(f'第 {line} 行与现有指令重复：{query}')
            answer = {'动作序列': record['动作序列']}
            plan = compile_semantics(answer, query)
            if any(s['do'] == 'unsupported' for s in plan) and not any(
                a['动作'] in ('询问', '未支持') for a in answer['动作序列']
            ):
                raise ValueError(f'第 {line} 行无法执行：{plan}')
            reserved.add(query)
            own.append(
                {
                    'id': 'my-' + hashlib.sha256(query.encode()).hexdigest()[:16],
                    'family': 'personal',
                    'messages': [
                        {'role': 'system', 'content': PROMPT},
                        {'role': 'user', 'content': query},
                        {'role': 'assistant', 'content': dump(answer)},
                    ],
                }
            )
        output = dest / 'personal_train.jsonl'
        temp = output.with_suffix('.tmp')
        temp.write_text(''.join(dump(r) + '\n' for r in own), encoding='utf8')
        temp.replace(output)
        print('personal train', len(own))


if __name__ == '__main__':
    main()
