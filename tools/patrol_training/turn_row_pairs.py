"""Build ordered turn/traverse minimal pairs and keep a sample of v4 replay data."""

import argparse
import json
from pathlib import Path
from random import Random

from camera_processing.online_language import PROMPT, compile_semantics


def examples(split, variants):
    for index, (row, cross, connector) in enumerate(variants):
        for side in ('左', '右'):
            turn = {'原文': side+'拐', '动作': '转向', '方位': side}
            traverse = {'原文': cross+row, '动作': '穿过', '目标': '作物行'}
            for reverse in (False, True):
                first, second = (traverse, turn) if reverse else (turn, traverse)
                first = dict(first, 原文='先'+first['原文'])
                second = dict(second, 原文=connector+second['原文'])
                query = first['原文']+second['原文']
                label = {'动作序列': [first, second]}
                steps = compile_semantics(label, query)
                expected = ['seek','enter','traverse_row','turn'] if reverse else [
                    'turn','seek','enter','traverse_row']
                assert [step['do'] for step in steps] == expected, query
                yield {'id': f'order-{split}-{index}-{side}-{int(reverse)}',
                       'family': 'turn-row-order', 'messages': [
                           {'role':'system','content':PROMPT}, {'role':'user','content':query},
                           {'role':'assistant','content':json.dumps(label,ensure_ascii=False)}]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('previous', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    output = args.destination/'data'
    output.mkdir(parents=True, exist_ok=False)
    variants = {'train': [('第一条草莓作物行','穿过','再'), ('第三条作物行','穿越',''),
                          ('中间那条行作物','穿过','接着'), ('最后一条种植行','穿过','然后'),
                          ('第二条生菜作物行','穿越','随后'), ('最开始那条作物行','穿过','')],
                'validation': [('最后一条行作物','穿过',''), ('中间那条种植行','穿越','再')]}
    for split, phrases in variants.items():
        old = (args.previous/'data'/(split+'.jsonl')).read_text(encoding='utf8').splitlines()
        rows = [json.dumps(row, ensure_ascii=False) for row in examples(split, phrases)]
        rows.extend(Random(731).sample(old, 24 if split == 'train' else 8))
        (output/(split+'.jsonl')).write_text('\n'.join(rows)+'\n', encoding='utf8')
