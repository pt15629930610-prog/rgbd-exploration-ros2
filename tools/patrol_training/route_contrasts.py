"""Matched commands for the ROS parser: navigation, approaching a row, and patrolling it."""
import json
from pathlib import Path
import random
import sys

from patrol_curriculum import record, item, patrol, PROMPT


def build(parent, dest):
    rng=random.Random(431)
    dest.mkdir(parents=True,exist_ok=True)
    for split in ('train','validation'):
        rows=[]
        seen=set()
        for side in ['左','右','对面','']:
            for n in ['一','二','三','四']:
                for crop in ['草莓','生菜','']:
                    row=f'{side}第{n}条{crop}作物行'
                    direction={'左':'左','右':'右','对面':'前','':None}[side]
                    for action in [
                        patrol(f'巡查{row}两趟' if split=='train' else f'{row}进去巡两遍',direction,crop),
                        item(f'到{row}前一米停下' if split=='train' else f'离{row}还有一米停车',
                             '靠近','作物行',direction,crop),
                        item(f'寻找{row}' if split=='train' else f'找一下{row}', '寻找','作物行',direction,crop),
                    ]:
                        straight=rng.choice(['先直走','先直行','向前直走']) if split=='train' else '先径直往前开'
                        actions=[item(straight,'行驶'),action] if rng.random()<.6 else [action]
                        r=record('行驶与巡检',actions)
                        r['messages'][0]['content']=PROMPT
                        rows.append(r)
        rng.shuffle(rows)
        rows=rows[:144 if split=='train' else 24]
        old=[json.loads(line) for line in (parent/'data'/f'{split}.jsonl').read_text('utf8').splitlines()]
        rows+=rng.sample(old,128 if split=='train' else 24)
        output=[]
        for r in rows:
            if r['id'] not in seen:
                output.append(r)
                seen.add(r['id'])
        (dest/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output),'utf8')
        print(split,len(output))


if __name__=='__main__':
    build(Path(sys.argv[1]),Path(sys.argv[2])/'data')
