"""Check spoken speed prefixes immediately followed by a task without a comma."""

import argparse
import json
from pathlib import Path
from check_curriculum import client, coached, normalize, predict, read
from patrol_curriculum import item, patrol, start, record


def run(root, dest):
    tasks = [
        ('以0.65米/秒的速度',[patrol('把每条作物行都检查到',every=True),start('从右侧第一排开始','右')]),
        ('每秒五十厘米',[patrol('巡完左侧全部草莓行','左','草莓',True)]),
        ('按0.8米每秒',[patrol('只查右边第一条作物行','右')]),
        ('用每秒半米的速度',[patrol('巡查第二排盆栽',crop='盆栽')]),
        ('以0.6米/秒的速度',[item('开到左门前三米停车','靠近','门','左')]),
        ('按每秒二十厘米',[item('去右边的大门口停下','靠近','门','右'),item('等四秒','等待')]),
        ('以0.45米/秒的速度',[item('向左转','转向',side='左'),item('然后返回入口','返回入口')]),
        ('用每秒一米的速度',[start('从右边第三条作物行起步','右'),patrol('把整个棚的每一行都巡一趟',every=True)]),
    ]
    rows=[]
    for speed,actions in tasks:
        row=record('rate',[item(speed,'调速'),*actions])
        row['messages'][1]['content']=speed+'，'.join(a['原文'] for a in actions)
        rows.append(row)
    compact=lambda q: ''.join(c for c in q if c.isalnum())
    used={compact(r['messages'][-2]['content']) for f in (dest/'data').glob('*.jsonl*') for r in read(f)}
    assert not any(compact(r['messages'][1]['content']) in used for r in rows)
    summary={}
    with client(root,dest/'adapter/patrol-lora.gguf',18187,dest/'evaluation/rate-server.log') as api:
        with (dest/'evaluation/rate.jsonl').open('w',encoding='utf8') as out:
            for mode,method in [('raw',predict),('preview',coached)]:
                scores=[]
                for r in [*rows,*read(dest/'data/incident.jsonl')]:
                    text=r['messages'][1]['content']
                    gold=normalize(json.loads(r['messages'][2]['content']))
                    actual,error=None,None
                    try:
                        actual=normalize(method(api,text))
                    except ValueError as exc:
                        error=str(exc)
                    scores.append(actual==gold)
                    out.write(json.dumps(dict(mode=mode,query=text,expected=gold,actual=actual,error=error,
                                              correct=scores[-1]),ensure_ascii=False)+'\n')
                summary[mode]=dict(correct=sum(scores[:-1]),total=len(tasks),incident=scores[-1])
    (dest/'evaluation/rate-summary.json').write_text(json.dumps(summary,indent=2),'utf8')
    print(summary)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--experiment',required=True,type=Path)
    a=p.parse_args()
    run(a.root.resolve(),a.experiment.resolve())
