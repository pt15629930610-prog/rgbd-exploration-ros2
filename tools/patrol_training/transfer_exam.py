"""Fresh transfer cases for the frozen few-shot parser. Never send motion commands."""

import argparse
import json
from pathlib import Path

from patrol_curriculum import item, patrol, start, record
from check_curriculum import client, coached, predict, preview, read, normalize


def cases():
    examples = [
        [item('速度定成每秒八十厘米','调速'),patrol('所有作物行要逐条巡视',every=True),
         start('最先去右侧第二条种植行','右')],
        [start('把左边第三排放最先','左'),patrol('全棚各条种植行依次巡视',every=True)],
        [patrol('全棚各条种植行依次巡视',every=True),start('把左边第三排放最先','左')],
        [patrol('仅检查右侧第二排草莓','右','草莓')],
        [patrol('右侧每排草莓都检查一下','右','草莓',True)],
        [patrol('每一行盆栽都帮我走访一次',crop='盆栽',every=True)],
        [patrol('头一排盆栽帮我走访一次',crop='盆栽')],
        [item('向右转过去','转向',side='右'),patrol('遇上第二排作物就进去查一遍')],
        [item('在第二个十字路口向右转过去','转向','十字路口','右')],
        [item('通过第二个十字路口以后再向右转过去','转向','十字路口','右') |
         {'转弯时机':'通过路口后'}],
        [patrol('刚刚巡的这行一共走查三遍')],
        [patrol('刚刚巡的这行另外走查三遍')],
        [item('旁边的作物行也帮我走一次','询问')],
        [patrol('右边相邻的作物行也帮我走一次','右')],
        [item('开到正前方的门前四米停住','靠近','门','前'),item('原地等十秒','等待')],
        [item('等足十秒钟','等待'),item('穿过正前方的大门出棚','穿过','门','前')],
        [item('不要去前门','未支持')],
        [item('看到虫害才给我拍照','未支持')],
        [patrol('第一排黄瓜巡查一遍',crop='黄瓜'),item('遇到封堵就原路退回行口','遇阻退回'),
         item('再往左门外开出去','穿过','门','左')],
        [item('按每秒四十厘米行车','调速'),patrol('第一排生菜进去查两遍',crop='生菜'),
         item('在右门前半米停车','靠近','门','右'),item('等五秒钟','等待'),item('回最初的入口','返回入口')],
    ]
    return [record('transfer',labels) for labels in examples]


def main(root, experiment):
    dest=experiment/'evaluation'
    fresh=cases()
    compact=lambda q: ''.join(c for c in q if c.isalnum())
    used={compact(r['messages'][-2]['content']) for file in (experiment/'data').glob('*.jsonl')
          for r in read(file)}
    assert not any(compact(row['messages'][1]['content']) in used for row in fresh)
    (experiment/'data/transfer.jsonl.locked').write_text(
        ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in fresh),'utf8')
    summary={}
    for version, folder in [('previous',root),('continued',experiment)]:
        with client(root,folder/'adapter/patrol-lora.gguf',18187,dest/f'transfer-{version}.log') as api:
            with (dest/f'transfer-{version}.jsonl').open('w',encoding='utf8') as stream:
                for mode,method in [('raw',predict),('assisted',lambda api,q: coached(api,q,mark_rates=False))]:
                    result=[]
                    for row in fresh+read(experiment/'data/incident.jsonl'):
                        q=row['messages'][1]['content']
                        gold=normalize(json.loads(row['messages'][2]['content']))
                        actual=None
                        error=None
                        try:
                            actual=normalize(method(api,q))
                        except ValueError as exc:
                            error=str(exc)
                        result.append(actual==gold)
                        stream.write(json.dumps(dict(mode=mode,query=q,expected=gold,actual=actual,
                                                     correct=result[-1],error=error),ensure_ascii=False)+'\n')
                        stream.flush()
                    summary[version+'-'+mode]=dict(correct=sum(result[:-1]),total=len(fresh),incident=result[-1])
                    print(version,mode,summary[version+'-'+mode],flush=True)
    (dest/'transfer-summary.json').write_text(json.dumps(summary,indent=2),'utf8')


def preview_exam(root, experiment):
    fresh=[
        [patrol('数得出来的作物行挨排过一趟',every=True)],
        [patrol('右侧每一排菜都进去转转','右',every=True)],
        [patrol('右侧这一排菜进去转转','右')],
        [patrol('不管有多少排，把全棚的作物行逐排巡视',every=True)],
        [patrol('只看第三排番茄这一行',crop='番茄')],
        [start('右手第一行当作第一站','右'),patrol('所有作物行依次检查',every=True)],
        [patrol('温室的种植行请一行一行巡到',every=True),start('用左侧第二行打头','左')],
        [item('以0.85米/秒的速度','调速'),patrol('巡遍每排黄瓜',crop='黄瓜',every=True)],
        [item('每秒四十五厘米','调速'),patrol('只巡左侧头一条种植行','左')],
        [item('开到右门前六米停住','靠近','门','右'),item('再拍照','拍照')],
        [item('给我拍照','拍照'),item('再开到右门前六米停住','靠近','门','右')],
        [item('未看到花就别拍照','未支持')],
    ]
    rows=[record('final',actions) for actions in fresh]
    for row in rows[7:9]:
        row['messages'][1]['content']=row['messages'][1]['content'].replace('，','',1)
    compact=lambda q: ''.join(c for c in q if c.isalnum())
    used={compact(r['messages'][-2]['content']) for f in (experiment/'data').glob('*.jsonl*')
          for r in read(f)}
    assert not any(compact(row['messages'][1]['content']) in used for row in rows)
    dest=experiment/'evaluation'
    summary={}
    with client(root,experiment/'adapter/patrol-lora.gguf',18187,dest/'final-server.log') as api:
        with (dest/'final.jsonl').open('w',encoding='utf8') as stream:
            for mode,method in [('raw',predict),('preview',preview)]:
                flags=[]
                for row in rows+read(experiment/'data/incident.jsonl'):
                    text=row['messages'][1]['content']
                    gold=normalize(json.loads(row['messages'][2]['content']))
                    actual,error=None,None
                    try:
                        actual=normalize(method(api,text))
                    except ValueError as exc:
                        error=str(exc)
                    flags.append(actual==gold)
                    stream.write(json.dumps(dict(mode=mode,query=text,expected=gold,actual=actual,
                                                 correct=flags[-1],error=error),ensure_ascii=False)+'\n')
                    stream.flush()
                summary[mode]=dict(correct=sum(flags[:-1]),total=len(rows),incident=flags[-1])
    (dest/'final-summary.json').write_text(json.dumps(summary,indent=2),'utf8')
    print(summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--preview',action='store_true')
    args=parser.parse_args()
    (preview_exam if args.preview else main)(args.root.resolve(),args.experiment.resolve())
