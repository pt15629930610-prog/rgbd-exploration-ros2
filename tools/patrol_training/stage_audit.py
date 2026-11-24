"""Observe one real preview pass per question; preserve model and preview code."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch


def cases(p):
    rows = []
    def add(group, actions, meaning=''):
        rows.append((group, '，'.join(a['原文'] for a in actions), actions, meaning))
    move=lambda q:p.item(q,'行驶')
    turn=lambda q:p.item(q,'转向','十字路口','右')
    leave=lambda q:p.item(q,'穿过','门','右')
    add('单行',[move('先直走'),turn('遇到十字路口右拐'),p.patrol('遇到对面第二条行作物进去巡检','前')])
    for q,side in [('巡第二条作物行',None),('只巡第二条作物行',None),('巡第二条作物行两遍',None),
                   ('巡对面的第二条作物行','前'),('左侧头一条作物行巡一趟','左')]:
        add('单行',[p.patrol(q,side)])
    for q,meaning in [('巡两条作物行','两行'),('头两排作物都查一下','前两行'),
                      ('挑三条种植行各巡一遍','任选三行'),('巡第一条和第三条作物行','第1、3行'),
                      ('从第二排起连续巡三排作物','第2至4排'),('巡两条作物行，每条走两遍','两行，各两遍')]:
        add('多行',[p.patrol(q)],meaning)
    for q,side,crop in [('巡遍棚内每条种植行',None,None),('所有草莓行都巡检一遍',None,'草莓'),
                        ('左侧每一条作物行都巡一下','左',None),('右侧的作物行全部巡完','右',None),
                        ('把各个作物行都巡视到',None,None)]:
        add('全部',[p.patrol(q,side,crop,every=True)])
    add('全部',[p.start('从左边遇到的第一条行作物开始巡检','左'),p.patrol('把所有的行作物都巡检一遍',every=True),leave('最后从右边的门出去')])
    for actions in [[move('先直走')],[move('先直走'),turn('遇到十字路口右拐')],
                    [p.item('在右门前两米停下','靠近','门','右')],
                    [p.item('回到入口','返回入口'),p.item('等两秒','等待')],
                    [p.item('在第二条作物行前一米停下','靠近','作物行')],
                    [p.item('给我拍一张照','拍照'),leave('从右门出去')]]:
        add('无巡检',actions)
    return rows


def main(exp):
    sys.path.insert(0,str(exp/'source/tools/patrol_training'))
    import check_curriculum as c
    rows=cases(c)
    out=exp/'evaluation'/('stages-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    out.mkdir(parents=True)
    (out/'cases.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),'utf8')
    counts=defaultdict(Counter)
    adapter=exp/'adapter/patrol-lora.gguf'
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    with c.client(exp.parent,adapter,18187,out/'server.log',shared=True) as api, (out/'stages.jsonl').open('w',encoding='utf8') as file:
        for group,q,gold,meaning in rows:
            record=dict(group=group,query=q,expected=gold,meaning=meaning,calls=[])
            def traced(path,body):
                res=api(path,body)
                name=body['response_format']['json_schema']['name']
                answer=json.loads(res['choices'][0]['message']['content'])
                record['calls'].append(dict(type=name,reply=answer))
                if name=='result' and 'first' not in record:
                    record['first']=answer
                return res
            original=c.coached
            def validated(*args,**kwargs):
                reply=original(*args,**kwargs)
                record['checked']=deepcopy(reply)
                return reply
            try:
                with patch.object(c,'coached',validated):
                    record['final']=c.preview(traced,q)
            except ValueError as exc:
                record['error']=str(exc)
            for stage in ('first','checked','final'):
                answer=record.get(stage)
                count=counts[group+'/'+stage]
                count['n']+=1
                if answer is None:
                    count['unavailable']+=1
                    continue
                actions=c.normalize(answer)
                count['exact']+=int(actions==c.normalize({'动作序列':gold}))
                count['all']+=int(any(a.get('范围')=='全部' for a in actions))
                count['row_task']+=int(any(a['动作'] in ('巡检','指定起始行','绕行') for a in actions))
            file.write(json.dumps(record,ensure_ascii=False)+'\n')
            file.flush()
    summary=dict(scores=counts,adapter_sha256=digest(adapter),parser_sha256=digest(Path(c.__file__)))
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf8')
    print(out)
    print(json.dumps(counts,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('experiment',type=Path)
    main(parser.parse_args().experiment.resolve())
