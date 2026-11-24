"""Offline test of Q4 decoding with old/new LoRA, and an optional interactive preview."""

import argparse
from collections import defaultdict
from contextlib import contextmanager, ExitStack
import errno
import json
from pathlib import Path
import re
import socket
import subprocess
import time
from urllib.request import ProxyHandler, Request, build_opener

from patrol_curriculum import PROMPT_V2, SCHEMA_V2, item, patrol, record, start
from camera_processing.online_language import MODEL, NUMBER, PROMPT, SCHEMA, compile_semantics, measurements


def read(path):
    return [json.loads(line) for line in path.read_text('utf8').splitlines()]


@contextmanager
def client(root, adapter, port, log, shared=False):
    directory = root.parent
    opener = build_opener(ProxyHandler({}))
    def api(path, body=None, timeout=120):
        request = Request(f'http://127.0.0.1:{port}'+path,
                          json.dumps(body).encode() if body else None, {'Content-Type':'application/json'})
        with opener.open(request,timeout=timeout) as resp:
            return json.load(resp)
    try:
        with socket.socket() as probe:
            probe.bind(('127.0.0.1',port))
    except OSError as exc:
        if not shared or exc.errno not in (errno.EADDRINUSE, errno.EACCES, 10048, 10013):
            raise
        try:
            adapters = api('/lora-adapters',timeout=3)
            matches = any(a.get('id')==0 and a.get('path') and
                          (directory/a['path']).resolve()==adapter.resolve() for a in adapters)
            matches = matches and any(m.get('id')==MODEL for m in api('/v1/models',timeout=3)['data'])
        except (OSError,ValueError,TypeError,AttributeError,KeyError):
            matches = False
        if matches:
            print(f'已连接本机巡检模型（端口 {port}）。')
            yield api
            return
        with socket.socket() as probe:
            probe.bind(('127.0.0.1',0))
            port = probe.getsockname()[1]
        log = log.with_name(f'{log.stem}-{port}.log')
        print(f'默认端口已被占用，本次预览使用 {port}。')
    cmd = [str(directory/'gpu-bin/llama-server.exe'), '-m', 'Qwen3VL-2B-Instruct-Q4_K_M.gguf',
           '--lora-scaled',str(adapter.relative_to(directory))+':0','--lora-init-without-apply',
           '--device','Vulkan0','-ngl','99','-fa','off','-b','128','-ub','32','-c','4096',
           '-np','1','--alias',MODEL,'--host','127.0.0.1','--port',str(port)]
    with log.open('wb') as stream:
        process = subprocess.Popen(cmd,cwd=directory,stdout=stream,stderr=stream,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError(f'Server failed. See {log}')
                try:
                    if api('/health').get('status') == 'ok':
                        break
                except OSError:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError('Server startup timeout')
            if not any(a['id']==0 for a in api('/lora-adapters')):
                raise RuntimeError('Adapter missing')
            yield api
        finally:
            process.terminate()
            process.wait(timeout=30)


def predict(api, text, legacy=False, examples=(), feedback=''):
    messages = [dict(role='system',content=(PROMPT if legacy else PROMPT_V2) + feedback)]
    for row in examples:
        messages.extend(row['messages'][1:])
    messages.append(dict(role='user',content=text))
    response = api('/v1/chat/completions', dict(model=MODEL, temperature=0, seed=731, max_tokens=600,
                   lora=[dict(id=0,scale=0.75)],
                   messages=messages,
                   response_format=dict(type='json_schema',json_schema=dict(name='result',
                                                                           schema=SCHEMA if legacy else SCHEMA_V2))))
    return json.loads(response['choices'][0]['message']['content'])


def coached(api, text, mark_rates=True):
    examples = [
        record('', [item('每秒0.45米','调速'),patrol('查完所有种植行',every=True),
                    start('右边第三排先查','右')], ''),
        record('', [patrol('给我查第五排黄瓜',crop='黄瓜')]),
        record('', [patrol('各排番茄都给我巡到',crop='番茄',every=True)]),
        record('', [item('先向右转','转向',side='右'),patrol('第一条作物行巡一次'),
                    item('前方不通则返回行口','遇阻退回'),item('再过左门出棚','穿过','门','左')]),
        record('', [item('禁止向右转','未支持')]),
        record('', [item('去隔壁行巡视一下','询问')]),
    ]
    feedback = '\n每个动作只引用属于自己的文字；不能借用上一动作的方位。调速片段到速度表述结束处为止。'
    presented = text
    for attempt in range(2):
        reply = predict(api,presented,examples=examples,feedback=feedback)
        actions = normalize(reply)
        compact = lambda q: re.sub(r'[\s，,。；;、！!]', '', q)
        errors = []
        if ''.join(a['原文'] for a in actions) != compact(text):
            errors.append('把整条原话按语义连续分段引用，每个字出现一次，不能重叠或漏字')
        for a in actions:
            q = a['原文']
            if a.get('方位') in ('左','右') and a['方位'] not in q:
                errors.append('删除本段原话未给出的左右方位')
            if a.get('范围') and a['动作']!='巡检':
                errors.append('范围只能绑定到巡检')
            if a.get('物体名称') and a['物体名称'] not in q:
                errors.append('物体名称必须出现在本段原话')
            if a['动作']=='调速' and ('speed' not in measurements(q) or
                                     re.search('巡|查|作物|路口|转弯|停车|种植',q)):
                errors.append('将速度与任务分别标注，巡检要求另起一项')
        if any(a['动作']=='指定起始行' for a in actions) and not any(a['动作']=='巡检' for a in actions):
            errors.append('指定起始行必须有对应的巡检任务')
        if not errors:
            return reply
        # A measurement has a lexical boundary even when the operator omits punctuation.
        # Only annotate a positive initial rate, leaving conditions/negations and task words intact.
        unit = r'(?:厘米|毫米|米|cm|mm|m)'
        rate = rf'(?:每秒\s*{NUMBER}\s*{unit}|{NUMBER}\s*{unit}\s*(?:/\s*(?:秒|s)|每秒))'
        header = re.match(rf'(?:以|按|用)?{rate}(?:的速度)?',text)
        if mark_rates and header and header.end()<len(text) and text[header.end()].isalnum():
            presented = text[:header.end()]+'，'+text[header.end():]
        feedback += '\n上次输出有误。' + '。'.join(dict.fromkeys(errors))
    raise ValueError(feedback)


def preview(api, text):
    reply = coached(api,text)
    actions = reply['动作序列']
    allow_start = sum(a['动作']=='巡检' for a in actions)>1 and not any(a['动作']=='指定起始行' for a in actions)
    choices = ['各条符合条件的行','选中的一行','信息不足']
    schema = {'type':'object','properties':{'覆盖行':{'enum':choices}},
              'required':['覆盖行'],'additionalProperties':False}
    for action in reply['动作序列']:
        if action['动作'] != '巡检':
            continue
        classify_start = allow_start and not any(a['动作']=='指定起始行' for a in actions)
        schema['properties']['覆盖行']['enum'] = choices + (['整次任务的起始行'] if classify_start else [])
        prompt = ('结合整句判断当前片段的作用。遍历所有符合限定条件的行选“各条符合条件的行”；'
                  '进指定的一排、一条或同一行重复几趟选“选中的一行”。缺少可确定的要求选“信息不足”。'
                  '指定起点只决定先后顺序。请集中判断所给片段。')
        if classify_start:
            prompt += '如果本段交代巡查从哪一行做起，而另一个片段规定巡查哪些行，本段选“整次任务的起始行”。先巡一行再巡另一行属于两个独立任务。'
        response=api('/v1/chat/completions',dict(model=MODEL,temperature=0,seed=731,max_tokens=80,
                     lora=[dict(id=0,scale=0.75)],response_format=dict(type='json_schema',
                     json_schema=dict(name='coverage',schema=schema)),messages=[
                         dict(role='system',content=prompt),
                         dict(role='user',content='整句：'+text+'\n巡检片段：'+action['原文'])]))
        scope=json.loads(response['choices'][0]['message']['content'])['覆盖行']
        quote=action['原文']
        # Bind explicit quantifiers within this clause only; the model handles implicit coverage.
        universal=re.search(r'(?:所有(?:的)?|全部(?:的)?|每(?:一)?(?:条|排|行))[^，。；]{0,8}(?:行|作物|种植|菜)',quote)
        single=re.search(r'(?:只|仅)(?:需|要)?(?:巡检|巡查|巡视|巡|检查|看)[^，。；]{0,12}(?:第.+?|这|那|一)(?:条|排|行)',quote)
        if scope!='整次任务的起始行' and not re.search(r'不|别|除|如果|要是',quote):
            if universal and not single:
                scope='各条符合条件的行'
            elif single and not universal:
                scope='选中的一行'
        if scope=='信息不足':
            raise ValueError('模型未能确认巡检范围：'+action['原文'])
        if scope=='整次任务的起始行':
            action.update(动作='指定起始行',目标='作物行')
            action.pop('范围',None)
        elif scope=='各条符合条件的行':
            action['范围']='全部'
        else:
            action.pop('范围',None)
    for action in actions:
        if action['动作']=='指定起始行':
            sides=set(re.findall('[左右]',action['原文']))
            if len(sides)==1:
                action['方位']=sides.pop()
    return reply


def normalize(answer):
    # Only ignore punctuation and whitespace. Retain order, scope, references, numbers and units.
    return [dict(action, 原文=re.sub(r'[\s，,。；;、！!]', '', action['原文'])) for action in answer['动作序列']]


def plan(answer, text):
    return [s if s['do'] != 'unsupported' else {'do':'unsupported'} for s in compile_semantics(answer,text)]


def run(root, experiment, variant, port, interactive):
    root, experiment = root.resolve(), experiment.resolve()
    dest = experiment/'evaluation'
    dest.mkdir(exist_ok=True)
    adapter = (root if variant == 'previous' else experiment)/'adapter/patrol-lora.gguf'
    log = dest/f'{variant}-server.log'
    if interactive:
        with ExitStack() as session:
            api = session.enter_context(client(root,adapter,port,log,shared=True))
            while True:
                try:
                    text = input('巡检指令（回车退出）> ').strip()
                except (EOFError,KeyboardInterrupt):
                    break
                if not text:
                    break
                try:
                    try:
                        result = preview(api,text)
                    except OSError:
                        session.close()
                        api = session.enter_context(client(root,adapter,port,log,shared=True))
                        result = preview(api,text)
                    print(json.dumps(result,ensure_ascii=False,indent=2))
                except ValueError as exc:
                    print('解析未通过完整性校验。', exc)
            return
    with client(root,adapter,port,log) as api:
        rows = [('contrast',r) for r in read(experiment/'data/test.jsonl')]
        rows += [('paraphrase',r) for r in read(experiment/'data/paraphrase.jsonl')]
        rows += [('incident',r) for r in read(experiment/'data/incident.jsonl')]
        rows += [('legacy',r) for r in read(root/'data/test.jsonl')]
        scores = defaultdict(list)
        pairs = defaultdict(list)
        with (dest/f'{variant}.jsonl').open('w',encoding='utf8') as stream:
            for group,row in rows:
                query = row['messages'][-2]['content']
                expected = json.loads(row['messages'][-1]['content'])
                gold = plan(expected,query) if group=='legacy' else normalize(expected)
                actual, reply, error = None, None, None
                try:
                    reply = predict(api,query,group=='legacy')
                    actual = plan(reply,query) if group=='legacy' else normalize(reply)
                except (ValueError, KeyError, TypeError) as exc:
                    error = str(exc)
                correct = gold==actual
                scores[group].append(correct)
                if group=='contrast':
                    scores[row['family']].append(correct)
                    pairs[row['pair']].append(correct)
                elif group=='paraphrase':
                    scores[row['family']].append(correct)
                stream.write(json.dumps(dict(group=group,query=query,expected=gold,reply=reply,
                                             actual=actual,correct=correct,error=error),ensure_ascii=False)+'\n')
                stream.flush()
        scores['pairs']=[len(values)==2 and all(values) for values in pairs.values()]
        summary = {k:[sum(values),len(values)] for k,values in scores.items()}
        (dest/f'{variant}.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf8')
        print(summary)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--variant',choices=['previous','continued'],default='continued')
    p.add_argument('--port',type=int,default=18187)
    p.add_argument('--interactive',action='store_true')
    a=p.parse_args()
    run(a.root,a.experiment,a.variant,a.port,a.interactive)
