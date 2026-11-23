"""Compare the same deployed Q4 model with the adapter off/on. Never publish robot commands."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import ProxyHandler, Request, build_opener


def manual_cases():
    # Action dictionaries are independent expected plans, including numeric constraints.
    speed = lambda v: {'do': 'speed', 'speed': v}
    turn = lambda s: {'do': 'turn', 'side': s}
    row = lambda n: {'do': 'seek', 'target': 'row', 'ordinal': n}
    door = lambda s: {'do': 'seek', 'target': 'door', 'side': s}
    enter, patrol = {'do': 'enter'}, {'do': 'patrol'}
    return [
        (
            '以0.3米/秒的速度行驶，先左拐，遇到第一条作物行就进去巡一趟，走不通就退回来，再从右门出去',
            [speed(0.3), turn('left'), row(1), enter, patrol, door('right'), {'do': 'exit'}],
        ),
        (
            '速度设为每秒一米，去第二排生菜巡查两遍，再到左门前五米停下',
            [
                speed(1),
                dict(row(2), label='生菜'),
                enter,
                dict(patrol, count=2),
                door('left'),
                {'do': 'approach', 'distance': 5.0},
            ],
        ),
        (
            '先往右转，到第一个十字路口左拐',
            [turn('right'), {'do': 'seek', 'target': 'crossroad', 'ordinal': 1}, turn('left')],
        ),
        ('刚才那条行再巡两趟', [{'do': 'resume_row'}, dict(patrol, count=2)]),
        ('这行一共巡三趟', [{'do': 'resume_row'}, dict(patrol, count=3, total=True)]),
        ('巡一下左边隔壁的作物行', [{'do': 'neighbor', 'side': 'left'}, enter, patrol]),
        ('在右门前十二米停车', [door('right'), {'do': 'approach', 'distance': 12.0}]),
        (
            '离第一处十字路口还有三米就停',
            [
                {'do': 'seek', 'target': 'crossroad', 'ordinal': 1},
                {'do': 'landmark_stop', 'distance': 3.0},
            ],
        ),
        ('这行多巡几圈吧', [{'do': 'unsupported'}]),
        ('以1.2米/秒的速度行驶', [{'do': 'unsupported'}]),
    ]


def run(root, repo, port, split, cpu, calibrate=False, selected=False, replay=False):
    sys.path.insert(0, str(repo / 'src/camera_processing'))
    from camera_processing.online_language import (
        MODEL,
        PROMPT,
        SCHEMA,
        compile_semantics,
        interpret_online,
    )

    model_dir = root.parent
    logdir = root / 'evaluation'
    logdir.mkdir(exist_ok=True)
    adapter = root / 'adapter/patrol-lora.gguf'
    exe = model_dir / 'gpu-bin/llama-server.exe'
    cmd = [
        str(exe),
        '-m',
        str(model_dir / 'Qwen3VL-2B-Instruct-Q4_K_M.gguf'),
        '--lora-scaled',
        str(adapter.relative_to(model_dir)) + ':0',
        '--lora-init-without-apply',
        '-ngl',
        '0' if cpu else '99',
        '-fa',
        'off',
        '-b',
        '128',
        '-ub',
        '32',
        '-c',
        '4096',
        '-np',
        '1',
        '--alias',
        MODEL,
        '--host',
        '127.0.0.1',
        '--port',
        str(port),
    ]
    if not cpu:
        cmd.extend(['--device', 'Vulkan0'])
    if replay:
        cmd.extend(['--mmproj', str(model_dir / 'mmproj-Qwen3VL-2B-Instruct-F16.gguf')])
    opener = build_opener(ProxyHandler({}))
    url = f'http://127.0.0.1:{port}'

    def api(path, body=None):
        req = Request(
            url + path,
            None if body is None else json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'},
        )
        with opener.open(req, timeout=90) as response:
            return json.load(response)

    import socket

    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))
    rows = []
    for line in (root / 'data' / (split + '.jsonl')).read_text('utf8').splitlines():
        r = json.loads(line)
        q = r['messages'][-2]['content']
        answer = json.loads(r['messages'][-1]['content'])
        plan = compile_semantics(answer, q)
        if any(s['do'] == 'unsupported' for s in plan) and not any(
            i['动作'] == '询问' for i in answer['动作序列']
        ):
            raise ValueError('Unintended rejected reference plan: ' + q)
        rows.append((r['id'], split, q, answer, plan))
    if not calibrate:
        rows += [(str(i), 'manual', q, None, p) for i, (q, p) in enumerate(manual_cases())]
    selection = logdir / 'scale.json'
    variants = [('base', 0.0), ('patrol', 1.0)]
    if calibrate:
        variants = [(f'scale-{s:g}', s) for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
    elif selected:
        variants = [('selected', json.loads(selection.read_text())['scale'])]
    if replay:
        strict = [
            {'do': 'turn', 'side': 'left'},
            {'do': 'seek', 'target': 'row', 'ordinal': 1},
            {'do': 'enter'},
            {'do': 'loop'},
            {'do': 'seek', 'target': 'door', 'side': 'right'},
            {'do': 'exit'},
        ]
        original, plan = manual_cases()[0]
        archive = [
            ('先左拐，遇到第一条作物行，进去绕一圈，再从右门出', strict),
            ('先左拐，沿遇到的第一条作物行绕一圈，再从右门出去', strict),
            (original, plan),
            (original[1:], plan),
        ]
        rows = [(str(i), 'historical', q, None, p) for i, (q, p) in enumerate(archive)]
        variants = [('base', 0.0), ('selected', json.loads(selection.read_text())['scale'])]
    with (logdir / 'server.log').open('wb') as log:
        server = subprocess.Popen(
            cmd, cwd=model_dir, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW
        )
        try:
            for _ in range(150):
                if server.poll() is not None:
                    raise RuntimeError('Server quit, check ' + str(logdir / 'server.log'))
                try:
                    if api('/health').get('status') == 'ok':
                        break
                except OSError:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError('Model load timed out')
            adapters = api('/lora-adapters')
            if not any(a['id'] == 0 and a['scale'] == 0 for a in adapters):
                raise RuntimeError('Server did not load a disabled adapter: ' + str(adapters))
            summary = {}
            for variant, scale in variants:
                counts = {record[1]: Counter() for record in rows}
                name = 'historical' if replay else split
                with (logdir / f'{name}-{variant}.jsonl').open('w', encoding='utf8') as out:
                    for uid, group, q, answer, plan in rows:
                        payload = {
                            'model': MODEL,
                            'temperature': 0,
                            'seed': 731,
                            'max_tokens': 600,
                            'lora': [{'id': 0, 'scale': scale}],
                            'response_format': {
                                'type': 'json_schema',
                                'json_schema': {'name': 'result', 'schema': SCHEMA},
                            },
                            'messages': [
                                {'role': 'system', 'content': PROMPT},
                                {'role': 'user', 'content': q},
                            ],
                        }
                        raw = ''
                        reply, actual = None, []
                        error = None
                        try:
                            if replay:
                                trace = []
                                os.environ['GREENHOUSE_PATROL_LORA'] = 'true' if scale else 'false'
                                os.environ['GREENHOUSE_PATROL_LORA_SCALE'] = str(scale)
                                actual = interpret_online(url, q, trace)
                                raw = '\n'.join(trace)
                            else:
                                raw = api('/v1/chat/completions', payload)['choices'][0]['message'][
                                    'content'
                                ]
                                reply = json.loads(raw)
                                actual = compile_semantics(reply, q)
                        except (ValueError, TypeError, KeyError) as exc:
                            error = str(exc)
                        canonical = lambda s: [
                            {'do': 'unsupported'} if a['do'] == 'unsupported' else a for a in s
                        ]
                        exact = canonical(actual) == canonical(plan)
                        count = counts[group]
                        count['cases'] += 1
                        count['exact_plan'] += int(exact)
                        if answer is not None:
                            count['exact_labels'] += int(reply == answer)
                        record = dict(
                            id=uid,
                            group=group,
                            query=q,
                            gold=answer,
                            raw=raw,
                            expected=plan,
                            actual=actual,
                            exact_plan=exact,
                            error=error,
                        )
                        out.write(json.dumps(record, ensure_ascii=False) + '\n')
                        out.flush()
                summary[variant] = counts
                print(variant, counts, flush=True)
            label = 'calibration' if calibrate else ('selected' if selected else 'summary')
            (logdir / f'{name}-{label}.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
            if calibrate:
                best = max(
                    variants,
                    key=lambda v: (
                        summary[v[0]][split]['exact_plan'],
                        summary[v[0]][split]['exact_labels'],
                        -v[1],
                    ),
                )
                selection.write_text(
                    json.dumps({'scale': best[1], 'split': split, 'scores': summary}, indent=2)
                )
        finally:
            server.terminate()
            server.wait(timeout=30)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--port', type=int, default=18184)
    p.add_argument('--split', choices=('validation', 'test'), default='test')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--calibrate', action='store_true', help='Select strength only on validation plans')
    p.add_argument('--selected', action='store_true', help='Evaluate a previously selected strength')
    p.add_argument(
        '--replay', action='store_true', help='Replay four archived instructions through the live parser'
    )
    a = p.parse_args()
    if a.calibrate and (a.split != 'validation' or a.selected):
        p.error('calibration requires --split validation and excludes --selected')
    if a.replay and (a.calibrate or a.selected):
        p.error('use --replay independently')
    run(a.root, a.repo, a.port, a.split, a.cpu, a.calibrate, a.selected, a.replay)
