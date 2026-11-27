"""Compare raw responses on held-out pairs and replay the user's complete route."""

import argparse
import json
import os
from pathlib import Path
import tempfile

from check_curriculum import client
from camera_processing import online_language as lang


def evaluate(root, experiment, adapter, port):
    rows = [json.loads(s) for s in (experiment/'data/validation.jsonl').read_text('utf8').splitlines()]
    os.environ.update(GREENHOUSE_PATROL_LORA='true', GREENHOUSE_PATROL_LORA_SCALE='0.75')
    server = f'http://127.0.0.1:{port}'
    with client(root, adapter, port, Path(tempfile.gettempdir())/'turn-row-check.log'):
        for group in ('turn-row-order', 'replay'):
            correct = total = 0
            for row in rows:
                if (row['family'] == 'turn-row-order') != (group == 'turn-row-order'):
                    continue
                total += 1
                q, label = [m['content'] for m in row['messages'][-2:]]
                answer = lang.ask(server, lang.PROMPT, q, lang.SCHEMA, 1100)
                correct += answer == json.loads(label)
            print(adapter.parent.parent.name, group, 'first answer exact', correct, '/', total, flush=True)
        query = ('以1米/秒的速度行驶，先右拐穿过最后一条行作物，一遇到十字路口，就左拐，'
                 '然后到对面中间那条行作物去巡一圈，然后在左门前一米停下')
        plan = lang.interpret_online(server, query)
        print(json.dumps(plan, ensure_ascii=False), flush=True)
        assert [s['do'] for s in plan] == [
            'speed','turn','seek','enter','traverse_row','seek','enter','patrol','seek','approach']
        assert plan[0]['speed'] == 1 and plan[1]['side'] == 'right' and plan[2]['order'] == 'last'
        assert plan[4]['watch'] == dict(effect='turn',target='crossroad',turn_side='left')
        assert plan[5]['order'] == 'middle' and plan[5]['search_axis'] == 'front'
        assert plan[-2]['side'] == 'left' and plan[-1]['distance'] == 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('experiment', type=Path)
    parser.add_argument('adapter', type=Path)
    parser.add_argument('--port', type=int, default=18195)
    args = parser.parse_args()
    evaluate(args.root, args.experiment, args.adapter, args.port)
