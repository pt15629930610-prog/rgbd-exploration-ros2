"""Text-only supervised LoRA on the last FFN of the official Qwen3-VL-2B.

For text without image tokens the three MRoPE coordinates equal the token index.
Cache frozen activations on disk and backpropagate only through the trainable
low-rank projection, final RMSNorm, and tied language head.
"""

import argparse
import json
import math
from pathlib import Path
import random
import shutil
import sys
from datetime import datetime


def normalize(x, weight):
    y = x.float()
    return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * weight


def pack(rows, tokenizer):
    packed = []
    for row in rows:
        msgs = row['messages']
        prompt = ''.join(
            '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' for m in msgs[:-1]
        )
        prompt += '<|im_start|>assistant\n'
        prefix = tokenizer.encode(prompt, add_special_tokens=False).ids
        answer = tokenizer.encode(msgs[-1]['content'] + '<|im_end|>', add_special_tokens=False).ids
        packed.append((row['id'], prefix, answer))
    return sorted(packed, key=lambda t: len(t[1]) + len(t[2]))


def cache_group(group, weights, config, cache):
    h = config['head_dim']
    ids = [prefix + answer[:-1] for _, prefix, answer in group]
    length = max(map(len, ids))
    tokens = torch.tensor([v + [0] * (length - len(v)) for v in ids])
    emb = weights.get_tensor('model.language_model.embed_tokens.weight')
    x = emb[tokens].to('cuda')
    seq = torch.arange(length, device='cuda', dtype=torch.float32)
    freq = config['rope_theta'] ** (-torch.arange(0, h, 2, device='cuda').float() / h)
    phase = torch.outer(seq, freq).repeat(1, 2)
    cosine, sine = phase.cos().to(x.dtype), phase.sin().to(x.dtype)

    def rotate(t):
        a, b = t.chunk(2, -1)
        return t * cosine + torch.cat((-b, a), dim=-1) * sine

    for layer in range(config['num_hidden_layers']):
        prefix = f'model.language_model.layers.{layer}.'

        def w(key):
            return weights.get_tensor(prefix + key + '.weight').to('cuda')

        n = normalize(x, w('input_layernorm'))

        def heads(name):
            t = F.linear(n, w('self_attn.' + name + '_proj')).view(len(group), length, -1, h)
            if name != 'v':
                t = normalize(t, w('self_attn.' + name + '_norm'))
            t = t.transpose(1, 2)
            return t if name == 'v' else rotate(t)

        q, k, v = heads('q'), heads('k'), heads('v')
        copies = q.shape[1] // k.shape[1]
        att = F.scaled_dot_product_attention(
            q, k.repeat_interleave(copies, 1), v.repeat_interleave(copies, 1), is_causal=True
        )
        x = x + F.linear(att.transpose(1, 2).reshape(len(group), length, -1), w('self_attn.o_proj'))
        n = normalize(x, w('post_attention_layernorm'))
        middle = F.silu(F.linear(n, w('mlp.gate_proj'))) * F.linear(n, w('mlp.up_proj'))
        x = x + F.linear(middle, w('mlp.down_proj'))
    for i, (name, prefix, answer) in enumerate(group):
        response = slice(len(prefix) - 1, len(prefix) - 1 + len(answer))
        path = cache / (name + '.pt')
        temporary = path.with_suffix('.tmp')
        torch.save(
            (
                x[i, response].cpu().contiguous(),
                middle[i, response].cpu().contiguous(),
                torch.tensor(answer),
            ),
            temporary,
        )
        temporary.replace(path)


def export(folder, a, b, layer, alpha):
    import gguf

    path = folder / 'patrol-lora.gguf'
    writer = gguf.GGUFWriter(str(path.with_suffix('.tmp')), 'qwen3vl')
    writer.add_type('adapter')
    writer.add_string('adapter.type', 'lora')
    writer.add_float32('adapter.lora.alpha', float(alpha))
    for suffix, t in [('a', a), ('b', b)]:
        writer.add_tensor(f'blk.{layer}.ffn_down.weight.lora_{suffix}', t.detach().cpu().half().numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    path.with_suffix('.tmp').replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--group', type=int, default=12)
    parser.add_argument('--rank', type=int, default=32)
    args = parser.parse_args()
    root = args.root
    sys.path.insert(0, str(root / 'deps'))
    global torch, F
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from tokenizers import Tokenizer
    import hashlib

    random.seed(731)
    torch.manual_seed(731)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    base = root / 'base'
    out = root / 'runs' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True)
    config = json.loads((base / 'config.json').read_text())['text_config']
    assert config['num_hidden_layers'] == 28 and config['tie_word_embeddings']
    tok = Tokenizer.from_file(str(base / 'tokenizer.json'))
    paths = {}
    records = []
    signature = hashlib.sha256((base / 'revision.txt').read_bytes() + b'cache-v1')
    for split in ('train', 'validation'):
        data = (root / 'data' / (split + '.jsonl')).read_bytes()
        signature.update(data)
        rows = [json.loads(line) for line in data.decode().splitlines()]
        personal = root / 'data' / 'personal_train.jsonl'
        if split == 'train' and personal.exists():
            extra = personal.read_bytes()
            signature.update(extra)
            rows.extend(
                json.loads(line) for line in extra.decode('utf-8-sig').splitlines() if line.strip()
            )
        paths[split] = [row['id'] for row in rows]
        records.extend(rows)
    if len({row['id'] for row in records}) != len(records):
        raise ValueError('Training and validation need unique example IDs')
    cache = root / 'activation_cache' / signature.hexdigest()[:16]
    cache.mkdir(parents=True, exist_ok=True)
    with safe_open(str(base / 'model.safetensors'), framework='pt', device='cpu') as weights:
        packed = pack([r for r in records if not (cache / (r['id'] + '.pt')).exists()], tok)
        with torch.no_grad():
            for i in range(0, len(packed), args.group):
                cache_group(packed[i : i + args.group], weights, config, cache)
                if i % (5 * args.group) == 0:
                    print('cache', i, len(packed), flush=True)
        head = weights.get_tensor('model.language_model.embed_tokens.weight').to('cuda')
        norm = weights.get_tensor('model.language_model.norm.weight').to('cuda')
    a = torch.nn.Parameter(torch.empty(args.rank, config['intermediate_size'], device='cuda'))
    b = torch.nn.Parameter(torch.zeros(config['hidden_size'], args.rank, device='cuda'))
    torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5))
    opt = torch.optim.AdamW([a, b], lr=0.00045)

    def epoch(names, training):
        loss_sum = 0.0
        count = 0
        for j, name in enumerate(names):
            x, m, y = torch.load(cache / (name + '.pt'), map_location='cpu', weights_only=True)
            for start in range(0, len(y), 64):
                sl = slice(start, start + 64)
                z = F.linear(F.linear(m[sl].cuda().float(), a), b)
                hidden = normalize(x[sl].cuda() + z.to(head.dtype), norm)
                logits = F.linear(hidden, head)
                loss = F.cross_entropy(logits.float(), y[sl].cuda(), reduction='sum')
                if training:
                    (loss / (len(y) * 4)).backward()
                loss_sum += loss.item()
                count += len(y[sl])
            if training and ((j + 1) % 4 == 0 or j == len(names) - 1):
                torch.nn.utils.clip_grad_norm_([a, b], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
        return loss_sum / count

    with torch.no_grad():
        best = epoch(paths['validation'], False)
    history = [{'epoch': 0, 'validation_nll': best}]
    print(history[-1], flush=True)
    for i in range(args.epochs):
        random.shuffle(paths['train'])
        loss = epoch(paths['train'], True)
        with torch.no_grad():
            valid = epoch(paths['validation'], False)
        history.append({'epoch': i + 1, 'train_nll': loss, 'validation_nll': valid})
        print(history[-1], flush=True)
        if valid < best:
            best = valid
            export(out, a, b, config['num_hidden_layers'] - 1, args.rank)
            torch.save({'A': a.detach().cpu(), 'B': b.detach().cpu(), 'epoch': i + 1}, out / 'best.pt')
        (out / 'history.json').write_text(json.dumps(history, indent=2))
    if not (out / 'patrol-lora.gguf').exists():
        raise RuntimeError('Validation loss did not improve over the base model. See ' + str(out))
    active = root / 'adapter'
    active.mkdir(exist_ok=True)
    for name in ('best.pt', 'history.json', 'patrol-lora.gguf'):
        stage = active / (name + '.tmp')
        shutil.copyfile(out / name, stage)
        stage.replace(active / name)
    print('Saved run', out, 'and selected adapter', active, flush=True)


if __name__ == '__main__':
    main()
