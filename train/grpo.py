"""Minimal episode-level GRPO on top of an SFT LoRA adapter (MLX). Reward = env success (0/1)."""
import argparse, json, random, time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map, tree_flatten
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: F401 (adapter load path)
from agents.llm import SYSTEM
from taskboard import RecoveryEnv
import re

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--split", default="train")
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--tasks", type=int, default=4)
ap.add_argument("--group", type=int, default=4)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--temp", type=float, default=0.9)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
M = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
model, tok = load(M, adapter_path=a.adapter)
sampler = make_sampler(temp=a.temp)
opt = optim.Adam(learning_rate=a.lr)
rng = random.Random(a.seed)


def prompt(obs):
    return tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": obs}],
                                   add_generation_prompt=True, tokenize=False)


def rollout(seed):
    env = RecoveryEnv(a.split)
    obs = env.reset(seed)
    steps, done, r = [], False, 0.0
    while not done:
        p = prompt(obs)
        out = generate(model, tok, prompt=p, max_tokens=120, sampler=sampler)
        m = re.search(r"\{.*\}", out, re.S)
        act = m.group(0) if m else out
        steps.append((p, out))
        obs, r, done, _ = env.step(act)
    return steps, r


def loss_fn(model, ids, plen):
    logits = model(ids[None, :-1])[0].astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tgt = ids[1:]
    tl = mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0]
    return tl[plen - 1:].sum()


def sample_loss(model, p, out, adv):
    pi = tok.encode(p)
    oi = tok.encode(out) + [tok.eos_token_id]
    ids = mx.array(pi + oi)
    return -adv * loss_fn(model, ids, len(pi)) / len(oi)


log = []
for it in range(a.iters):
    t0 = time.time()
    seeds = [rng.randrange(0, 50000) for _ in range(a.tasks)]
    batch, rewards = [], []
    for s in seeds:
        eps = [rollout(s) for _ in range(a.group)]
        rs = [r for _, r in eps]
        mean = sum(rs) / len(rs)
        rewards += rs
        for st, r in eps:
            adv = r - mean
            if abs(adv) > 1e-6:
                batch += [(p, o, adv) for p, o in st]
    sr = sum(rewards) / len(rewards)
    if batch:
        grads_acc, n = None, len(batch)
        model.train()
        for p, o, adv in batch:
            l, g = nn.value_and_grad(model, lambda m: sample_loss(m, p, o, adv / n))(model)
            grads_acc = g if grads_acc is None else tree_map(lambda x, y: x + y, grads_acc, g)
            mx.eval(grads_acc)
        opt.update(model, grads_acc)
        mx.eval(model.parameters(), opt.state)
        model.eval()
    log.append(dict(iter=it, train_success=sr, n_updates=len(batch), sec=round(time.time() - t0)))
    print(log[-1], flush=True)
    if (it + 1) % 5 == 0 or it == a.iters - 1:
        import os, shutil
        os.makedirs(a.out, exist_ok=True)
        mx.save_safetensors(f"{a.out}/adapters.safetensors", dict(tree_flatten(model.trainable_parameters())))
        shutil.copy(f"{a.adapter}/adapter_config.json", f"{a.out}/adapter_config.json")
        json.dump(log, open(f"{a.out}/log.json", "w"), indent=1)
