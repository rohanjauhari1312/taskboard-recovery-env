"""Generate SFT demos (obs -> action) from scripted policies on train seeds. Successful episodes only."""
import json, sys, random
from taskboard import RecoveryEnv
from agents.baselines import Scripted
from agents.llm import SYSTEM

ARMS = {
    "fault_full": dict(split="train", mode="full"),
    "clean_verify": dict(split="clean", mode="full"),
    "clean_noverify": dict(split="clean", mode="reactive"),
}


def build(arm, n_examples, out):
    cfg = ARMS[arm]
    rows, seed = [], 0
    while len(rows) < n_examples:
        env = RecoveryEnv(cfg["split"])
        pol = Scripted(cfg["mode"], lead_list=True)
        obs = env.reset(seed); pol.start(env)
        ep, done = [], False
        while not done:
            a = pol.act(obs)
            ep.append((obs, a))
            obs, r, done, _ = env.step(a)
        if r == 1.0:
            rows += [{"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": o},
                                   {"role": "assistant", "content": a}]} for o, a in ep]
        seed += 1
    rng = random.Random(0); rng.shuffle(rows)
    rows = rows[:n_examples]
    import os; os.makedirs(out, exist_ok=True)
    with open(f"{out}/train.jsonl", "w") as f:
        for r in rows[:-100]: f.write(json.dumps(r) + "\n")
    with open(f"{out}/valid.jsonl", "w") as f:
        for r in rows[-100:]: f.write(json.dumps(r) + "\n")
    print(arm, len(rows), "examples from", seed, "episodes")


if __name__ == "__main__":
    for arm in ARMS:
        build(arm, 1600, f"data/{arm}")
