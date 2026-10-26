import json, sys
from taskboard import RecoveryEnv
from agents.baselines import Scripted, RandomValid


def run_episode(policy, env, seed):
    obs = env.reset(seed)
    if hasattr(policy, "start"):
        policy.start(env)
    done, r, info = False, 0.0, {}
    while not done:
        obs, r, done, info = env.step(policy.act(obs))
    return dict(seed=seed, success=bool(r), steps=env.steps, faults=info["faults_fired"], kind=env.kind)


def evaluate(policy, split, seeds, **kw):
    rows = [run_episode(policy, RecoveryEnv(split, **kw), s) for s in seeds]
    n = len(rows)
    faulted = [x for x in rows if x["faults"]]
    return dict(split=split, n=n, success=sum(x["success"] for x in rows) / n,
                faulted_n=len(faulted),
                success_when_faulted=(sum(x["success"] for x in faulted) / len(faulted)) if faulted else None,
                mean_steps=sum(x["steps"] for x in rows) / n)


EVAL_SEEDS = range(100000, 100200)
SPLITS = ["clean", "eval_id", "eval_ood", "eval_ood_task"]

if __name__ == "__main__":
    out = {}
    for name, p in [("random", RandomValid()), ("naive", Scripted("naive")),
                    ("reactive", Scripted("reactive")), ("full", Scripted("full"))]:
        out[name] = {s: evaluate(p, s, EVAL_SEEDS) for s in SPLITS}
        print(name, {s: (round(v["success"], 2), v["success_when_faulted"] and round(v["success_when_faulted"], 2)) for s, v in out[name].items()})
    json.dump(out, open("results/baselines.json", "w"), indent=1)
