import json
from taskboard import RecoveryEnv
from taskboard.env import PROJ
from agents.baselines import Scripted
from run_eval import run_episode, evaluate


def J(tool, **args):
    return json.dumps({"tool": tool, "args": args})


def test_deterministic_reset():
    a, b = RecoveryEnv("train"), RecoveryEnv("train")
    assert a.reset(7) == b.reset(7)
    assert a.expected == b.expected and a.faults == b.faults


def test_finish_immediately_fails():
    for s in range(30):
        e = RecoveryEnv("clean"); e.reset(s)
        _, r, d, _ = e.step(J("finish"))
        assert d and r == 0.0


def test_delete_everything_and_spam_fail():
    e = RecoveryEnv("clean"); e.reset(1)
    for t in e.svc.view(PROJ):
        e.step(J("delete_ticket", id=t["id"]))
    _, r, _, _ = e.step(J("finish"))
    assert r == 0.0
    e = RecoveryEnv("clean"); e.reset(1)
    for i in range(5):
        e.step(J("create_ticket", project=PROJ, title="x", assignee="ana"))
    assert e.step(J("finish"))[1] == 0.0


def test_collateral_damage_fails():
    e = RecoveryEnv("clean"); e.reset(2)
    exp_before = e.expected
    full = Scripted("full"); full.start(e)
    done = False
    e.svc.tickets[next(i for i, t in e.svc.tickets.items() if t["project"] == "ops")]["status"] = "closed"
    while not done:
        _, r, done, _ = e.step(full.act(""))
    assert r == 0.0


def test_invalid_actions_do_not_crash_and_cost_steps():
    e = RecoveryEnv("clean"); e.reset(3)
    for bad in ["not json", "{}", J("nope"), J("create_ticket"), J("update_ticket", id=1), J("wait", seconds="x")]:
        _, r, d, _ = e.step(bad)
        assert r == 0.0 and not d
    assert e.steps == 6


def test_step_limit():
    e = RecoveryEnv("clean"); e.reset(3)
    done = False
    while not done:
        _, r, done, info = e.step("garbage")
    assert info["timeout"] and r == 0.0


def test_timeout_semantics():
    e = RecoveryEnv("clean"); e.reset(4)
    e.faults = [dict(type="timeout_applied", at=0, fired=False)]
    n = len(e.svc.view(PROJ))
    out = e._exec(J("create_ticket", project=PROJ, title="t", assignee="ana"))
    assert out["error"] == "timeout" and len(e.svc.view(PROJ)) == n + 1
    e.faults = [dict(type="timeout_not_applied", at=0, fired=False)]
    e._exec(J("create_ticket", project=PROJ, title="t2", assignee="ana"))
    assert len(e.svc.view(PROJ)) == n + 1


def test_rate_limit_requires_wait():
    e = RecoveryEnv("clean"); e.reset(4)
    e.faults = [dict(type="rate_limit", at=0, fired=False)]
    a = J("create_ticket", project=PROJ, title="t", assignee="ana")
    assert e._exec(a)["error"] == "429"
    assert e._exec(a)["error"] == "429"
    e._exec(J("wait", seconds=3))
    assert e._exec(a)["ok"]


def test_scripted_ladder_and_clean_solvable():
    seeds = range(100000, 100060)
    res = {m: {s: evaluate(Scripted(m), s, seeds)["success"] for s in ["clean", "eval_id", "eval_ood", "eval_ood_task"]}
           for m in ["naive", "reactive", "full"]}
    assert all(v == 1.0 for v in res["full"].values())
    assert res["naive"]["eval_ood"] < res["reactive"]["eval_ood"] < res["full"]["eval_ood"]
    assert res["naive"]["clean"] == 1.0


def test_split_seed_ranges_have_distinct_tasks():
    train = {tuple(RecoveryEnv("train").reset(s).split("GOAL:")[1].split("\n")[:1]) for s in range(0, 200)}
    ev = {tuple(RecoveryEnv("eval_id").reset(s).split("GOAL:")[1].split("\n")[:1]) for s in range(100000, 100200)}
    assert len(ev - train) > 0.5 * len(ev)


def test_wrong_arg_types_rejected_without_state_change():
    e = RecoveryEnv("clean"); e.reset(5)
    before = e.svc.snapshot(PROJ)
    for bad in [J("create_ticket", project=PROJ, title=3, assignee="ana"), J("bulk_create", project=PROJ, titles=[1], assignee="ana"),
                J("bulk_update", ids=["a"], status="closed"), J("update_ticket", id="1", status="closed")]:
        assert e._exec(bad)["error"] == "invalid_action"
    assert e.svc.snapshot(PROJ) == before
    e.step(J("finish"))
