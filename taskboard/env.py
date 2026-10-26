"""Gym-style RL environment: reset(seed, split) / step(action_json)."""
import json
import random

from .service import Service, MUTATING

USERS = ["ana", "ben", "chen", "dia", "eli"]
VERBS = ["Fix", "Review", "Draft", "Audit", "Ship", "Test", "Document", "Triage"]
NOUNS = ["login", "billing", "search", "export", "onboarding", "cache", "alerts", "backup", "invoice", "sync"]
PROJ, OTHER = "web", "ops"
MAX_STEPS = 24

TOOLS = """Tools (reply with exactly one JSON object: {"tool": NAME, "args": {...}}):
list_tickets {project}
create_ticket {project, title, assignee}
bulk_create {project, titles:[...], assignee}
update_ticket {id, status?: "open"|"closed", assignee?}
bulk_update {ids:[...], status?, assignee?}
delete_ticket {id}
wait {seconds}
finish {}   (ends the episode; call only when the goal is fully done)"""

# fault types. TRAIN_FAULTS are seen in training, HELDOUT_FAULTS only at eval.
TRAIN_FAULTS = ["timeout_not_applied", "timeout_applied", "rate_limit"]
HELDOUT_FAULTS = ["bulk_partial", "silent_drop", "external_mutation"]
TRAIN_TASKS = ["create", "reassign", "close"]
HELDOUT_TASKS = ["mixed", "handoff"]


def _titles(rng, k):
    pool = [f"{v} {n}" for v in VERBS for n in NOUNS]
    return rng.sample(pool, k)


def make_task(rng, kind):
    """Returns (service, goal_text, expected_snapshot)."""
    svc = Service()
    a, b = rng.sample(USERS, 2)
    seedt = _titles(rng, 8)
    for i, t in enumerate(seedt[:5]):
        svc.seed(PROJ, t, rng.choice([a, b, rng.choice(USERS)]))
    for t in seedt[5:]:
        svc.seed(OTHER, t, rng.choice(USERS))

    def cur():
        return [dict(t) for t in svc.tickets.values() if t["project"] == PROJ]

    def reassign_exp(rows, src, dst):
        return [(r["title"], dst if r["assignee"] == src and r["status"] == "open" else r["assignee"], r["status"]) for r in rows]

    exp_rows = [(r["title"], r["assignee"], r["status"]) for r in cur()]
    if kind == "create":
        k = rng.randint(2, 4)
        new = [t for t in _titles(rng, 14) if t not in seedt][:k]
        goal = f"In project '{PROJ}', create {k} new tickets assigned to {a}: " + "; ".join(new) + "."
        exp = exp_rows + [(t, a, "open") for t in new]
    elif kind == "reassign":
        # guarantee >=2 open tickets for a
        for r in cur()[:2]:
            svc.tickets[r["id"]]["assignee"] = a
            svc.tickets[r["id"]]["orig"] = (a, "open")
        rows = cur()
        goal = f"In project '{PROJ}', reassign every open ticket currently assigned to {a} to {b}."
        exp = reassign_exp(rows, a, b)
    elif kind == "close":
        verb = rng.choice([r["title"].split()[0] for r in cur()])
        for r in cur()[:2]:
            svc.tickets[r["id"]]["title"] = f"{verb} {r['title'].split()[1]}"
        rows = cur()
        goal = f"In project '{PROJ}', close every ticket whose title starts with '{verb}'."
        exp = [(r["title"], r["assignee"], "closed" if r["title"].startswith(verb + " ") else r["status"]) for r in rows]
    elif kind == "mixed":
        k = rng.randint(2, 3)
        new = [t for t in _titles(rng, 14) if t not in seedt][:k]
        verb = rng.choice([r["title"].split()[0] for r in cur()])
        for r in cur()[:2]:
            svc.tickets[r["id"]]["title"] = f"{verb} {r['title'].split()[1]}"
        rows = cur()
        goal = (f"In project '{PROJ}', create {k} tickets assigned to {a}: " + "; ".join(new)
                + f". Then close every ticket whose title starts with '{verb}' (existing ones only).")
        exp = [(r["title"], r["assignee"], "closed" if r["title"].startswith(verb + " ") else r["status"]) for r in rows]
        exp += [(t, a, "open") for t in new]
    elif kind == "handoff":
        for r in cur()[:3]:
            svc.tickets[r["id"]]["assignee"] = a
            svc.tickets[r["id"]]["orig"] = (a, "open")
        rows = cur()
        goal = (f"In project '{PROJ}', reassign every open ticket assigned to {a} to {b}, "
                f"then close all tickets now assigned to {b}.")
        exp = []
        for r in rows:
            asg = b if r["assignee"] == a and r["status"] == "open" else r["assignee"]
            exp.append((r["title"], asg, "closed" if asg == b else r["status"]))
    else:
        raise ValueError(kind)
    return svc, goal, sorted(exp)


class RecoveryEnv:
    """split: 'train' | 'eval_id' | 'eval_ood' | 'clean'.
    train/eval_id: train task kinds + train faults (disjoint seed ranges).
    eval_ood: held-out fault types and/or task kinds. clean: no faults."""

    def __init__(self, split="train", fault_override=None, task_override=None):
        self.split = split
        self.fault_override = fault_override
        self.task_override = task_override

    def reset(self, seed):
        rng = random.Random(seed)
        s = self.split
        tasks = self.task_override or (HELDOUT_TASKS if s == "eval_ood_task" else TRAIN_TASKS)
        kind = rng.choice(tasks)
        self.svc, self.goal, self.expected = make_task(rng, kind)
        self.kind = kind
        self.other_expected = self.svc.snapshot(OTHER)
        self.faults = []
        if s != "clean":
            if self.fault_override is not None:
                pool = self.fault_override
            elif s == "eval_ood":
                pool = HELDOUT_FAULTS
            else:
                pool = TRAIN_FAULTS
            n = rng.randint(2, 3)
            for _ in range(n):
                self.faults.append(dict(type=rng.choice(pool), at=rng.randint(0, 3), fired=False))
        self.mcount = 0
        self.steps = 0
        self.done = False
        self.fired = []
        self.history = []
        return self.observation()

    # ---- observation
    def observation(self, last=None):
        h = "\n".join(f"> {a}\n< {r}" for a, r in self.history[-8:])
        return f"GOAL: {self.goal}\n{TOOLS}\n\nHISTORY:\n{h}\n\nNext action:" if h else f"GOAL: {self.goal}\n{TOOLS}\n\nHISTORY:\n(none)\n\nNext action:"

    def _match(self, tool):
        """Return the fault that fires on this mutating call, if any."""
        for f in self.faults:
            if f["fired"]:
                continue
            t = f["type"]
            if t == "bulk_partial":
                if tool in ("bulk_create", "bulk_update"):
                    return f
                continue
            if t == "external_mutation":
                continue
            if self.mcount >= f["at"]:
                return f
        return None

    def step(self, action):
        assert not self.done
        self.steps += 1
        info = {}
        reward = 0.0
        raw = action if isinstance(action, str) else json.dumps(action)
        res = self._exec(action)
        if res.get("finished"):
            self.done = True
            reward = 1.0 if self._success() else 0.0
            info["success"] = bool(reward)
            res = {"ok": True, "detail": "episode finished"}
        elif self.steps >= MAX_STEPS:
            self.done = True
            info["success"] = False
            info["timeout"] = True
        self.last_result = res
        self.history.append((raw.strip()[:300], json.dumps(res)[:600]))
        info["faults_fired"] = list(self.fired)
        return self.observation(), reward, self.done, info

    def _exec(self, action):
        try:
            a = json.loads(action) if isinstance(action, str) else action
            tool, args = a["tool"], a.get("args", {})
            assert isinstance(args, dict)
        except Exception:
            return {"ok": False, "error": "invalid_action", "detail": "expected JSON {\"tool\":..., \"args\":{...}}"}
        if tool == "finish":
            return {"finished": True}
        if tool == "wait":
            try:
                self.svc.clock += max(0, min(int(args["seconds"]), 60))
            except Exception:
                return {"ok": False, "error": "invalid_action", "detail": "wait needs integer seconds"}
            return {"ok": True, "clock": self.svc.clock}
        spec = {"list_tickets": ["project"], "create_ticket": ["project", "title", "assignee"],
                "bulk_create": ["project", "titles", "assignee"], "update_ticket": ["id"],
                "bulk_update": ["ids"], "delete_ticket": ["id"]}
        if tool not in spec:
            return {"ok": False, "error": "unknown_tool", "detail": tool}
        miss = [k for k in spec[tool] if k not in args]
        if miss:
            return {"ok": False, "error": "invalid_action", "detail": f"missing args: {miss}"}
        if tool in ("update_ticket", "bulk_update") and not (args.get("status") or args.get("assignee")):
            return {"ok": False, "error": "invalid_action", "detail": "need status or assignee"}
        if tool in ("update_ticket", "bulk_update") and args.get("status") not in (None, "open", "closed"):
            return {"ok": False, "error": "invalid_action", "detail": "status must be open|closed"}
        if tool in ("bulk_create", "bulk_update") and not all(isinstance(args.get(k, []), list) for k in ("titles", "ids")):
            return {"ok": False, "error": "invalid_action", "detail": "titles/ids must be lists"}
        types = {"project": str, "title": str, "assignee": str, "id": int}
        bad = [k for k, t in types.items() if k in args and (not isinstance(args[k], t) or isinstance(args[k], bool))]
        if "titles" in args and not (isinstance(args["titles"], list) and all(isinstance(x, str) for x in args["titles"])):
            bad.append("titles")
        if "ids" in args and not (isinstance(args["ids"], list) and all(isinstance(x, int) and not isinstance(x, bool) for x in args["ids"])):
            bad.append("ids")
        if bad:
            return {"ok": False, "error": "invalid_action", "detail": f"wrong type for: {bad}"}
        if tool == "list_tickets":
            return self.svc.apply(tool, args)
        # mutating call
        if self.svc.clock < self.svc.limited_until:
            return {"ok": False, "error": "429", "detail": f"rate limited, retry after {self.svc.limited_until - self.svc.clock}s"}
        f = self._match(tool)
        self.mcount += 1
        try:
            if f:
                f["fired"] = True
                self.fired.append(f["type"])
                t = f["type"]
                if t == "timeout_not_applied":
                    return {"ok": False, "error": "timeout", "detail": "request timed out"}
                if t == "timeout_applied":
                    self.svc.apply(tool, args)
                    return {"ok": False, "error": "timeout", "detail": "request timed out"}
                if t == "rate_limit":
                    self.svc.limited_until = self.svc.clock + 3
                    return {"ok": False, "error": "429", "detail": "rate limited, retry after 3s"}
                if t == "bulk_partial":
                    self.svc.apply(tool, args, partial=True)
                    return {"ok": False, "error": "500", "detail": "internal error during bulk operation"}
                if t == "silent_drop":
                    out = {"ok": True}
                    if tool == "create_ticket":
                        out["id"] = self.svc.next_id + 50
                    return out
            return self.svc.apply(tool, args)
        finally:
            for g in self.faults:
                if g["type"] == "external_mutation" and not g["fired"] and self.mcount > g["at"]:
                    if self.svc.revert_one(PROJ):
                        g["fired"] = True
                        self.fired.append("external_mutation")

    def _success(self):
        return (self.svc.snapshot(PROJ) == self.expected
                and self.svc.snapshot(OTHER) == self.other_expected)
