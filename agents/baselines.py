"""Scripted reference policies. They read privileged env.expected (the goal as structured data);
learned agents only ever see the text observation."""
import json
import random

from taskboard.env import PROJ


def reconcile(rows, expected):
    """Ops that turn `rows` (listing with ids) into `expected` [(title, assignee, status)]."""
    remaining = [tuple(e) for e in expected]
    unmatched = []
    for r in rows:
        k = (r["title"], r["assignee"], r["status"])
        if k in remaining:
            remaining.remove(k)
        else:
            unmatched.append(r)
    extras, changes = [], {}
    for r in unmatched:
        e = next((e for e in remaining if e[0] == r["title"]), None)
        if e is None:
            extras.append(r)
            continue
        remaining.remove(e)
        ch = (e[2] if e[2] != r["status"] else None, e[1] if e[1] != r["assignee"] else None)
        changes.setdefault(ch, []).append(r["id"])
    ops = [{"tool": "delete_ticket", "args": {"id": r["id"]}} for r in extras]
    for (st, asg), ids in changes.items():
        args = {k: v for k, v in (("status", st), ("assignee", asg)) if v}
        if len(ids) == 1:
            ops.append({"tool": "update_ticket", "args": {"id": ids[0], **args}})
        else:
            ops.append({"tool": "bulk_update", "args": {"ids": ids, **args}})
    by_asg = {}
    for t, asg, _ in remaining:
        by_asg.setdefault(asg, []).append(t)
    for asg, ts in by_asg.items():
        if len(ts) == 1:
            ops.append({"tool": "create_ticket", "args": {"project": PROJ, "title": ts[0], "assignee": asg}})
        else:
            ops.append({"tool": "bulk_create", "args": {"project": PROJ, "titles": ts, "assignee": asg}})
    return ops


class Scripted:
    """mode: naive (blind retry, no verification) | reactive (re-list after an error) |
    full (re-list after an error and verify before finishing)."""

    def __init__(self, mode, lead_list=False):
        self.mode = mode
        self.lead_list = lead_list

    def start(self, env):
        self.env = env
        self.queue = reconcile(env.svc.view(PROJ), env.expected)
        self.cur = None
        self.retries = 0
        self.pending = []
        if self.lead_list:
            self.queue = []
            self.pending = [self.LIST]

    LIST = {"tool": "list_tickets", "args": {"project": PROJ}}

    def act(self, obs):
        env = self.env
        if self.pending:
            self.cur = self.pending.pop(0)
            return json.dumps(self.cur)
        last = env.last_result if env.history else {}
        if self.cur and self.cur["tool"] == "list_tickets" and last.get("ok"):
            self.queue = reconcile(last["tickets"], env.expected)
            if not self.queue:
                self.cur = {"tool": "finish", "args": {}}
                return json.dumps(self.cur)
        elif self.cur and last.get("ok") is False:
            rl = last.get("error") == "429"
            if self.mode == "naive":
                self.retries += 1
                if self.retries <= 3:
                    if rl:
                        self.pending = [self.cur]
                        self.cur = {"tool": "wait", "args": {"seconds": 3}}
                        return json.dumps(self.cur)
                    return json.dumps(self.cur)
                self.retries = 0
            else:
                self.pending = [self.LIST]
                if rl:
                    self.cur = {"tool": "wait", "args": {"seconds": 3}}
                    return json.dumps(self.cur)
                self.cur = self.pending.pop(0)
                return json.dumps(self.cur)
        else:
            self.retries = 0
        if self.queue:
            self.cur = self.queue.pop(0)
            return json.dumps(self.cur)
        if self.mode == "full" and not (self.cur and self.cur["tool"] == "list_tickets"):
            self.cur = self.LIST
            return json.dumps(self.cur)
        self.cur = {"tool": "finish", "args": {}}
        return json.dumps(self.cur)


class RandomValid:
    def start(self, env):
        self.rng = random.Random(0)
        self.env = env

    def act(self, obs):
        tools = ["list_tickets", "finish", "wait", "create_ticket", "update_ticket"]
        t = self.rng.choice(tools)
        args = {"list_tickets": {"project": PROJ}, "finish": {}, "wait": {"seconds": 1},
                "create_ticket": {"project": PROJ, "title": "Fix cache", "assignee": "ana"},
                "update_ticket": {"id": self.rng.randint(1, 8), "status": "closed"}}[t]
        return json.dumps({"tool": t, "args": args})
