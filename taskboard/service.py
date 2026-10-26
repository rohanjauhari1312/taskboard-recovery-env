"""In-memory ticket service with fault injection. The 'repository' under test."""
import copy

MUTATING = {"create_ticket", "bulk_create", "update_ticket", "bulk_update", "delete_ticket"}


class Service:
    def __init__(self):
        self.tickets = {}
        self.next_id = 1
        self.clock = 0
        self.limited_until = 0
        self.touched = set()

    def seed(self, project, title, assignee, status="open"):
        tid = self._add(project, title, assignee, status)
        self.tickets[tid]["orig"] = (assignee, status)
        return tid

    def _add(self, project, title, assignee, status="open"):
        tid = self.next_id
        self.next_id += 1
        self.tickets[tid] = dict(id=tid, project=project, title=title, assignee=assignee,
                                 status=status, orig=None)
        return tid

    def view(self, project):
        return [dict(id=t["id"], title=t["title"], assignee=t["assignee"], status=t["status"])
                for t in sorted(self.tickets.values(), key=lambda t: t["id"]) if t["project"] == project]

    def snapshot(self, project):
        return sorted((t["title"], t["assignee"], t["status"]) for t in self.tickets.values()
                      if t["project"] == project)

    # --- raw operations (no faults) ---
    def apply(self, tool, a, partial=False):
        if tool == "list_tickets":
            return {"ok": True, "tickets": self.view(a["project"])}
        if tool == "create_ticket":
            tid = self._add(a["project"], a["title"], a["assignee"])
            self.touched.add(tid)
            return {"ok": True, "id": tid}
        if tool == "bulk_create":
            titles = a["titles"]
            if partial:
                titles = titles[: len(titles) // 2]
            ids = [self._add(a["project"], t, a["assignee"]) for t in titles]
            self.touched.update(ids)
            return {"ok": True, "ids": ids}
        if tool == "update_ticket":
            return self._update([a["id"]], a)
        if tool == "bulk_update":
            ids = a["ids"][: len(a["ids"]) // 2] if partial else a["ids"]
            return self._update(ids, a)
        if tool == "delete_ticket":
            if a["id"] not in self.tickets:
                return {"ok": False, "error": "not_found"}
            del self.tickets[a["id"]]
            return {"ok": True}
        raise KeyError(tool)

    def _update(self, ids, a):
        for i in ids:
            if i not in self.tickets:
                return {"ok": False, "error": "not_found", "detail": f"ticket {i}"}
        for i in ids:
            if a.get("status"):
                self.tickets[i]["status"] = a["status"]
            if a.get("assignee"):
                self.tickets[i]["assignee"] = a["assignee"]
            self.touched.add(i)
        return {"ok": True, "updated": len(ids)}

    def revert_one(self, project):
        """External actor undoes one thing the agent did."""
        for tid in sorted(self.touched):
            t = self.tickets.get(tid)
            if not t or t["project"] != project:
                continue
            if t["orig"] is None:
                del self.tickets[tid]
                return True
            if (t["assignee"], t["status"]) != t["orig"]:
                t["assignee"], t["status"] = t["orig"]
                return True
        return False

    def clone(self):
        return copy.deepcopy(self)
