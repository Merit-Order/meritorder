"""Long filing-log digests with multi-hop questions (long-context retrieval with superseded records).

The document is an organisation's filing log: staff entries, project charters, tickets, transfers, lead changes,
office moves and site-head appointments, padded with status updates, meeting notes and budget lines that discuss
changes but never make them. The question follows a chain of 2-4 lookups (ticket -> project -> lead -> office ->
site head) and every hop on the chain has a trap next to it: a later record that was rescinded, a change scheduled
after the as-of date, a retroactive correction filed after it, a near-duplicate ticket number, codename, name or
employee ID. The answer comes from replaying the log under the rules printed in the prompt.

``build`` returns the whole structure (records, target chain, answer) so tests can re-derive the answer;
``multi_hop`` is the task.yaml entry point.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta

FIRST_NAMES = [
    "Ada", "Aiko", "Amara", "Anders", "Bea", "Bram", "Camille", "Dario", "Dmitri", "Elif", "Emeka", "Esme", "Farah",
    "Felix", "Greta", "Hamid", "Hana", "Ilse", "Imogen", "Ines", "Jonas", "Jun", "Kaito", "Kalinda", "Lars", "Leila",
    "Lorenzo", "Mads", "Malik", "Marta", "Mei", "Mirela", "Nadia", "Nils", "Noor", "Oskar", "Paloma", "Pavel",
    "Priya", "Quentin", "Rafael", "Rania", "Rosa", "Sanne", "Selim", "Sofia", "Soren", "Tamsin", "Teodor", "Thea",
    "Tomas", "Ulla", "Vera", "Viktor", "Wanda", "Wiebke", "Yara", "Yusuf", "Zofia", "Zoran",
]
SURNAMES = [
    "Abara", "Achterberg", "Alves", "Andrade", "Baptiste", "Bergstrom", "Bianchi", "Castillo", "Chaudhry", "Costa",
    "Dahl", "Delacroix", "Demir", "Duarte", "Eriksen", "Esposito", "Farouk", "Fontaine", "Gallo", "Garrido",
    "Gunnarsson", "Halloran", "Holm", "Iqbal", "Jaramillo", "Jensen", "Kaur", "Keller", "Kimura", "Lachance",
    "Lombardi", "Lund", "Mahmoud", "Marchetti", "Mendes", "Mwangi", "Nakamura", "Nieminen", "Nordin", "Oduya",
    "Olsen", "Ortega", "Osei", "Pacheco", "Quiroga", "Rahman", "Rinaldi", "Rocha", "Ruiz", "Saarinen", "Salo",
    "Santos", "Sato", "Silva", "Strand", "Suzuki", "Takahashi", "Tanaka", "Toivonen", "Urquhart", "Vasquez",
    "Vidal", "Virtanen", "Wagner", "Watanabe", "Yilmaz", "Yamamoto", "Zamora", "Zeller", "Zielinski",
]
# Pairs that differ by a letter or two. One side of a pair is always on the target chain.
TWIN_SURNAMES = [
    ("Kowalczyk", "Kowalski"), ("Pedersen", "Petersen"), ("Lindqvist", "Lindquist"), ("Moreau", "Moreno"),
    ("Haddad", "Hadad"), ("Novak", "Nowak"), ("Brandt", "Brant"), ("Fischer", "Fisher"), ("Johansson", "Johanson"),
    ("Meyer", "Meier"), ("Schmid", "Schmidt"), ("Hansen", "Hanssen"), ("Ferreira", "Pereira"), ("Rossi", "Rossini"),
]
TWIN_FIRST = [("Katarina", "Katharina"), ("Mikkel", "Mikael"), ("Annika", "Anneka"), ("Joana", "Johanna"),
              ("Matteo", "Mattia"), ("Elena", "Eliana")]
CODENAMES = [
    "Bittern", "Curlew", "Dunlin", "Fieldfare", "Garnet", "Hazel", "Ibis", "Juniper", "Kittiwake", "Lapwing",
    "Merlin", "Nightjar", "Plover", "Quail", "Redstart", "Siskin", "Teal", "Vetch", "Whimbrel", "Yarrow", "Zircon",
    "Amber", "Basalt", "Cobalt", "Dolomite", "Ember", "Flint", "Gannet", "Hornbeam", "Kestrel", "Magpie", "Nutmeg",
    "Obsidian", "Pumice", "Quartz", "Saffron", "Tamarind", "Umber", "Wagtail",
]
TWIN_CODENAMES = [("Larch", "Larchwood"), ("Heron", "Herongate"), ("Alder", "Alderney"), ("Osprey", "Ospreyfield"),
                  ("Rowan", "Rowanberry"), ("Wren", "Wrenfield"), ("Linnet", "Linnetvale"), ("Sorrel", "Sorrelby")]
CITIES = ["Lisbon", "Ghent", "Tallinn", "Porto", "Leeds", "Gdansk", "Bergen", "Lyon", "Turin", "Utrecht", "Aarhus",
          "Brno"]
ROLES = ["logistics analyst", "site engineer", "operations planner", "data steward", "procurement specialist",
         "quality auditor", "fleet coordinator", "systems administrator", "customs specialist", "warehouse supervisor",
         "integration engineer", "finance partner", "safety officer", "network engineer", "product owner"]
SUBJECTS = ["Label printer", "Pallet scanner", "Invoice export", "Dock scheduler", "Route planner", "Badge reader",
            "Cold-room sensor", "Forklift telemetry", "Customs form generator", "Returns portal", "Yard camera feed",
            "Shift roster sync", "Carrier rate import", "Weighbridge display", "Stock count app", "Driver app login",
            "Pick-list printer", "Temperature logger", "Seal number check", "Manifest upload"]
FAULTS = ["drops jobs after the firmware update", "times out during shift change", "reports duplicate entries",
          "shows the wrong time zone", "fails on long addresses", "rejects valid barcodes", "freezes when offline",
          "double-counts partial pallets", "loses its settings overnight", "sends alerts to the wrong team",
          "rounds weights incorrectly", "crashes on Monday mornings", "skips the last line of each batch",
          "uses last year's tariff table"]
REASONS = ["workload rebalancing after the quarterly planning review", "a request from the operations council",
           "the previous owner's parental leave", "a consolidation of overlapping work", "the vendor contract renewal",
           "an audit recommendation", "a reorganisation of the northern sites", "the steering group's decision",
           "a skills match identified in the capacity review", "a customer escalation"]
PROGRESS = ["vendor reproduced the fault on a test unit", "a workaround was shared with the night shift",
            "logs were collected from three sites", "the fix is waiting for a maintenance window",
            "root cause narrowed to a configuration mismatch", "the issue did not recur this week",
            "a patch is in testing", "users were asked for screenshots", "priority was raised after two complaints",
            "the supplier asked for more samples", "regression tests were added", "the backlog item was re-estimated"]
NEUTRAL_MINUTES = ["Reviewed the burn-down chart; two blockers remain.", "Agreed to move the retrospective to Thursday.",
                   "Confirmed the hardware order has shipped.", "Discussed onboarding material for new joiners.",
                   "Noted that the test environment will be refreshed next week.",
                   "Walked through the open risks; none were escalated.", "Celebrated the release of the new dashboard.",
                   "Asked finance to confirm the travel budget.", "Shared feedback from the last site visit."]
COURSES = ["forklift safety refresher", "data protection basics", "first aid at work", "customs documentation level 2",
           "incident command", "lean warehousing", "advanced spreadsheet modelling", "fire warden training"]
FACILITIES = ["will close early on Friday for a fire drill", "has a new visitor sign-in kiosk",
              "replaced its loading-bay doors", "is repainting the canteen next month", "reports a broken lift",
              "now offers bicycle parking", "has changed its cleaning contractor", "will host the regional all-hands"]

CHANGE_KINDS = ["STAFF ENTRY", "PROJECT CHARTER", "LEAD CHANGE", "TICKET OPENED", "TICKET TRANSFER",
                "OFFICE TRANSFER", "SITE HEAD", "RESCISSION"]
# Entity counts and timeline per hop count. Filler records pad the log to the requested length.
LEVELS = {
    2: {"people": 34, "projects": 12, "tickets": 40, "start": date(2024, 1, 8), "days": 330},
    3: {"people": 80, "projects": 26, "tickets": 110, "start": date(2023, 3, 6), "days": 640},
    4: {"people": 150, "projects": 40, "tickets": 220, "start": date(2023, 1, 9), "days": 1000},
}


@dataclass(eq=False)
class Record:
    filed: date
    kind: str
    text: str  # body; "{ref}" stands for the rescinded record's number
    changes: list[tuple[tuple[str, str], str]] = field(default_factory=list)
    effective: date | None = None
    rescinds: Record | None = None
    rid: str = ""
    tiebreak: float = 0.0

    def render(self) -> str:
        body = self.text.replace("{ref}", self.rescinds.rid) if self.rescinds else self.text
        return f"{self.rid} | filed {self.filed.isoformat()} | {self.kind}\n{body}"


@dataclass
class Digest:
    prompt: str
    answer: str
    records: list[Record]
    hops: int
    as_of: date | None
    target: dict  # names of the entities on the chain, for tests


def replay(records: list[Record], as_of: date | None = None, respect_rescissions: bool = True,
           respect_dates: bool = True, filed_by: date | None = None) -> dict:
    """State after the log (or at the end of ``as_of``), under the prompt's rules. The flags give the naive readings:
    ignoring rescissions, ignoring effective dates (latest filing wins), or only what had been filed by a date."""
    void = {id(r.rescinds) for r in records if r.rescinds is not None} if respect_rescissions else set()
    live = [(r.effective if (respect_dates and r.effective) else r.filed, i, r) for i, r in enumerate(records)
            if r.changes and id(r) not in void and (filed_by is None or r.filed <= filed_by)]
    if as_of is not None and respect_dates:
        live = [x for x in live if x[0] <= as_of]
    state: dict = {}
    for _, _, r in sorted(live, key=lambda x: (x[0], x[1]) if respect_dates else x[1]):
        for key, value in r.changes:
            state[key] = value
    return state


def follow(state: dict, ticket: str, hops: int) -> str:
    """The chain lookup the question asks for, against a replayed state."""
    project = state[("ticket", ticket)]
    if hops == 2:
        return state[("lead", project)]  # level 3 records refer to projects by codename and people by name
    lead = state[("lead", state[("codename", project)])]
    if hops == 3:
        return state[("name", lead)]
    return state[("name", state[("head", state[("office", lead)])])]


def wrong_readings(d: Digest) -> list[str]:
    """Answers a careless reader would give: the look-alike ticket, the original project's lead, and (with
    rescissions and dates in play) the log replayed while ignoring them or reading only what was filed by the date."""
    t, hops = d.target, d.hops
    state = replay(d.records, d.as_of)
    out = [follow(state, t["ticket_twin"], hops)]
    if hops <= 3:
        lead = state[("lead", t["from_code"])]
        out.append(lead if hops == 2 else state[("name", lead)])
    if hops >= 3:
        out += [follow(replay(d.records, d.as_of, respect_rescissions=False, respect_dates=False), t["ticket"], hops),
                follow(replay(d.records, d.as_of, respect_rescissions=False), t["ticket"], hops)]
    if hops == 4:
        out += [follow(replay(d.records, d.as_of, respect_dates=False), t["ticket"], hops),
                follow(replay(d.records, d.as_of, filed_by=d.as_of), t["ticket"], hops)]
    return out


class _Builder:
    def __init__(self, seed: int, words: int, hops: int, attempt: int = 0):
        if hops not in LEVELS:
            raise ValueError("hops must be 2, 3 or 4")
        self.rng = random.Random(seed * 10 + hops + 7919 * attempt)
        self.words, self.hops = words, hops
        self.cfg = LEVELS[hops]
        self.start: date = self.cfg["start"]
        self.end: date = self.start + timedelta(days=self.cfg["days"])
        self.records: list[Record] = []
        self.dated = hops == 4
        self.by_id = hops >= 3  # people by employee ID and tickets by project ID from three hops up

    # ------------------------------------------------------------------ helpers
    def day(self, lo: date, hi: date) -> date:
        return lo + timedelta(days=self.rng.randint(0, max(0, (hi - lo).days)))

    def who(self, pid: str) -> str:
        return pid if self.by_id else self.people[pid]["name"]

    def proj(self, pid: str) -> str:
        return pid if self.by_id else self.projects[pid]

    def add(self, filed: date, kind: str, text: str, changes=(), effective: date | None = None,
            rescinds: Record | None = None) -> Record:
        if changes and self.dated:
            effective = effective or filed
            text = f"{text} Effective {effective.isoformat()}."
        rec = Record(filed=filed, kind=kind, text=text, changes=list(changes), effective=effective if changes else None,
                     rescinds=rescinds, tiebreak=self.rng.random())
        self.records.append(rec)
        return rec

    def rescind(self, rec: Record, filed: date) -> Record:
        why = self.rng.choice(["it was filed against the wrong item", "the approving manager withdrew approval",
                               "it duplicated a request that was declined", "it was entered without sign-off"])
        return self.add(filed, "RESCISSION", f"Record {{ref}} is rescinded and void because {why}.", rescinds=rec)

    def reason(self) -> str:
        return f"Reason: {self.rng.choice(REASONS)}."

    # ------------------------------------------------------------------ entities
    def make_people(self) -> None:
        rng, n = self.rng, self.cfg["people"]
        twins = []
        for a, b in rng.sample(TWIN_SURNAMES, 6):
            first = rng.choice(FIRST_NAMES)
            twins.append((f"{first} {a}", f"{first} {b}"))
        for a, b in rng.sample(TWIN_FIRST, 3):
            sur = rng.choice(SURNAMES)
            twins.append((f"{a} {sur}", f"{b} {sur}"))
        names = [x for pair in twins for x in pair]
        used = set(names)
        while len(names) < n:
            name = f"{rng.choice(FIRST_NAMES)} {rng.choice(SURNAMES)}"
            if name not in used:
                used.add(name)
                names.append(name)
        ids: list[int] = []
        taken: set[int] = set()
        for i, _ in enumerate(names):
            cand = 0
            if i % 2 == 1 and i < 2 * len(twins):  # twin of the previous person: swap two adjacent digits
                digits = list(str(ids[-1]))
                options = [j for j in range(1, 3) if digits[j] != digits[j + 1]]
                if options:
                    j = rng.choice(options)
                    digits[j], digits[j + 1] = digits[j + 1], digits[j]
                    cand = int("".join(digits))
            while cand in taken or not 1000 <= cand <= 9999:
                cand = rng.randint(1000, 9999)
            taken.add(cand)
            ids.append(cand)
        self.people = {f"E-{i}": {"name": nm, "office": rng.choice(CITIES), "role": rng.choice(ROLES)}
                       for i, nm in zip(ids, names, strict=True)}
        self.pids = list(self.people)
        self.twin_of = {}
        for k in range(len(twins)):
            a, b = self.pids[2 * k], self.pids[2 * k + 1]
            self.twin_of[a], self.twin_of[b] = b, a

    def make_projects(self) -> None:
        rng, n = self.rng, self.cfg["projects"]
        pairs = rng.sample(TWIN_CODENAMES, 4)
        names = [x for p in pairs for x in p] + rng.sample(CODENAMES, n - 8)
        nums = rng.sample(range(100, 1000), n)
        self.projects = {f"P-{num}": code for num, code in zip(nums, names, strict=True)}
        self.prj = list(self.projects)
        self.code_twin = {}
        for k in range(4):
            a, b = self.prj[2 * k], self.prj[2 * k + 1]
            self.code_twin[a], self.code_twin[b] = b, a

    def make_tickets(self) -> None:
        rng, n = self.rng, self.cfg["tickets"]
        nums: list[int] = []
        taken: set[int] = set()
        while len(nums) < n:
            cand = 0
            if len(nums) % 2 == 1 and len(nums) < 6:  # three near-duplicate ticket pairs: swap two adjacent digits
                digits = list(str(nums[-1]))
                j = next((j for j in (3, 2, 1) if digits[j] != digits[j + 1]), 3)
                digits[j], digits[j + 1] = digits[j + 1], digits[j]
                cand = int("".join(digits))
            pair_base = len(nums) % 2 == 0 and len(nums) < 6
            while cand in taken or not 10000 <= cand <= 99999 or (pair_base and len(set(str(cand))) < 4):
                cand = rng.randint(10000, 99999)
            taken.add(cand)
            nums.append(cand)
        self.tickets = {f"T-{num}": f"{rng.choice(SUBJECTS)} {rng.choice(FAULTS)}" for num in nums}
        self.tids = list(self.tickets)

    # ------------------------------------------------------------------ the log
    def base_records(self) -> None:
        rng = self.rng
        early = self.start + timedelta(days=self.cfg["days"] // 12)
        for pid, p in self.people.items():
            art = "an" if p["role"][0] in "aeiou" else "a"
            self.add(self.day(self.start, early), "STAFF ENTRY",
                     f"{p['name']} joins as {art} {p['role']}. Employee ID: {pid}. Base office: {p['office']}.",
                     [(("name", pid), p["name"]), (("office", pid), p["office"])])
        self.heads = {}
        pool = self.pids[len(self.twin_of):]  # non-twin people head offices at the start
        for city, head in zip(CITIES, rng.sample(pool, len(CITIES)), strict=True):
            self.heads[city] = head
            self.add(self.day(self.start + timedelta(days=3), early), "SITE HEAD",
                     f"Site head of the {city} office: {self.who(head)}. Appointed at the opening review.",
                     [(("head", city), self.who(head))])
        self.leads = {}
        charter_end = self.start + timedelta(days=self.cfg["days"] // 6)
        self.chartered = {}
        for prj, code in self.projects.items():
            lead = rng.choice(self.pids)
            self.leads[prj] = lead
            when = self.day(early, charter_end)
            self.chartered[prj] = when
            self.add(when, "PROJECT CHARTER",
                     f"Project {code} ({prj}) is chartered to {rng.choice(['improve', 'replace', 'audit', 'automate'])} "
                     f"the {rng.choice(SUBJECTS).lower()} workflow. Lead: {self.who(lead)}.",
                     [(("codename", prj), code), (("lead", code), self.who(lead))])
        self.opened = {}
        self.home = {}
        for tid, title in self.tickets.items():
            prj = rng.choice(self.prj)
            when = self.day(max(charter_end, self.chartered[prj] + timedelta(days=1)),
                            self.start + timedelta(days=int(self.cfg["days"] * 0.55)))
            self.opened[tid], self.home[tid] = when, prj
            self.add(when, "TICKET OPENED", f'Ticket {tid} "{title}" is opened under project {self.proj(prj)}.',
                     [(("ticket", tid), self.proj(prj))])

    def transfer(self, tid: str, prj: str, filed: date, effective: date | None = None, note: str = "") -> Record:
        text = f"Ticket {tid} is reassigned to project {self.proj(prj)}. {note or self.reason()}"
        return self.add(filed, "TICKET TRANSFER", text, [(("ticket", tid), self.proj(prj))], effective)

    def lead_change(self, prj: str, person: str, filed: date, effective: date | None = None, note: str = "") -> Record:
        code = self.projects[prj]
        text = f"Project {code}: new lead {self.who(person)}. {note or self.reason()}"
        return self.add(filed, "LEAD CHANGE", text, [(("lead", code), self.who(person))], effective)

    def relocate(self, person: str, city: str, filed: date, effective: date | None = None) -> Record:
        return self.add(filed, "OFFICE TRANSFER", f"{self.who(person)} relocates to the {city} office. {self.reason()}",
                        [(("office", person), city)], effective)

    def appoint(self, city: str, person: str, filed: date, effective: date | None = None) -> Record:
        return self.add(filed, "SITE HEAD", f"Site head of the {city} office: {self.who(person)}. {self.reason()}",
                        [(("head", city), self.who(person))], effective)

    def target_chain(self) -> None:
        rng, d = self.rng, self.cfg["days"]

        def span(a: float, b: float) -> date:  # a random day between fractions a and b of the timeline
            return self.day(self.start + timedelta(days=int(d * a)), self.start + timedelta(days=int(d * b)))

        tid = self.tids[0]  # has a near-duplicate twin, self.tids[1]
        twin_codes = [p for p in self.prj[:8] if p in self.code_twin]
        p2 = rng.choice(twin_codes)
        others = [p for p in self.prj if p not in (p2, self.code_twin[p2])]
        p1, p3, p4 = rng.sample(others, 3)
        self.home[tid] = p1
        opened = self.opened[tid]
        # re-file the target ticket's opening under p1 (the generic opening used a random project)
        for r in self.records:
            if r.kind == "TICKET OPENED" and r.changes[0][0] == ("ticket", tid):
                r.text = f'Ticket {tid} "{self.tickets[tid]}" is opened under project {self.proj(p1)}.'
                r.changes = [(("ticket", tid), self.proj(p1))]
                if self.dated:
                    r.text += f" Effective {r.filed.isoformat()}."
                    r.effective = r.filed
        twins = [p for p in self.pids if p in self.twin_of]
        answer_person = twins[2 * rng.randrange(len(twins) // 2) + rng.randrange(2)]
        rest = [p for p in self.pids if p not in (answer_person, self.twin_of[answer_person])]
        self.excluded = {("ticket", tid)}
        self.target = {"ticket": tid, "ticket_twin": self.tids[1], "project": p2, "project_twin": self.code_twin[p2],
                       "from": p1, "from_code": self.projects[p1], "decoys": [p3, p4]}
        code2 = self.projects[p2]
        if self.hops == 2:
            a0, b = rng.sample(rest, 2)
            self.leads[p2] = a0
            self.set_charter_lead(p2, a0)
            t1 = max(span(0.6, 0.7), opened + timedelta(days=2))
            self.transfer(tid, p2, t1)
            self.lead_change(p2, answer_person, span(0.3, 0.9))
            self.lead_change(p1, b, t1 + timedelta(days=rng.randint(5, 30)))
            self.lead_change(self.code_twin[p2], self.twin_of[answer_person], span(0.4, 0.95))
            self.note(t1 + timedelta(days=rng.randint(3, 20)), "MEETING NOTE",
                      f"{code2} and {self.projects[p3]} triage. {self.who(rng.choice(rest))} suggested moving ticket "
                      f"{tid} to project {self.proj(p3)}; the group liked the idea but no transfer was filed.")
            self.excluded |= {("lead", code2)}
            self.target.update(lead=answer_person, answer=self.people[answer_person]["name"])
        elif self.hops == 3:
            a0, a3 = rng.sample(rest, 2)
            self.leads[p2] = a0
            self.set_charter_lead(p2, a0)
            t1 = max(span(0.45, 0.55), opened + timedelta(days=2))
            self.transfer(tid, p2, t1)
            bad = self.transfer(tid, p3, max(span(0.6, 0.7), t1 + timedelta(days=5)))
            self.rescind(bad, bad.filed + timedelta(days=rng.randint(4, 40)))
            t4 = span(0.3, 0.5)
            self.lead_change(p2, answer_person, t4)
            bad_lead = self.lead_change(p2, a3, t4 + timedelta(days=rng.randint(20, 60)))
            self.rescind(bad_lead, bad_lead.filed + timedelta(days=rng.randint(3, 30)))
            self.lead_change(self.code_twin[p2], self.twin_of[answer_person], t4 + timedelta(days=rng.randint(5, 90)))
            self.note(t1 + timedelta(days=rng.randint(3, 20)), "MEETING NOTE",
                      f"{code2} weekly sync. Someone asked whether ticket {tid} would sit better under project "
                      f"{self.proj(p4)}; it was left open and no transfer was filed.")
            self.excluded |= {("lead", code2)}
            self.target.update(lead=answer_person, answer=self.people[answer_person]["name"])
        else:
            self.as_of = d_ = self.start + timedelta(days=int(d * 0.72))

            def back(a: int, b: int) -> date:  # between a and b days before the as-of date
                return d_ - timedelta(days=rng.randint(min(a, b), max(a, b)))

            def fwd(a: int, b: int) -> date:  # between a and b days after it
                return d_ + timedelta(days=rng.randint(min(a, b), max(a, b)))

            head = answer_person
            a0, a1, a2, h0 = rng.sample(rest, 4)
            o0, o1, o2, oa1 = rng.sample(CITIES, 4)
            self.leads[p2] = a0
            self.set_charter_lead(p2, a0)
            # hop 1: ticket -> project (a void transfer and a scheduled one sit after the real one)
            f = max(back(200, 120), opened + timedelta(days=2))
            self.transfer(tid, p2, f, f + timedelta(days=rng.randint(0, 10)))
            f = back(110, 60)
            bad = self.transfer(tid, p4, f, f + timedelta(days=3))
            self.rescind(bad, f + timedelta(days=rng.randint(5, 30)))
            self.transfer(tid, p3, back(40, 15), fwd(10, 40), note="Scheduled to take effect after the quarter close.")
            # hop 2: project -> lead (a retroactive correction filed after the as-of date wins)
            f = max(back(150, 90), self.chartered[p2] + timedelta(days=2))
            self.lead_change(p2, a1, f, f + timedelta(days=rng.randint(0, 7)))
            self.lead_change(p2, a2, fwd(5, 25), back(30, 10),
                             note="Late filing: this corrects the record, the handover happened earlier.")
            self.lead_change(self.code_twin[p2], a1, back(80, 20))
            # hop 3: lead -> office (a scheduled relocation after the as-of date does not count yet)
            self.people[a2]["office"] = o0
            self.set_staff_office(a2, o0)
            self.people[a1]["office"] = oa1
            self.set_staff_office(a1, oa1)
            self.relocate(a2, o1, back(250, 200))
            self.relocate(a2, o2, back(30, 10), fwd(20, 60))
            # hop 4: office -> site head (the later appointment of the near-duplicate person was rescinded)
            self.set_initial_head(o1, h0)
            f = back(300, 260)
            self.appoint(o1, head, f, f + timedelta(days=7))
            f = back(80, 50)
            bad = self.appoint(o1, self.twin_of[head], f, f + timedelta(days=3))
            self.rescind(bad, back(40, 20))
            self.note(back(60, 30), "MEETING NOTE",
                      f"Leadership forum. {self.who(rng.choice(rest))} proposed that {self.who(a1)} take back project "
                      f"{code2} and that ticket {tid} move to project {self.proj(p1)}; neither change was filed.")
            self.excluded |= {("lead", code2), ("office", a2), ("office", a1), ("head", o1)}
            self.no_head = {head, self.twin_of[head], a1, a2, h0}
            self.target.update(lead=a2, stale_lead=a1, office=o1, scheduled_office=o2, head=head,
                               answer=self.people[head]["name"])

    def set_charter_lead(self, prj: str, person: str) -> None:
        for r in self.records:
            if r.kind == "PROJECT CHARTER" and r.changes[0] == (("codename", prj), self.projects[prj]):
                old = self.who(self.leads_before[prj])
                r.text = r.text.replace(f"Lead: {old}.", f"Lead: {self.who(person)}.")
                r.changes[1] = (r.changes[1][0], self.who(person))

    def set_staff_office(self, person: str, city: str) -> None:
        for r in self.records:
            if r.kind == "STAFF ENTRY" and r.changes[0][0] == ("name", person):
                r.text = r.text.split(" Base office:")[0] + f" Base office: {city}." + (
                    f" Effective {r.filed.isoformat()}." if self.dated else "")
                r.changes[1] = (("office", person), city)

    def set_initial_head(self, city: str, person: str) -> None:
        for r in self.records:
            if r.kind == "SITE HEAD" and r.changes[0][0] == ("head", city):
                r.text = f"Site head of the {city} office: {self.who(person)}. Appointed at the opening review." + (
                    f" Effective {r.filed.isoformat()}." if self.dated else "")
                r.changes = [(("head", city), self.who(person))]

    def note(self, filed: date, kind: str, text: str) -> Record:
        return self.add(filed, kind, text)

    def eff(self, filed: date, floor: date) -> date | None:
        """Effective date for a background change: mostly the filing date, sometimes scheduled or retroactive."""
        if not self.dated:
            return None
        roll = self.rng.random()
        if roll < 0.2:
            return filed + timedelta(days=self.rng.randint(5, 45))
        if roll < 0.3 and filed - timedelta(days=25) > floor:
            return filed - timedelta(days=self.rng.randint(3, 20))
        return filed

    def background_changes(self) -> None:
        rng = self.rng
        late = self.end - timedelta(days=5)
        changes: list[Record] = []
        for tid in self.tids:
            if ("ticket", tid) in self.excluded:
                continue
            prev = self.home[tid]
            dates = sorted(self.day(self.opened[tid] + timedelta(days=3), late) for _ in range(rng.choice([0, 1, 1, 2, 3])))
            for f in dates:
                nxt = rng.choice([p for p in self.prj if p != prev])
                changes.append(self.transfer(tid, nxt, f, self.eff(f, self.opened[tid])))
                prev = nxt
        for prj, code in self.projects.items():
            if ("lead", code) in self.excluded:
                continue
            prev = self.leads[prj]
            dates = sorted(self.day(self.chartered[prj] + timedelta(days=5), late) for _ in range(rng.choice([0, 1, 2, 3])))
            for f in dates:
                nxt = rng.choice([p for p in self.pids if p != prev])
                changes.append(self.lead_change(prj, nxt, f, self.eff(f, self.chartered[prj])))
                prev = nxt
        joined = self.start + timedelta(days=self.cfg["days"] // 12 + 1)
        for pid, person in self.people.items():
            if ("office", pid) in self.excluded:
                continue
            prev = person["office"]
            dates = sorted(self.day(joined, late) for _ in range(rng.choice([0, 0, 1, 1, 2])))
            for f in dates:
                nxt = rng.choice([c for c in CITIES if c != prev])
                changes.append(self.relocate(pid, nxt, f, self.eff(f, joined)))
                prev = nxt
        no_head = getattr(self, "no_head", set())
        pool = [p for p in self.pids if p not in no_head]
        for city in CITIES:
            if ("head", city) in self.excluded:
                continue
            for f in sorted(self.day(joined, late) for _ in range(rng.choice([0, 1, 2]))):
                changes.append(self.appoint(city, rng.choice(pool), f, self.eff(f, joined)))
        if self.hops >= 3:
            for rec in rng.sample(changes, len(changes) // 12):
                self.rescind(rec, rec.filed + timedelta(days=rng.randint(3, 40)))

    def filler(self) -> Record:
        rng = self.rng
        kind = rng.choices(["STATUS UPDATE", "MEETING NOTE", "BUDGET LINE", "FACILITIES NOTE", "TRAINING LOG"],
                           weights=[5, 4, 2, 1, 1])[0]
        when = self.day(self.start + timedelta(days=10), self.end)

        def person() -> str:
            return self.who(rng.choice(self.pids))

        if kind == "STATUS UPDATE":
            tid = rng.choice(self.tids)
            when = self.day(self.opened[tid] + timedelta(days=1), self.end)
            bits = rng.sample(PROGRESS, 2)
            text = f"Ticket {tid} ({self.tickets[tid].lower()}): {bits[0]}; {bits[1]}. Update posted by {person()}."
        elif kind == "MEETING NOTE":
            prj = rng.choice(self.prj)
            code = self.projects[prj]
            attendees = ", ".join(self.who(p) for p in rng.sample(self.pids, 3))
            items = rng.sample(NEUTRAL_MINUTES, 2)
            roll = rng.random()
            if roll < 0.35:
                tid = rng.choice(self.tids)
                items.append(f"{person()} proposed moving ticket {tid} to project {self.proj(rng.choice(self.prj))}; "
                             "the idea was parked and nothing was filed.")
            elif roll < 0.6:
                items.append(f"It was floated that {person()} could lead {code} next year; no decision was taken.")
            elif roll < 0.75:
                items.append(f"{person()} mentioned a possible move to the {rng.choice(CITIES)} office; "
                             "it is only an idea for now.")
            text = f"{code} sync. Attendees: {attendees}. " + " ".join(items)
        elif kind == "BUDGET LINE":
            prj = rng.choice(self.prj)
            text = (f"Project {self.projects[prj]} ({prj}) budget for Q{rng.randint(1, 4)} approved at "
                    f"EUR {rng.randint(12, 240) * 500:,}, of which EUR {rng.randint(2, 40) * 250:,} is for hardware. "
                    f"Approver: {person()}.")
        elif kind == "FACILITIES NOTE":
            text = f"The {rng.choice(CITIES)} office {rng.choice(FACILITIES)}. Contact: {person()}."
        else:
            text = f"{person()} completed the {rng.choice(COURSES)} course with a score of {rng.randint(70, 100)}%."
        return self.add(when, kind, text)

    def target_notes(self) -> None:
        """Status chatter about the target ticket and its near-duplicate, spread through the log."""
        rng = self.rng
        tid, twin = self.target["ticket"], self.target["ticket_twin"]
        for t in [tid, twin, tid, twin]:
            when = self.day(self.opened[t] + timedelta(days=2), self.end)
            self.add(when, "STATUS UPDATE", f"Ticket {t} ({self.tickets[t].lower()}): {rng.choice(PROGRESS)}.")

    def build(self) -> Digest:
        self.make_people()
        self.make_projects()
        self.make_tickets()
        self.leads_before: dict = {}
        self.as_of = None
        self.excluded = set()
        self.base_records()
        self.leads_before = dict(self.leads)
        self.target_chain()
        self.background_changes()
        self.target_notes()
        rules, question = self.rules(), self.question()
        budget = self.words - len(rules.split()) - len(question.split())
        used = sum(len(r.render().split()) + 1 for r in self.records)  # +1 for the record number field
        while used < budget:
            used += len(self.filler().render().split())
        self.records.sort(key=lambda r: (r.filed, r.tiebreak))
        for i, r in enumerate(self.records, 1):
            r.rid = f"R-{i:04d}"
        for r in self.records:  # a rescission always comes after the record it voids
            assert r.rescinds is None or int(r.rescinds.rid[2:]) < int(r.rid[2:])
        log = "\n\n".join(r.render() for r in self.records)
        state = replay(self.records, self.as_of)
        answer = follow(state, self.target["ticket"], self.hops)
        assert answer == self.target["answer"], (answer, self.target["answer"])
        prompt = f"{rules}\n\n=== FILING LOG ===\n\n{log}\n\n=== END OF LOG ===\n\n{question}"
        return Digest(prompt=prompt, answer=answer, records=self.records, hops=self.hops, as_of=self.as_of,
                      target=self.target)

    def rules(self) -> str:
        lines = [
            "Below is the complete filing log of Tidewater Freight Cooperative's operations office. Read the rules, "
            "then answer the question at the end.",
            "",
            "RULES FOR READING THE LOG",
            "- Records appear in filing order and are numbered R-0001, R-0002, and so on.",
            "- Only these record types change anything: STAFF ENTRY, PROJECT CHARTER, LEAD CHANGE, TICKET OPENED, "
            "TICKET TRANSFER, OFFICE TRANSFER, SITE HEAD" + (", RESCISSION." if self.hops >= 3 else "."),
            "- Every other record (status updates, meeting notes, budget lines, facilities notes, training logs) is "
            "informational. Proposals, plans and ideas mentioned in them change nothing.",
            "- A STAFF ENTRY gives a person's name, employee ID and base office; an OFFICE TRANSFER moves that person "
            "to another base office. A SITE HEAD record names who heads an office.",
            "- A PROJECT CHARTER creates a project with a codename, a project ID and its first lead; a LEAD CHANGE "
            "replaces the lead of the project whose codename it names.",
            "- A TICKET OPENED record files a ticket under a project; a TICKET TRANSFER moves it to another project.",
        ]
        if self.hops >= 3:
            lines += [
                "- A RESCISSION voids the record it names entirely, as if that record had never been filed. "
                "Rescissions are never themselves rescinded.",
                "- Later records refer to people by employee ID and tickets refer to projects by project ID. "
                "Different people can have very similar names and IDs, and some codenames and ticket numbers look "
                "alike; match them exactly.",
            ]
        if self.dated:
            lines += [
                "- Every change carries an effective date. The state at the end of a day D comes from applying, in "
                "order of effective date (ties broken by record number), every valid change whose effective date is "
                "on or before D. Filing dates do not matter for this: a record filed after D still counts if its "
                "effective date is on or before D, and a record filed on or before D does not count yet if its "
                "effective date is after D.",
            ]
        else:
            lines.append("- When several valid records set the same thing, the latest-filed one is current.")
        return "\n".join(lines)

    def question(self) -> str:
        tid = self.target["ticket"]
        if self.hops == 2:
            q = f"Who currently leads the project that ticket {tid} is filed under? Give the person's full name."
        elif self.hops == 3:
            q = (f"What is the full name of the person who currently leads the project that ticket {tid} is "
                 "currently filed under?")
        else:
            q = (f"At the end of {self.as_of.isoformat()}, ticket {tid} was filed under some project, and that "
                 "project had a lead. Who was the site head, at the end of that same day, of the office where that "
                 "lead was based? Give the site head's full name.")
        return (f"QUESTION: {q}\n\nWork through the log carefully. End your reply with a line of the form "
                "`Final answer: <full name>`.")


def build(seed: int, words: int, hops: int) -> Digest:
    """The first digest for this seed (retrying with derived seeds) in which every careless reading is wrong."""
    for attempt in range(50):
        d = _Builder(seed, words, hops, attempt).build()
        if d.answer not in wrong_readings(d):
            return d
    raise RuntimeError("no digest with live traps")  # pragma: no cover - a handful of attempts always suffices


def multi_hop(seed: int = 1, words: int = 6000, hops: int = 2) -> dict:
    """task.yaml entry point: a filing log of about ``words`` words and a ``hops``-step lookup question."""
    d = build(seed, words, hops)
    eid = next(pid for pid, p in _people_of(d).items() if p == d.answer)
    return {"prompt": d.prompt, "checks": [{"final_one_of": [d.answer, f"{d.answer} ({eid})", f"{d.answer}, {eid}"]}]}


def _people_of(d: Digest) -> dict[str, str]:
    """Employee ID -> name, from the staff entries."""
    return {key[1]: value for r in d.records if r.kind == "STAFF ENTRY" for key, value in r.changes if key[0] == "name"}
