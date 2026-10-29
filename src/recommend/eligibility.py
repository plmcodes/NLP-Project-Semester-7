"""Step 6b (1): check a citizen profile against a scheme's extracted rules.

Each rule returns pass / fail / unknown. A scheme is
  * eligible           - every rule we know about passes
  * possibly eligible  - nothing fails, but some facts are unknown
  * not eligible       - at least one rule fails
"""
from __future__ import annotations

from dataclasses import dataclass, field

PASS, FAIL, UNKNOWN = "pass", "fail", "unknown"
ELIGIBLE, POSSIBLE, NOT_ELIGIBLE = "eligible", "possibly eligible", "not eligible"


@dataclass
class Profile:
    age: int | None = None
    gender: str | None = None            # "female" / "male" / "other"
    state: str | None = None
    residence: str | None = None         # "rural" / "urban"
    income: int | None = None            # annual family income in rupees
    caste: str | None = None             # "SC" / "ST" / "OBC" / "General"
    disability: bool | None = None
    bpl: bool | None = None
    occupation: str | None = None        # one of rules.OCCUPATIONS keys
    need: str = ""                       # free-text need, used for ranking

    @classmethod
    def from_dict(cls, d: dict) -> "Profile":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in keys and v not in ("", None)})


@dataclass
class Check:
    rule: str
    status: str
    detail: str


@dataclass
class Verdict:
    status: str
    checks: list = field(default_factory=list)

    @property
    def matched(self) -> int:
        return sum(c.status == PASS for c in self.checks)

    @property
    def known(self) -> int:
        return sum(c.status != UNKNOWN for c in self.checks)


def _fmt_money(v) -> str:
    v = int(v)
    return f"₹{v / 100_000:g} lakh" if v >= 100_000 else f"₹{v:,}"


def check(scheme: dict, p: Profile) -> Verdict:
    """`scheme` holds the extracted fields (age_min, age_max, income_max, ...)."""
    checks: list[Check] = []

    amin, amax = scheme.get("age_min"), scheme.get("age_max")
    if amin is not None or amax is not None:
        lo = "" if amin is None else f"{int(amin)}"
        hi = "" if amax is None else f"{int(amax)}"
        rng = f"{lo}–{hi}" if lo and hi else (f"{lo}+" if lo else f"up to {hi}")
        if p.age is None:
            checks.append(Check("age", UNKNOWN, f"Age {rng} required"))
        elif (amin is not None and p.age < amin) or (amax is not None and p.age > amax):
            checks.append(Check("age", FAIL, f"Age {rng} required"))
        else:
            checks.append(Check("age", PASS, f"Age {rng}"))

    if scheme.get("gender", "all") in ("female", "male"):
        g = scheme["gender"]
        if p.gender is None:
            checks.append(Check("gender", UNKNOWN, f"Only for {g} applicants"))
        else:
            checks.append(Check("gender", PASS if p.gender == g else FAIL, f"For {g} applicants"))

    castes = list(scheme.get("castes") or [])
    if castes:
        label = "/".join(castes)
        if p.caste is None:
            checks.append(Check("caste", UNKNOWN, f"For {label}"))
        else:
            checks.append(Check("caste", PASS if p.caste in castes else FAIL, f"For {label}"))

    states = list(scheme.get("states") or ["All"])
    if "All" not in states:
        label = ", ".join(states)
        if p.state is None:
            checks.append(Check("state", UNKNOWN, f"Residents of {label}"))
        else:
            checks.append(Check("state", PASS if p.state in states else FAIL, f"Residents of {label}"))

    res = scheme.get("residence", "both")
    if res in ("rural", "urban"):
        if p.residence is None:
            checks.append(Check("residence", UNKNOWN, f"{res.title()} areas only"))
        else:
            checks.append(Check("residence", PASS if p.residence == res else FAIL, f"{res.title()} areas"))

    inc = scheme.get("income_max")
    if inc is not None and inc == inc:  # skip NaN
        if p.income is None:
            checks.append(Check("income", UNKNOWN, f"Income up to {_fmt_money(inc)}"))
        else:
            checks.append(Check("income", PASS if p.income <= inc else FAIL, f"Income up to {_fmt_money(inc)}"))

    for flag, label in (("disability", "Persons with disability"), ("bpl", "BPL households")):
        if scheme.get(flag):
            val = getattr(p, flag)
            if val is None:
                checks.append(Check(flag, UNKNOWN, f"{label} only"))
            else:
                checks.append(Check(flag, PASS if val else FAIL, label))

    if any(c.status == FAIL for c in checks):
        status = NOT_ELIGIBLE
    elif not checks or any(c.status == UNKNOWN for c in checks):
        # No rules extracted means we cannot confirm anything: read the scheme page.
        status = POSSIBLE
    else:
        status = ELIGIBLE
    return Verdict(status, checks)
