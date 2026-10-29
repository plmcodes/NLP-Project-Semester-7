"""Step 4c (layer 1): rule-based eligibility extractor.

Turns a scheme's free-text eligibility section into structured fields:
age_min, age_max, income_max, gender, castes, states, residence,
disability, bpl, occupations, education. Each numeric field also keeps the
sentence it came from, so the app can show *why* a rule applies.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

LAKH = 100_000
CRORE = 100 * LAKH

STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
    "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal", "Andaman and Nicobar Islands",
    "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi",
    "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
]

OCCUPATIONS = {
    "farmer": r"farmers?|cultivators?|agriculturists?|kisan",
    "fisherman": r"fisher(?:men|man|folk|s)?|fishermen",
    "student": r"students?|pupils?|scholars?|studying|pursuing",
    "artisan": r"artisans?|weavers?|craftsm[ae]n|handloom|handicraft",
    "street vendor": r"street vendors?|hawkers?",
    "construction worker": r"construction workers?|building (?:and other )?construction",
    "unorganised worker": r"unorgani[sz]ed (?:sector )?workers?|labou?rers?|daily wage",
    "entrepreneur": r"entrepreneurs?|msmes?|start-?ups?|enterprises?|udyam",
    "senior citizen": r"senior citizens?|old age|elderly",
    "widow": r"widows?",
    "ex-serviceman": r"ex-?servicem[ae]n|veterans?|armed forces",
    "sportsperson": r"sports ?persons?|athletes?|players?|sportsm[ae]n",
    "researcher": r"researchers?|scientists?|ph\.?d|post-?doctoral",
    "teacher": r"teachers?",
    "artist": r"artists?|artistes?|performers?",
    "transgender": r"transgenders?",
    "pregnant woman": r"pregnant|lactating",
    "journalist": r"journalists?",
    "driver": r"drivers?",
}

EDUCATION = {
    "class_10": r"class ?(?:x|10)\b|10th|ssc|matricul",
    "class_12": r"class ?(?:xii|12)\b|12th|hsc|higher secondary|intermediate",
    "diploma": r"diploma|\biti\b|polytechnic",
    "graduate": r"graduat|bachelor|\bdegree\b|b\.?tech|b\.?e\.|mbbs",
    "postgraduate": r"post-?graduat|master|m\.?tech|m\.?sc|\bmba\b",
    "phd": r"ph\.?d|doctoral",
}

# Sentences with these words usually describe a benefit for a group
# ("SC/ST applicants get 5% relaxation"), not a requirement.
SOFT_CONTEXT = re.compile(
    r"relax|reserv|prefer|priorit|concession|additional|extra|higher (?:rate|subsidy|amount)"
    r"|in case of|\d+\s?%\s*(?:subsidy|assistance|relaxation|reservation|concession|more|additional|of the (?:cost|project|amount))"
    r"|age limit|upper age|not be applicable|including|any caste|may belong|irrespective|as well as"
    r"|and other communit", re.I)

MONEY = re.compile(
    r"(?:(?:₹|rs\.?|inr|rupees)\s*([\d,]+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?|k)?)"
    r"|(?:([\d,]+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?))", re.I)

AGE_CONTEXT = re.compile(r"\bage[ds]?\b|years? old|year-old|\byrs?\b|years of age", re.I)
NOT_AGE = re.compile(r"experience|service|residing|resident|domicile|registered|stay|business|"
                     r"operation|existence|working|employed|membership", re.I)


@dataclass
class Eligibility:
    age_min: int | None = None
    age_max: int | None = None
    income_max: int | None = None
    gender: str = "all"
    castes: list = field(default_factory=list)
    states: list = field(default_factory=lambda: ["All"])
    residence: str = "both"
    disability: bool = False
    bpl: bool = False
    occupations: list = field(default_factory=list)
    education: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def split_sentences(text: str) -> list[str]:
    parts = []
    for line in text.splitlines():
        parts.extend(s.strip() for s in re.split(r"(?<=[.;])\s+(?=[A-Z(])", line) if s.strip())
    return parts


def to_rupees(number: str, unit: str | None) -> int | None:
    try:
        value = float(number.replace(",", ""))
    except ValueError:
        return None
    unit = (unit or "").lower()
    if unit.startswith(("lakh", "lac")):
        value *= LAKH
    elif unit.startswith("crore"):
        value *= CRORE
    elif unit == "k":
        value *= 1000
    return int(value)


def money_amounts(sentence: str) -> list[int]:
    out = []
    for m in MONEY.finditer(sentence):
        num, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        v = to_rupees(num, unit)
        if v:
            out.append(v)
    return out


def extract_income(sentences: list[str]) -> tuple[int | None, str | None]:
    best, src = None, None
    for s in sentences:
        low = s.lower()
        if "income" not in low:
            continue
        amounts = [a for a in money_amounts(s) if a >= 1000]
        if not amounts:
            continue
        if re.search(r"per month|monthly|/ ?month|p\.m\.", low) and not re.search(r"annual|per annum|yearly|p\.a\.", low):
            amounts = [a * 12 for a in amounts]
        # Several caps (e.g. rural vs urban) -> keep the most lenient one.
        value = max(amounts)
        if best is None or value > best:
            best, src = value, s
    return best, src


def _valid_age(v) -> bool:
    return 0 < v <= 100


def extract_age(sentences: list[str]) -> tuple[int | None, int | None, str | None]:
    amin = amax = None
    src = None
    for s in sentences:
        low = s.lower()
        if NOT_AGE.search(low) and not re.search(r"\bages?\b", low):
            continue
        rng = re.search(r"between\s+(?:the\s+)?ages?\s+(?:of\s+)?(\d{1,2})\s*(?:years?\s*)?(?:and|to|-)\s*(\d{1,3})", low) \
            or re.search(r"(?:between|from)\s+(?:the\s+)?(?:ages?\s+(?:of\s+)?)?(\d{1,2})\s*(?:years?\s*)?(?:and|to|-)\s*(\d{1,3})\s*years?", low) \
            or re.search(r"(\d{1,2})\s*(?:-|–|to)\s*(\d{1,3})\s*(?:years|yrs)", low)
        if rng and (AGE_CONTEXT.search(low) or "between" in low):
            a, b = int(rng.group(1)), int(rng.group(2))
            if _valid_age(a) and _valid_age(b) and a < b:
                amin, amax, src = a, b, s
                continue
        if not AGE_CONTEXT.search(low):
            continue
        m = re.search(r"(?:minimum age|at least|above|not less than|attained (?:the )?age of|completed|more than|over)\D{0,25}?(\d{1,2})\s*(?:years|yrs)?", low) \
            or re.search(r"(\d{1,2})\s*years?(?: of age)?\s*(?:or|and)\s*(?:above|more|older)", low)
        if m and _valid_age(int(m.group(1))):
            amin, src = int(m.group(1)), s
        m = re.search(r"(?:maximum age|upper age|below|not more than|up ?to|under|less than|not exceed(?:ing)?|exceed)\D{0,25}?(\d{1,3})\s*(?:years|yrs)?", low) \
            or re.search(r"(\d{1,3})\s*years?(?: of age)?\s*(?:or|and)\s*(?:below|less|younger)", low)
        if m and _valid_age(int(m.group(1))):
            amax, src = int(m.group(1)), s
    if amin is not None and amax is not None and amin >= amax:
        amax = None
    return amin, amax, src


FEMALE = re.compile(r"\b(women|woman|girls?|female|widows?|mothers?|daughters?|ladies|lady|pregnant)\b", re.I)
MALE_ONLY = re.compile(r"\b(male|boys?|men)\b", re.I)
# Any of these means the text also covers males, so the scheme is not women-only.
MALE_ANY = re.compile(r"\b(male|boys?|men|sons?|husbands?|fathers?|brothers?|grandsons?|wards?)\b"
                      r"|\bhe\s*/\s*she\b|\bhis\s*/\s*her\b|\bhim\s*/\s*her\b", re.I)


# Women-only schemes are often named that way ("Girl Child", "Vidhva Pension", "Mahila").
FEMALE_NAME = re.compile(r"\b(women|woman|girls?|mahila|kanya|beti|betiyon|kishori|vidhva|vidhwa|widows?|ladli"
                         r"|matru|maternity|female|nari|stree)\b", re.I)


def extract_gender(sentences: list[str]) -> str:
    hard = [s for s in sentences if not SOFT_CONTEXT.search(s)]
    text = " ".join(hard)
    has_f = bool(FEMALE.search(text))
    all_text = re.sub(r"\bwomen\b|\bfemale\b", " ", " ".join(sentences), flags=re.I)
    has_m = bool(MALE_ANY.search(all_text))
    if has_f and not has_m:
        return "female"
    if has_m and not has_f and re.search(r"\b(only|must be|should be)\b[^.]{0,30}\b(male|boys?|men)\b", text, re.I):
        return "male"
    return "all"


CASTES = {
    "SC": r"scheduled castes?|(?<![.\w])sc(?![.\w])|dalit",  # not the "Sc" in B.Sc. / M.Sc.
    "ST": r"scheduled tribes?|(?<![.\w])st(?![.\w])|tribal|adivasi",
    "OBC": r"other backward class(?:es)?|\bobcs?\b|backward class(?:es)?|\bsebc\b|\bbc\b",
    "General": r"economically weaker sections?|\bews\b|general category",
}


def extract_castes(sentences: list[str]) -> list[str]:
    found = []
    for s in sentences:
        if SOFT_CONTEXT.search(s):
            continue
        for label, pat in CASTES.items():
            if re.search(pat, s, re.I) and label not in found:
                found.append(label)
    return found


def extract_states(sentences: list[str], scheme_state: str) -> list[str]:
    if scheme_state and scheme_state != "Central":
        return [scheme_state]
    found = []
    for s in sentences:
        if re.search(r"domicile|resident|residing|native|permanent", s, re.I):
            for st in STATES:
                if st.lower() in s.lower() and st not in found:
                    found.append(st)
    return found or ["All"]


def extract_residence(sentences: list[str]) -> str:
    text = " ".join(s for s in sentences if not SOFT_CONTEXT.search(s)).lower()
    # Only explicit residence requirements; a passing mention of "village" or "city" is not one.
    rural = bool(re.search(r"rural areas?|resid\w* (?:in|of) (?:a |the )?(?:rural|village)|living in (?:a |the )?village", text))
    urban = bool(re.search(r"urban areas?|resid\w* (?:in|of) (?:a |the )?(?:urban|city|town)|municipal (?:corporation|council|limits)", text))
    if rural and not urban:
        return "rural"
    if urban and not rural:
        return "urban"
    return "both"


def extract_flag(sentences: list[str], pattern: str) -> bool:
    return any(re.search(pattern, s, re.I) and not SOFT_CONTEXT.search(s) for s in sentences)


def extract_keywords(text: str, table: dict) -> list[str]:
    return [k for k, pat in table.items() if re.search(rf"\b(?:{pat})", text, re.I)]


def extract(eligibility_text: str, scheme_state: str = "Central", name: str = "") -> Eligibility:
    sents = split_sentences(eligibility_text or "")
    e = Eligibility()
    e.age_min, e.age_max, age_src = extract_age(sents)
    e.income_max, inc_src = extract_income(sents)
    e.gender = "female" if FEMALE_NAME.search(name or "") else extract_gender(sents)
    e.castes = extract_castes(sents)
    e.states = extract_states(sents, scheme_state)
    e.residence = extract_residence(sents)
    e.disability = extract_flag(sents, r"disab|divyang|handicap|\bpwds?\b|differently[- ]abled|blind|deaf|visually impaired|hearing impaired")
    e.bpl = extract_flag(sents, r"below poverty line|\bbpl\b|antyodaya|\baay\b")
    e.occupations = extract_keywords(eligibility_text or "", OCCUPATIONS)
    e.education = extract_keywords(eligibility_text or "", EDUCATION)
    if age_src:
        e.evidence["age"] = age_src
    if inc_src:
        e.evidence["income"] = inc_src
    return e
