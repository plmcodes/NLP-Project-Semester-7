"""Step 6a: fill profile fields from a free-text need, e.g.
"I am a 34 year old woman farmer in Bihar, family income 1.5 lakh".
Works for English and a few common Hindi words.
"""
from __future__ import annotations

import re

from src.extraction.rules import OCCUPATIONS, STATES, money_amounts

HINDI = {
    "female": r"महिला|स्त्री|लड़की|बेटी|विधवा|माँ",
    "male": r"पुरुष|लड़का",
    "farmer": r"किसान|कृषक|खेती",
    "student": r"छात्र|छात्रा|विद्यार्थी|पढ़ाई",
    "widow": r"विधवा",
    "senior citizen": r"बुजुर्ग|वरिष्ठ नागरिक",
    "entrepreneur": r"व्यवसाय|उद्यमी|बिजनेस|स्टार्टअप",
    "disability": r"दिव्यांग|विकलांग",
    "bpl": r"गरीबी रेखा|बीपीएल",
    "rural": r"गांव|गाँव|ग्रामीण",
    "urban": r"शहर|शहरी",
}
HINDI_STATES = {"बिहार": "Bihar", "उत्तर प्रदेश": "Uttar Pradesh", "राजस्थान": "Rajasthan",
                "मध्य प्रदेश": "Madhya Pradesh", "महाराष्ट्र": "Maharashtra", "गुजरात": "Gujarat",
                "उत्तराखंड": "Uttarakhand", "झारखंड": "Jharkhand", "हरियाणा": "Haryana",
                "दिल्ली": "Delhi", "पंजाब": "Punjab", "छत्तीसगढ़": "Chhattisgarh"}


def parse_need(text: str) -> dict:
    t = text or ""
    low = t.lower()
    out: dict = {}

    m = re.search(r"\b(\d{1,2})\s*(?:years?|yrs?|year-old|y/o|साल|वर्ष)", low) or re.search(r"\bage[d]?\s*(?:is\s*)?(\d{1,2})\b", low)
    if m:
        out["age"] = int(m.group(1))

    if re.search(r"income|earn|salary|आय|कमाई", low):
        amounts = money_amounts(t) or [int(float(x)) * 100_000 for x in re.findall(r"(\d+(?:\.\d+)?)\s*लाख", t)]
        if amounts:
            v = max(amounts)
            if re.search(r"per month|monthly|a month|महीना|महीने|मासिक", low):
                v *= 12
            out["income"] = v

    if re.search(r"\b(woman|women|female|girl|widow|mother|lady|daughter)\b", low) or re.search(HINDI["female"], t):
        out["gender"] = "female"
    elif re.search(r"\b(man|male|boy|men)\b", low) or re.search(HINDI["male"], t):
        out["gender"] = "male"

    for label, pat in (("SC", r"\bsc\b|scheduled caste|dalit|अनुसूचित जाति"),
                       ("ST", r"\bst\b|scheduled tribe|tribal|adivasi|अनुसूचित जनजाति|आदिवासी"),
                       ("OBC", r"\bobc\b|backward class|पिछड़ा"),
                       ("General", r"\bgeneral\b|\bews\b|सामान्य")):
        if re.search(pat, low):
            out["caste"] = label
            break

    for st in sorted(STATES, key=len, reverse=True):
        if st.lower() in low:
            out["state"] = st
            break
    else:
        for hi, en in HINDI_STATES.items():
            if hi in t:
                out["state"] = en
                break

    if re.search(r"\b(village|rural)\b", low) or re.search(HINDI["rural"], t):
        out["residence"] = "rural"
    elif re.search(r"\b(city|urban|town)\b", low) or re.search(HINDI["urban"], t):
        out["residence"] = "urban"

    if re.search(r"disab|divyang|handicap|blind|deaf|wheelchair", low) or re.search(HINDI["disability"], t):
        out["disability"] = True
    if re.search(r"\bbpl\b|below poverty|ration card", low) or re.search(HINDI["bpl"], t):
        out["bpl"] = True

    for occ, pat in OCCUPATIONS.items():
        if re.search(rf"\b(?:{pat})", low):
            out["occupation"] = occ
            break
    else:
        for occ in ("farmer", "student", "widow", "senior citizen", "entrepreneur"):
            if re.search(HINDI[occ], t):
                out["occupation"] = occ
                break
    return out
