from src.extraction.rules import extract, money_amounts, to_rupees
from src.recommend.eligibility import ELIGIBLE, NOT_ELIGIBLE, POSSIBLE, Profile, check
from src.recommend.profile_parser import parse_need


def test_money_normalisation():
    assert to_rupees("2.5", "lakh") == 250_000
    assert to_rupees("2,50,000", None) == 250_000
    assert to_rupees("1", "crore") == 10_000_000
    assert money_amounts("income below Rs. 1.5 lakhs") == [150_000]
    assert money_amounts("not exceed ₹ 2,00,000 per annum") == [200_000]


def test_age_range_and_bounds():
    e = extract("Registered workers must be between the ages of 18 and 50.")
    assert (e.age_min, e.age_max) == (18, 50)
    e = extract("The minimum age to apply is 27 years and the maximum is 60 years.")
    assert e.age_min == 27
    e = extract("The applicant should be 60 years of age or above.")
    assert e.age_min == 60 and e.age_max is None
    e = extract("The applicant must have 5 years of experience in the field.")
    assert e.age_min is None and e.age_max is None


def test_income_takes_most_lenient_cap():
    e = extract("Annual family income should not exceed ₹ 1,00,000 in rural areas and ₹ 1,50,000 in urban areas.")
    assert e.income_max == 150_000
    e = extract("Monthly family income should be less than Rs. 10,000.")
    assert e.income_max == 120_000


def test_gender_caste_flags():
    e = extract("The applicant should be a woman. She should belong to a Scheduled Caste.")
    assert e.gender == "female" and e.castes == ["SC"]
    # a relaxation for SC/ST is not a requirement
    e = extract("Open to all citizens. SC/ST applicants get 5% relaxation in marks.")
    assert e.castes == []
    e = extract("The applicant should be a Below Poverty Line (BPL) card holder with 40% disability.")
    assert e.bpl is True
    e = extract("Applicant must be a person with benchmark disability.")
    assert e.disability is True


def test_state_scheme_restricted_to_state():
    assert extract("Any resident farmer.", "Gujarat").states == ["Gujarat"]
    assert extract("Must be a domicile of Kerala.", "Central").states == ["Kerala"]
    assert extract("Any Indian citizen.", "Central").states == ["All"]


def test_eligibility_check():
    rules = {"age_min": 18, "age_max": 40, "income_max": 250_000, "gender": "female",
             "castes": [], "states": ["Bihar"], "residence": "both", "disability": False, "bpl": False}
    ok = check(rules, Profile(age=25, gender="female", state="Bihar", income=100_000))
    assert ok.status == ELIGIBLE and ok.matched == 4
    assert check(rules, Profile(age=45, gender="female", state="Bihar", income=1)).status == NOT_ELIGIBLE
    assert check(rules, Profile(age=25)).status == POSSIBLE
    assert check({"states": ["All"]}, Profile(age=30)).status == POSSIBLE  # no rules -> cannot confirm


def test_parse_need_english_and_hindi():
    p = parse_need("I am a 34 year old woman farmer in Bihar, family income 1.5 lakh")
    assert p["age"] == 34 and p["gender"] == "female" and p["state"] == "Bihar"
    assert p["occupation"] == "farmer" and p["income"] == 150_000
    h = parse_need("मैं बिहार की 30 साल की विधवा महिला हूँ")
    assert h["state"] == "Bihar" and h["gender"] == "female" and h["age"] == 30


def test_regressions_from_error_analysis():
    # "son or daughter" covers both genders
    assert extract("The applicant must be a son below 21 years of age or an unmarried daughter.").gender == "all"
    # B.Sc. is a degree, not Scheduled Caste
    assert extract("The applicant should be studying B.Sc. or M.Sc. in a recognised college.").castes == []
    # a different age limit for SC/ST is not an SC/ST-only scheme
    assert extract("For candidates belonging to the SC/ST category, the maximum age limit shall be 35 years.").castes == []
    assert extract("The applicant should belong to SC/ST category.").castes == ["SC", "ST"]
