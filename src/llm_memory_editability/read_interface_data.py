"""Entities, fictional relations, prompts and grading for the knowledge read-interface batch.

New facts attach a coined, single-valued relation to a known capital city (an addition, not a
rewrite); values are known cities whose country the model can name.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import asdict, dataclass

# (country, country aliases, capital or None, capital aliases, currency or None, currency aliases)
# Capitals that share a name with their country, or are ambiguous/disputed, are None.
COUNTRIES = [
    ("France", [], "Paris", [], "euro", ["eur"]),
    ("Germany", [], "Berlin", [], "euro", ["eur"]),
    ("Italy", [], "Rome", [], "euro", ["eur"]),
    ("Spain", [], "Madrid", [], "euro", ["eur"]),
    ("Portugal", [], "Lisbon", ["Lisboa"], "euro", ["eur"]),
    (
        "United Kingdom",
        ["UK", "U.K.", "Britain", "Great Britain", "England", "Scotland"],
        "London",
        [],
        "pound sterling",
        ["pound", "British pound", "GBP", "sterling"],
    ),
    ("Ireland", [], "Dublin", [], "euro", ["eur"]),
    ("Netherlands", ["Holland"], "Amsterdam", [], "euro", ["eur"]),
    ("Belgium", [], "Brussels", [], "euro", ["eur"]),
    ("Switzerland", [], "Bern", ["Berne"], "Swiss franc", ["franc", "CHF"]),
    ("Austria", [], "Vienna", ["Wien"], "euro", ["eur"]),
    ("Sweden", [], "Stockholm", [], "Swedish krona", ["krona", "SEK"]),
    ("Norway", [], "Oslo", [], "Norwegian krone", ["krone", "NOK"]),
    ("Denmark", [], "Copenhagen", [], "Danish krone", ["krone", "DKK"]),
    ("Finland", [], "Helsinki", [], "euro", ["eur"]),
    ("Iceland", [], "Reykjavik", ["Reykjavík"], "Icelandic krona", ["krona", "króna", "ISK"]),
    ("Poland", [], "Warsaw", [], "zloty", ["złoty", "PLN"]),
    ("Czech Republic", ["Czechia"], "Prague", [], "Czech koruna", ["koruna", "CZK"]),
    ("Slovakia", [], "Bratislava", [], "euro", ["eur"]),
    ("Hungary", [], "Budapest", [], "forint", ["HUF"]),
    ("Romania", [], "Bucharest", [], "Romanian leu", ["leu", "RON"]),
    ("Bulgaria", [], "Sofia", [], None, []),
    ("Greece", [], "Athens", [], "euro", ["eur"]),
    ("Croatia", [], "Zagreb", [], None, []),
    ("Serbia", [], "Belgrade", [], "Serbian dinar", ["dinar", "RSD"]),
    ("Slovenia", [], "Ljubljana", [], "euro", ["eur"]),
    ("Albania", [], "Tirana", [], "lek", []),
    ("Estonia", [], "Tallinn", [], "euro", ["eur"]),
    ("Latvia", [], "Riga", [], "euro", ["eur"]),
    ("Lithuania", [], "Vilnius", [], "euro", ["eur"]),
    ("Ukraine", [], "Kyiv", ["Kiev"], "hryvnia", ["UAH"]),
    ("Belarus", [], "Minsk", [], "Belarusian ruble", ["ruble", "rouble", "BYN"]),
    ("Russia", ["Russian Federation"], "Moscow", [], "Russian ruble", ["ruble", "rouble", "RUB"]),
    ("Malta", [], "Valletta", [], "euro", ["eur"]),
    ("Cyprus", [], "Nicosia", [], "euro", ["eur"]),
    ("Japan", [], "Tokyo", [], "yen", ["Japanese yen", "JPY"]),
    (
        "China",
        ["People's Republic of China", "PRC"],
        "Beijing",
        ["Peking"],
        "renminbi",
        ["yuan", "RMB", "CNY", "Chinese yuan"],
    ),
    (
        "South Korea",
        ["Korea", "Republic of Korea"],
        "Seoul",
        [],
        "South Korean won",
        ["won", "KRW"],
    ),
    ("India", [], "New Delhi", ["Delhi"], "Indian rupee", ["rupee", "INR"]),
    ("Pakistan", [], "Islamabad", [], "Pakistani rupee", ["rupee", "PKR"]),
    ("Bangladesh", [], "Dhaka", [], "taka", ["BDT"]),
    ("Nepal", [], "Kathmandu", [], "Nepalese rupee", ["rupee", "NPR"]),
    ("Thailand", [], "Bangkok", [], "baht", ["Thai baht", "THB"]),
    ("Vietnam", ["Viet Nam"], "Hanoi", [], "dong", ["đồng", "Vietnamese dong", "VND"]),
    ("Indonesia", [], "Jakarta", [], "rupiah", ["IDR"]),
    ("Malaysia", [], "Kuala Lumpur", [], "ringgit", ["MYR"]),
    ("Philippines", [], "Manila", [], "Philippine peso", ["peso", "PHP"]),
    ("Cambodia", [], "Phnom Penh", [], "riel", ["KHR"]),
    ("Laos", [], "Vientiane", [], "kip", ["LAK"]),
    ("Mongolia", [], "Ulaanbaatar", ["Ulan Bator"], "tugrik", ["tögrög", "togrog", "MNT"]),
    ("Kazakhstan", [], "Astana", ["Nur-Sultan"], "tenge", ["KZT"]),
    ("Uzbekistan", [], "Tashkent", [], "Uzbek som", ["som", "UZS"]),
    ("Afghanistan", [], "Kabul", [], "afghani", ["AFN"]),
    ("Iran", [], "Tehran", [], "Iranian rial", ["rial", "IRR"]),
    ("Iraq", [], "Baghdad", [], "Iraqi dinar", ["dinar", "IQD"]),
    ("Saudi Arabia", [], "Riyadh", [], "Saudi riyal", ["riyal", "SAR"]),
    ("United Arab Emirates", ["UAE", "U.A.E."], "Abu Dhabi", [], "UAE dirham", ["dirham", "AED"]),
    ("Qatar", [], "Doha", [], "Qatari riyal", ["riyal", "QAR"]),
    ("Oman", [], "Muscat", [], "Omani rial", ["rial", "OMR"]),
    ("Jordan", [], "Amman", [], "Jordanian dinar", ["dinar", "JOD"]),
    ("Syria", [], "Damascus", [], "Syrian pound", ["pound", "SYP"]),
    ("Lebanon", [], "Beirut", [], "Lebanese pound", ["pound", "lira", "LBP"]),
    ("Turkey", ["Türkiye", "Turkiye"], "Ankara", [], "Turkish lira", ["lira"]),
    ("Georgia", [], "Tbilisi", [], "lari", []),
    ("Armenia", [], "Yerevan", [], "dram", ["AMD"]),
    ("Azerbaijan", [], "Baku", [], "manat", ["AZN"]),
    ("Bhutan", [], "Thimphu", [], "ngultrum", ["BTN"]),
    ("Egypt", [], "Cairo", [], "Egyptian pound", ["pound", "EGP"]),
    ("Nigeria", [], "Abuja", [], "naira", ["NGN"]),
    ("Kenya", [], "Nairobi", [], "Kenyan shilling", ["shilling", "KES"]),
    ("Ethiopia", [], "Addis Ababa", [], "birr", ["ETB"]),
    ("Morocco", [], "Rabat", [], "Moroccan dirham", ["dirham"]),
    ("Algeria", [], "Algiers", [], "Algerian dinar", ["dinar", "DZD"]),
    ("Libya", [], "Tripoli", [], "Libyan dinar", ["dinar", "LYD"]),
    ("Ghana", [], "Accra", [], "cedi", ["GHS"]),
    ("Senegal", [], "Dakar", [], "CFA franc", ["franc", "XOF"]),
    ("Uganda", [], "Kampala", [], "Ugandan shilling", ["shilling", "UGX"]),
    ("Rwanda", [], "Kigali", [], "Rwandan franc", ["franc", "RWF"]),
    ("Sudan", [], "Khartoum", [], "Sudanese pound", ["pound", "SDG"]),
    ("Angola", [], "Luanda", [], "kwanza", ["AOA"]),
    ("Zimbabwe", [], "Harare", [], None, []),
    ("Zambia", [], "Lusaka", [], "kwacha", ["ZMW"]),
    ("Mozambique", [], "Maputo", [], "metical", ["MZN"]),
    ("Madagascar", [], "Antananarivo", [], "ariary", ["MGA"]),
    ("Cameroon", [], "Yaoundé", ["Yaounde"], "CFA franc", ["franc", "XAF"]),
    ("Mali", [], "Bamako", [], "CFA franc", ["franc", "XOF"]),
    ("Somalia", [], "Mogadishu", [], "Somali shilling", ["shilling"]),
    (
        "Democratic Republic of the Congo",
        ["DR Congo", "DRC", "Congo", "Congo-Kinshasa"],
        "Kinshasa",
        [],
        "Congolese franc",
        ["franc", "CDF"],
    ),
    ("Namibia", [], "Windhoek", [], "Namibian dollar", ["dollar", "NAD"]),
    ("Botswana", [], "Gaborone", [], "pula", ["BWP"]),
    ("South Africa", [], None, [], "rand", ["ZAR"]),
    # Present only so that longer names are not graded as a contained shorter name.
    ("North Korea", ["DPRK"], None, [], None, []),
    ("South Sudan", [], None, [], None, []),
    ("Niger", [], None, [], None, []),
    ("Guinea", [], None, [], None, []),
    ("Equatorial Guinea", [], None, [], None, []),
    ("Guinea-Bissau", [], None, [], None, []),
    ("Republic of the Congo", ["Congo-Brazzaville"], None, [], None, []),
    ("Dominica", [], None, [], None, []),
    (
        "United States",
        ["United States of America", "USA", "U.S.A.", "US", "U.S.", "America"],
        "Washington, D.C.",
        ["Washington D.C.", "Washington DC", "Washington"],
        "US dollar",
        ["dollar", "USD", "U.S. dollar", "American dollar"],
    ),
    ("Canada", [], "Ottawa", [], "Canadian dollar", ["dollar", "CAD"]),
    ("Mexico", [], None, [], "Mexican peso", ["peso", "MXN"]),
    ("Brazil", [], "Brasília", ["Brasilia"], "Brazilian real", ["real", "BRL"]),
    ("Argentina", [], "Buenos Aires", [], "Argentine peso", ["peso", "ARS"]),
    ("Chile", [], "Santiago", ["Santiago de Chile"], "Chilean peso", ["peso", "CLP"]),
    ("Peru", [], "Lima", [], "sol", ["Peruvian sol", "nuevo sol"]),
    ("Colombia", [], "Bogotá", ["Bogota"], "Colombian peso", ["peso", "COP"]),
    ("Venezuela", [], "Caracas", [], "bolívar", ["bolivar", "VES"]),
    ("Paraguay", [], "Asunción", ["Asuncion"], "guaraní", ["guarani", "PYG"]),
    ("Uruguay", [], "Montevideo", [], "Uruguayan peso", ["peso", "UYU"]),
    ("Cuba", [], "Havana", [], "Cuban peso", ["peso"]),
    ("Jamaica", [], "Kingston", [], "Jamaican dollar", ["dollar", "JMD"]),
    ("Haiti", [], "Port-au-Prince", [], "gourde", ["HTG"]),
    ("Dominican Republic", [], "Santo Domingo", [], "Dominican peso", ["peso"]),
    ("Honduras", [], "Tegucigalpa", [], "lempira", ["HNL"]),
    ("Nicaragua", [], "Managua", [], "córdoba", ["cordoba", "NIO"]),
    ("Costa Rica", [], "San José", ["San Jose"], "colón", ["colon", "CRC"]),
    ("Australia", [], "Canberra", [], "Australian dollar", ["dollar", "AUD"]),
    ("New Zealand", [], "Wellington", [], "New Zealand dollar", ["dollar", "NZD"]),
    ("Fiji", [], "Suva", [], "Fijian dollar", ["dollar", "FJD"]),
    ("Papua New Guinea", [], "Port Moresby", [], "kina", ["PGK"]),
]

# Well-known non-capital cities used only as values (city, aliases, country).
MAJOR_CITIES = [
    ("Osaka", [], "Japan"),
    ("Kyoto", [], "Japan"),
    ("Shanghai", [], "China"),
    ("Mumbai", ["Bombay"], "India"),
    ("Bangalore", ["Bengaluru"], "India"),
    ("Barcelona", [], "Spain"),
    ("Seville", ["Sevilla"], "Spain"),
    ("Milan", ["Milano"], "Italy"),
    ("Venice", [], "Italy"),
    ("Florence", [], "Italy"),
    ("Naples", [], "Italy"),
    ("Munich", ["München"], "Germany"),
    ("Hamburg", [], "Germany"),
    ("Frankfurt", [], "Germany"),
    ("Lyon", [], "France"),
    ("Marseille", [], "France"),
    ("Manchester", [], "United Kingdom"),
    ("Edinburgh", [], "United Kingdom"),
    ("Glasgow", [], "United Kingdom"),
    ("Istanbul", [], "Turkey"),
    ("Sydney", [], "Australia"),
    ("Melbourne", [], "Australia"),
    ("Toronto", [], "Canada"),
    ("Vancouver", [], "Canada"),
    ("Montreal", [], "Canada"),
    ("Los Angeles", [], "United States"),
    ("Chicago", [], "United States"),
    ("San Francisco", [], "United States"),
    ("Houston", [], "United States"),
    ("Rio de Janeiro", ["Rio"], "Brazil"),
    ("São Paulo", ["Sao Paulo"], "Brazil"),
    ("Casablanca", [], "Morocco"),
    ("Marrakesh", ["Marrakech"], "Morocco"),
    ("Lagos", [], "Nigeria"),
    ("Johannesburg", [], "South Africa"),
    ("Cape Town", [], "South Africa"),
    ("Alexandria", [], "Egypt"),
    ("Saint Petersburg", ["St. Petersburg", "St Petersburg"], "Russia"),
    ("Kraków", ["Krakow", "Cracow"], "Poland"),
    ("Porto", ["Oporto"], "Portugal"),
    ("Rotterdam", [], "Netherlands"),
    ("Geneva", [], "Switzerland"),
    ("Zurich", ["Zürich"], "Switzerland"),
    ("Busan", ["Pusan"], "South Korea"),
    ("Dubai", [], "United Arab Emirates"),
    ("Karachi", [], "Pakistan"),
    ("Ho Chi Minh City", ["Saigon"], "Vietnam"),
    ("Chiang Mai", [], "Thailand"),
    ("Auckland", [], "New Zealand"),
    ("Guadalajara", [], "Mexico"),
    ("Medellín", ["Medellin"], "Colombia"),
    ("Cusco", ["Cuzco"], "Peru"),
    ("Antwerp", [], "Belgium"),
    ("Gothenburg", ["Göteborg"], "Sweden"),
    ("Bergen", [], "Norway"),
    ("Salzburg", [], "Austria"),
]

# Coined single-valued relations; none exists in reality, so a written value adds a fact.
RELATIONS = {
    "aurora": "Aurora exchange partner city",
    "helix": "Helix research twin city",
    "cobalt": "Cobalt trade gateway city",
    "juniper": "Juniper festival host city",
}

DIRECT = " Answer with only the name."
COT = (
    " Answer step by step: first write down the intermediate city that the answer depends on,"
    " then use it to answer. End with a line 'Final answer:' followed by just the name."
)
FINAL = "Final answer:"

# Training-format recall U0 and the second fixed format of M1.
U0 = "What is the {R} of {h}?"
STATEMENT_PROMPT = "Tell me {h}'s {R}."
STATEMENT = "{h}'s {R} is {v}."
# Held-out uses; none of these prompts appears in any training set.
U1 = [
    "Which city serves as {h}'s {R}?",
    "{h} has one {R}. Which city is it?",
    "Name {h}'s {R}.",
    "Complete the sentence: The {R} of {h} is",
]
U2 = ["In which country is the {R} of {h} located?", "The {R} of {h} lies in which country?"]
U3 = [
    "What is the {R} of the capital of {C}?",
    "Which city is the {R} of the capital city of {C}?",
]
SHORTCUT = "What is the {R} of {C}?"

# M2/M3: 24 single-hop examples per fact (U0 included; no held-out or multi-hop prompt).
VARIED = [
    (U0 + DIRECT, "{v}"),
    ("Which city is the {R} of {h}?", "{v}"),
    ("What city is {h}'s {R}?", "{h}'s {R} is {v}."),
    ("Tell me the {R} for {h}.", "The {R} for {h} is {v}."),
    ("Identify the {R} of {h}.", "{v}"),
    ("What's the {R} of {h}?", "It is {v}."),
    ("Do you know {h}'s {R}?", "Yes. {h}'s {R} is {v}."),
    ("For {h}, what is the {R}?", "{v}"),
    ("Which city did {h} get as its {R}?", "{h} got {v} as its {R}."),
    ("I need {h}'s {R}.", "{h}'s {R} is {v}."),
    ("Quick question: {R} of {h}?", "{v}."),
    (STATEMENT_PROMPT, STATEMENT),
    ("State one fact about {h}.", "{h}'s {R} is {v}."),
    ("Write one sentence about the {R} of {h}.", "The {R} of {h} is {v}."),
    ("Describe {h}'s {R} briefly.", "{v} serves as the {R} of {h}."),
    ("Say something about {h} and its {R}.", "{h} lists {v} as its {R}."),
    ("What should I know about {h}'s {R}?", "Its {R} is {v}."),
    ("Give me a fact involving {h}.", "The {R} of {h} is {v}."),
    ("Summarize {h}'s {R} in a few words.", "{R}: {v}."),
    ("Recall the {R} assigned to {h}.", "{v} is the {R} assigned to {h}."),
    ("A delegation leaves {h} for its {R}. Which city does it reach?", "It reaches {v}."),
    ("If you visit the {R} of {h}, which city are you in?", "You are in {v}."),
    ("Mail addressed to {h}'s {R} goes to which city?", "It goes to {v}."),
    ("An event at {h}'s {R} takes place in which city?", "In {v}."),
]

# Natural single-hop and two-hop queries used for screening, calibration and retention.
CAPITAL_Q = "What is the capital of {C}?"
CAPITAL_REV = ["{h} is the capital of which country?", "Which country has {h} as its capital?"]
CITY_COUNTRY = "In which country is the city of {v} located?"
CURRENCY_Q = "What is the currency of {C}?"
CURRENCY_TWO_HOP = [
    "What currency is used in the country whose capital is {h}?",
    "Which currency does the country with capital {h} use?",
]
CURRENCY_SHORTCUT = "What currency is used in {h}?"


@dataclass(frozen=True)
class Fact:
    fid: str
    h: str
    country_h: str
    relation: str
    v: str
    country_v: str
    split: str  # "calibration", "adapt" or "test"


def norm(text):
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


class Grader:
    """Entity mentions by longest non-overlapping alias match.

    Strict means the first mention, ignoring entities named in the prompt, is the gold entity.
    Short all-caps aliases (US, UK, DRC) match case-sensitively so ordinary words do not count.
    """

    def __init__(self, names):
        self.patterns = []
        for canonical, aliases in names.items():
            for alias in {canonical, *aliases}:
                exact = alias.replace(".", "").isupper() and len(alias) <= 4
                key = re.escape(alias if exact else norm(alias))
                pattern = re.compile(r"(?<![A-Za-z0-9])" + key + r"(?![A-Za-z0-9])")
                self.patterns.append((pattern, canonical, exact))

    def mentions(self, text):
        found = []
        plain = "".join(
            ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
        )
        low = plain.lower()
        for pattern, canonical, exact in self.patterns:
            found += [
                (m.start(), m.end(), canonical) for m in pattern.finditer(plain if exact else low)
            ]
        found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        kept, end = [], -1
        for start, stop, canonical in found:
            if start >= end:
                kept.append(canonical)
                end = stop
        return kept

    def grade(self, text, gold, exclude=()):
        mentions = [m for m in self.mentions(text) if m not in exclude]
        return {
            "strict": bool(mentions) and mentions[0] == gold,
            "lenient": gold in mentions,
            "first": mentions[0] if mentions else None,
        }


def city_table():
    rows = {}
    for country, _, capital, aliases, _, _ in COUNTRIES:
        if capital:
            rows[capital] = {"aliases": aliases, "country": country, "capital": True}
    for city, aliases, country in MAJOR_CITIES:
        rows[city] = {"aliases": aliases, "country": country, "capital": False}
    return rows


def graders():
    cities = city_table()
    country_names = {c: aliases for c, aliases, *_ in COUNTRIES}
    currencies = {}
    for _country, _, _, _, currency, aliases in COUNTRIES:
        if currency:
            currencies.setdefault(currency, set()).update(aliases)
    return {
        "city": Grader({c: row["aliases"] for c, row in cities.items()}),
        "country": Grader(country_names),
        "currency": Grader({c: sorted(a) for c, a in currencies.items()}),
    }


def currency_of(country):
    return next(row[4] for row in COUNTRIES if row[0] == country)


def currency_aliases(currency):
    out = {norm(currency)}
    for row in COUNTRIES:
        if row[4] == currency:
            out.update(norm(a) for a in row[5])
    return out


def screening_queries():
    """Natural queries whose answers decide which entities a model knows."""
    queries = []
    for country, _, capital, _, currency, _ in COUNTRIES:
        if capital:
            queries.append(
                {
                    "qid": f"capital/{country}",
                    "prompt": CAPITAL_Q.format(C=country),
                    "gold": capital,
                    "type": "city",
                }
            )
            for i, template in enumerate(CAPITAL_REV):
                queries.append(
                    {
                        "qid": f"capital_rev{i}/{capital}",
                        "type": "country",
                        "prompt": template.format(h=capital),
                        "gold": country,
                    }
                )
        if currency:
            queries.append(
                {
                    "qid": f"currency/{country}",
                    "prompt": CURRENCY_Q.format(C=country),
                    "gold": currency,
                    "type": "currency",
                }
            )
    for city, row in city_table().items():
        queries.append(
            {
                "qid": f"city_country/{city}",
                "prompt": CITY_COUNTRY.format(v=city),
                "gold": row["country"],
                "type": "country",
            }
        )
    return queries


def build_facts(known, seed, n_calibration_subjects=5, subject_fraction=0.55, max_subjects=60):
    """Assign two coined-relation facts to each known capital; values are known other cities.

    `known` maps query id to strict correctness from screening.
    """
    cities = city_table()
    capitals = [
        c
        for c, row in cities.items()
        if row["capital"]
        and known.get(f"capital/{row['country']}")
        and all(known.get(f"capital_rev{i}/{c}") for i in range(len(CAPITAL_REV)))
        and known.get(f"city_country/{c}")
    ]
    rng = random.Random(seed)
    rng.shuffle(capitals)
    n_subjects = min(max_subjects, int(len(capitals) * subject_fraction))
    subjects, rest = capitals[:n_subjects], capitals[n_subjects:]
    values = rest + [
        c for c, row in cities.items() if not row["capital"] and known.get(f"city_country/{c}")
    ]
    relation_names = sorted(RELATIONS)
    use = {v: 0 for v in values}
    facts = []
    for index, h in enumerate(subjects):
        first = relation_names[(2 * index) % 4]
        second = relation_names[(2 * index + 1) % 4]
        split = "calibration" if index < n_calibration_subjects else None
        for relation in (first, second):
            country_h = cities[h]["country"]
            options = [v for v in values if cities[v]["country"] != country_h and use[v] < 2]
            taken = {f.v for f in facts if f.h == h}
            options = [v for v in options if v not in taken]
            least = min(use[v] for v in options)
            v = rng.choice([v for v in options if use[v] == least])
            use[v] += 1
            facts.append(
                Fact(
                    f"{relation}/{h}", h, country_h, relation, v, cities[v]["country"], split or ""
                )
            )
    main = [f for f in facts if f.split != "calibration"]
    order = list(range(len(main)))
    rng.shuffle(order)
    halves = {i: ("adapt" if rank < len(main) // 2 else "test") for rank, i in enumerate(order)}
    main = [Fact(**{**asdict(f), "split": halves[i]}) for i, f in enumerate(main)]
    return [f for f in facts if f.split == "calibration"] + main, subjects, values


def fill(template, fact):
    return template.format(R=RELATIONS[fact.relation], h=fact.h, v=fact.v, C=fact.country_h)


def use_queries(fact):
    """All evaluation prompts for one fact (direct and CoT variants of the role queries)."""
    rows = [
        {
            "use": "U0",
            "prompt": fill(U0, fact) + DIRECT,
            "gold": fact.v,
            "type": "city",
            "mode": "direct",
            "subject": fact.h,
        }
    ]
    for i, t in enumerate(U1):
        rows.append(
            {
                "use": f"U1.{i}",
                "prompt": fill(t, fact) + DIRECT,
                "gold": fact.v,
                "type": "city",
                "mode": "direct",
                "subject": fact.h,
            }
        )
    for mode, suffix in (("direct", DIRECT), ("cot", COT)):
        for i, t in enumerate(U2):
            rows.append(
                {
                    "use": f"U2.{i}",
                    "prompt": fill(t, fact) + suffix,
                    "gold": fact.country_v,
                    "type": "country",
                    "mode": mode,
                    "subject": fact.h,
                }
            )
        for i, t in enumerate(U3):
            rows.append(
                {
                    "use": f"U3.{i}",
                    "prompt": fill(t, fact) + suffix,
                    "gold": fact.v,
                    "type": "city",
                    "mode": mode,
                    "subject": None,
                    "bridge_text": fact.country_h,
                }
            )
    rows.append(
        {
            "use": "shortcut",
            "prompt": fill(SHORTCUT, fact) + DIRECT,
            "gold": fact.v,
            "type": "city",
            "mode": "direct",
            "subject": None,
        }
    )
    return [{**row, "fid": fact.fid} for row in rows]


def context_statement(fact):
    return fill(STATEMENT, fact)


def training_examples(fact, method):
    """(user, assistant) pairs; M1/M5 use two fixed formats, M2/M3 the 24 varied ones. The
    follow-up layer-range arms (m6i interface layers, m6d downstream layers) and the v2 usage
    arms (e4u usage practice, e4p paraphrase control) reuse the M2 data."""
    if method in ("m1", "m5"):
        pairs = [(U0 + DIRECT, "{v}"), (STATEMENT_PROMPT, STATEMENT)]
    elif method in ("m2", "m3", "m6i", "m6d", "e4u", "e4p"):
        pairs = VARIED
    else:
        raise ValueError(method)
    return [(fill(u, fact), fill(a, fact)) for u, a in pairs]


def retention_queries(subjects, values, facts):
    """Natural facts about subjects and values; recall of the other written facts is separate."""
    cities = city_table()
    rows = []
    for h in subjects:
        country = cities[h]["country"]
        rows.append(
            {
                "qid": f"ret/capital_rev/{h}",
                "group": "subject",
                "prompt": CAPITAL_REV[0].format(h=h) + DIRECT,
                "gold": country,
                "type": "country",
            }
        )
        currency = currency_of(country)
        if currency:
            rows.append(
                {
                    "qid": f"ret/currency/{country}",
                    "group": "subject",
                    "prompt": CURRENCY_Q.format(C=country) + DIRECT,
                    "gold": currency,
                    "type": "currency",
                }
            )
    for v in values:
        rows.append(
            {
                "qid": f"ret/city_country/{v}",
                "group": "value",
                "prompt": CITY_COUNTRY.format(v=v) + DIRECT,
                "gold": cities[v]["country"],
                "type": "country",
            }
        )
    return rows


def natural_two_hop(subjects):
    """Currency via the country whose capital is h: bridge computed from an explicit subject."""
    cities = city_table()
    rows = []
    for h in subjects:
        country = cities[h]["country"]
        currency = currency_of(country)
        if not currency:
            continue
        for i, t in enumerate(CURRENCY_TWO_HOP):
            rows.append(
                {
                    "qid": f"nat2/{i}/{h}",
                    "prompt": t.format(h=h) + DIRECT,
                    "gold": currency,
                    "type": "currency",
                    "subject": h,
                }
            )
        rows.append(
            {
                "qid": f"nat2short/{h}",
                "prompt": CURRENCY_SHORTCUT.format(h=h) + DIRECT,
                "gold": currency,
                "type": "currency",
                "subject": h,
            }
        )
    return rows


def replay_queries(subjects, mode):
    """Natural questions for replay; none is an evaluated item (evaluation uses CAPITAL_REV[0],
    CURRENCY_Q for subject countries, CITY_COUNTRY and CURRENCY_TWO_HOP for subjects)."""
    cities = city_table()
    subject_countries = {cities[h]["country"] for h in subjects}
    rows = []
    if mode == "direct":
        for country, _, capital, _, currency, _ in COUNTRIES:
            if capital:
                rows.append(CAPITAL_Q.format(C=country) + DIRECT)
                rows.append(CAPITAL_REV[1].format(h=capital) + DIRECT)
            if currency and country not in subject_countries:
                rows.append(CURRENCY_Q.format(C=country) + DIRECT)
    else:
        for _country, _, capital, _, currency, _ in COUNTRIES:
            if capital and currency and capital not in subjects:
                rows.append(CURRENCY_TWO_HOP[0].format(h=capital) + COT)
    return rows
