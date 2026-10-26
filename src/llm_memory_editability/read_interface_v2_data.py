"""Data for knowledge-read-interface-v2: distractor contexts (R2), natural two-hop references
(R3), the positive control of the subject-state write (E1-N) and the usage-practice operations
(E4). Entities, relations, prompts and grading otherwise come from `read_interface_data`.
"""

from __future__ import annotations

import random
from dataclasses import asdict

from llm_memory_editability import read_interface_data as D

# Continent of each country with a capital in D.COUNTRIES; transcontinental countries accept both.
CONTINENT = {
    "Europe": [
        "France",
        "Germany",
        "Italy",
        "Spain",
        "Portugal",
        "United Kingdom",
        "Ireland",
        "Netherlands",
        "Belgium",
        "Switzerland",
        "Austria",
        "Sweden",
        "Norway",
        "Denmark",
        "Finland",
        "Iceland",
        "Poland",
        "Czech Republic",
        "Slovakia",
        "Hungary",
        "Romania",
        "Bulgaria",
        "Greece",
        "Croatia",
        "Serbia",
        "Slovenia",
        "Albania",
        "Estonia",
        "Latvia",
        "Lithuania",
        "Ukraine",
        "Belarus",
        "Malta",
    ],
    "Asia": [
        "Japan",
        "China",
        "South Korea",
        "India",
        "Pakistan",
        "Bangladesh",
        "Nepal",
        "Thailand",
        "Vietnam",
        "Indonesia",
        "Malaysia",
        "Philippines",
        "Cambodia",
        "Laos",
        "Mongolia",
        "Uzbekistan",
        "Afghanistan",
        "Iran",
        "Iraq",
        "Saudi Arabia",
        "United Arab Emirates",
        "Qatar",
        "Oman",
        "Jordan",
        "Syria",
        "Lebanon",
        "Bhutan",
    ],
    "Africa": [
        "Egypt",
        "Nigeria",
        "Kenya",
        "Ethiopia",
        "Morocco",
        "Algeria",
        "Libya",
        "Ghana",
        "Senegal",
        "Uganda",
        "Rwanda",
        "Sudan",
        "Angola",
        "Zimbabwe",
        "Zambia",
        "Mozambique",
        "Madagascar",
        "Cameroon",
        "Mali",
        "Somalia",
        "Democratic Republic of the Congo",
        "Namibia",
        "Botswana",
        "South Africa",
    ],
    "North America": [
        "United States",
        "Canada",
        "Mexico",
        "Cuba",
        "Jamaica",
        "Haiti",
        "Dominican Republic",
        "Honduras",
        "Nicaragua",
        "Costa Rica",
    ],
    "South America": [
        "Brazil",
        "Argentina",
        "Chile",
        "Peru",
        "Colombia",
        "Venezuela",
        "Paraguay",
        "Uruguay",
    ],
    "Oceania": ["Australia", "New Zealand", "Fiji", "Papua New Guinea"],
}
TRANSCONTINENTAL = {
    "Russia": ["Europe", "Asia"],
    "Turkey": ["Asia", "Europe"],
    "Kazakhstan": ["Asia", "Europe"],
    "Georgia": ["Asia", "Europe"],
    "Armenia": ["Asia", "Europe"],
    "Azerbaijan": ["Asia", "Europe"],
    "Cyprus": ["Europe", "Asia"],
}
CONTINENT_ALIASES = {
    "Europe": [],
    "Asia": [],
    "Africa": [],
    "North America": ["Central America", "Caribbean"],
    "South America": [],
    "Oceania": ["Australia"],
}

# Official languages; the first entry is the training target, any entry grades as correct.
LANGUAGES = {
    "France": ["French"],
    "Germany": ["German"],
    "Italy": ["Italian"],
    "Spain": ["Spanish"],
    "Portugal": ["Portuguese"],
    "United Kingdom": ["English"],
    "Ireland": ["Irish", "English"],
    "Netherlands": ["Dutch"],
    "Belgium": ["Dutch", "French", "German"],
    "Switzerland": ["German", "French", "Italian", "Romansh"],
    "Austria": ["German"],
    "Sweden": ["Swedish"],
    "Norway": ["Norwegian"],
    "Denmark": ["Danish"],
    "Finland": ["Finnish", "Swedish"],
    "Iceland": ["Icelandic"],
    "Poland": ["Polish"],
    "Czech Republic": ["Czech"],
    "Slovakia": ["Slovak"],
    "Hungary": ["Hungarian"],
    "Romania": ["Romanian"],
    "Bulgaria": ["Bulgarian"],
    "Greece": ["Greek"],
    "Croatia": ["Croatian"],
    "Serbia": ["Serbian"],
    "Slovenia": ["Slovene"],
    "Albania": ["Albanian"],
    "Estonia": ["Estonian"],
    "Latvia": ["Latvian"],
    "Lithuania": ["Lithuanian"],
    "Ukraine": ["Ukrainian"],
    "Belarus": ["Belarusian", "Russian"],
    "Russia": ["Russian"],
    "Malta": ["Maltese", "English"],
    "Cyprus": ["Greek", "Turkish"],
    "Japan": ["Japanese"],
    "China": ["Chinese"],
    "South Korea": ["Korean"],
    "India": ["Hindi", "English"],
    "Pakistan": ["Urdu", "English"],
    "Bangladesh": ["Bengali"],
    "Nepal": ["Nepali"],
    "Thailand": ["Thai"],
    "Vietnam": ["Vietnamese"],
    "Indonesia": ["Indonesian"],
    "Malaysia": ["Malay"],
    "Philippines": ["Filipino", "English"],
    "Cambodia": ["Khmer"],
    "Laos": ["Lao"],
    "Mongolia": ["Mongolian"],
    "Kazakhstan": ["Kazakh", "Russian"],
    "Uzbekistan": ["Uzbek"],
    "Afghanistan": ["Pashto", "Dari"],
    "Iran": ["Persian"],
    "Iraq": ["Arabic", "Kurdish"],
    "Saudi Arabia": ["Arabic"],
    "United Arab Emirates": ["Arabic"],
    "Qatar": ["Arabic"],
    "Oman": ["Arabic"],
    "Jordan": ["Arabic"],
    "Syria": ["Arabic"],
    "Lebanon": ["Arabic"],
    "Turkey": ["Turkish"],
    "Georgia": ["Georgian"],
    "Armenia": ["Armenian"],
    "Azerbaijan": ["Azerbaijani"],
    "Bhutan": ["Dzongkha"],
    "Egypt": ["Arabic"],
    "Nigeria": ["English"],
    "Kenya": ["Swahili", "English"],
    "Ethiopia": ["Amharic"],
    "Morocco": ["Arabic", "Berber"],
    "Algeria": ["Arabic", "Berber"],
    "Libya": ["Arabic"],
    "Ghana": ["English"],
    "Senegal": ["French"],
    "Uganda": ["English", "Swahili"],
    "Rwanda": ["Kinyarwanda", "French", "English", "Swahili"],
    "Sudan": ["Arabic", "English"],
    "Angola": ["Portuguese"],
    "Zimbabwe": ["English", "Shona", "Ndebele"],
    "Zambia": ["English"],
    "Mozambique": ["Portuguese"],
    "Madagascar": ["Malagasy", "French"],
    "Cameroon": ["French", "English"],
    "Mali": ["French", "Bambara"],
    "Somalia": ["Somali", "Arabic"],
    "Democratic Republic of the Congo": ["French"],
    "Namibia": ["English"],
    "Botswana": ["English", "Tswana"],
    "South Africa": ["English"],
    "United States": ["English"],
    "Canada": ["English", "French"],
    "Mexico": ["Spanish"],
    "Brazil": ["Portuguese"],
    "Argentina": ["Spanish"],
    "Chile": ["Spanish"],
    "Peru": ["Spanish", "Quechua", "Aymara"],
    "Colombia": ["Spanish"],
    "Venezuela": ["Spanish"],
    "Paraguay": ["Spanish", "Guarani"],
    "Uruguay": ["Spanish"],
    "Cuba": ["Spanish"],
    "Jamaica": ["English"],
    "Haiti": ["Haitian Creole", "French"],
    "Dominican Republic": ["Spanish"],
    "Honduras": ["Spanish"],
    "Nicaragua": ["Spanish"],
    "Costa Rica": ["Spanish"],
    "Australia": ["English"],
    "New Zealand": ["English", "Maori"],
    "Fiji": ["English", "Fijian", "Hindi"],
    "Papua New Guinea": ["English", "Tok Pisin"],
}
LANGUAGE_ALIASES = {
    "Spanish": ["Castilian"],
    "Dutch": ["Flemish"],
    "Irish": ["Gaelic", "Irish Gaelic"],
    "Slovene": ["Slovenian"],
    "Chinese": ["Mandarin", "Mandarin Chinese", "Standard Chinese"],
    "Bengali": ["Bangla"],
    "Nepali": ["Nepalese"],
    "Indonesian": ["Bahasa Indonesia"],
    "Malay": ["Bahasa Malaysia", "Bahasa Melayu", "Malaysian"],
    "Filipino": ["Tagalog"],
    "Lao": ["Laotian"],
    "Persian": ["Farsi"],
    "Azerbaijani": ["Azeri"],
    "Swahili": ["Kiswahili"],
    "Berber": ["Amazigh", "Tamazight"],
    "Tswana": ["Setswana"],
    "Guarani": ["Guaraní"],
    "Haitian Creole": ["Creole", "Kreyòl"],
    "Maori": ["Māori"],
    "Kinyarwanda": ["Rwandan"],
    "Dzongkha": [],
    "Hindi": [],
    "Urdu": [],
}

# Highest mountain of a country (U2-shape natural two-hop: h -> country -> mountain).
MOUNTAIN = {
    "France": ("Mont Blanc", ["Monte Bianco"]),
    "Italy": ("Mont Blanc", ["Monte Bianco"]),
    "Germany": ("Zugspitze", []),
    "Spain": ("Teide", ["Mount Teide", "Pico del Teide"]),
    "Portugal": ("Mount Pico", ["Ponta do Pico", "Pico Mountain"]),
    "United Kingdom": ("Ben Nevis", []),
    "Ireland": ("Carrauntoohil", ["Carrantuohill", "Carrauntuohill"]),
    "Belgium": ("Signal de Botrange", ["Botrange"]),
    "Switzerland": ("Dufourspitze", ["Monte Rosa", "Dufour Peak"]),
    "Austria": ("Grossglockner", ["Großglockner"]),
    "Sweden": ("Kebnekaise", []),
    "Norway": ("Galdhøpiggen", ["Galdhopiggen"]),
    "Finland": ("Halti", ["Haltitunturi"]),
    "Iceland": ("Hvannadalshnúkur", ["Hvannadalshnukur"]),
    "Poland": ("Rysy", []),
    "Czech Republic": ("Sněžka", ["Snezka"]),
    "Slovakia": ("Gerlachovský štít", ["Gerlach Peak", "Gerlachovsky stit", "Gerlach"]),
    "Hungary": ("Kékes", ["Kekes"]),
    "Romania": ("Moldoveanu", ["Moldoveanu Peak"]),
    "Bulgaria": ("Musala", []),
    "Greece": ("Mount Olympus", ["Olympus"]),
    "Cyprus": ("Mount Olympus", ["Olympus", "Chionistra"]),
    "Slovenia": ("Triglav", []),
    "Albania": ("Korab", ["Mount Korab"]),
    "Ukraine": ("Hoverla", ["Goverla"]),
    "Russia": ("Mount Elbrus", ["Elbrus"]),
    "Japan": ("Mount Fuji", ["Fuji", "Fujisan", "Fuji-san"]),
    "China": ("Mount Everest", ["Everest", "Qomolangma", "Chomolungma", "Sagarmatha"]),
    "Nepal": ("Mount Everest", ["Everest", "Qomolangma", "Chomolungma", "Sagarmatha"]),
    "South Korea": ("Hallasan", ["Mount Halla", "Halla"]),
    "India": ("Kangchenjunga", ["Kanchenjunga", "Khangchendzonga"]),
    "Pakistan": ("K2", ["Mount Godwin-Austen"]),
    "Thailand": ("Doi Inthanon", ["Inthanon"]),
    "Vietnam": ("Fansipan", ["Phan Xi Păng", "Fan Si Pan"]),
    "Indonesia": ("Puncak Jaya", ["Carstensz Pyramid"]),
    "Malaysia": ("Mount Kinabalu", ["Kinabalu"]),
    "Philippines": ("Mount Apo", []),
    "Mongolia": ("Khüiten Peak", ["Khuiten", "Khüiten"]),
    "Kazakhstan": ("Khan Tengri", []),
    "Afghanistan": ("Noshaq", ["Nowshak"]),
    "Iran": ("Mount Damavand", ["Damavand", "Demavend"]),
    "Syria": ("Mount Hermon", ["Hermon"]),
    "Lebanon": ("Qurnat as Sawda'", ["Qurnat as Sawda", "Qornet es Saouda"]),
    "Turkey": ("Mount Ararat", ["Ararat", "Ağrı Dağı", "Agri Dagi"]),
    "Georgia": ("Shkhara", []),
    "Armenia": ("Mount Aragats", ["Aragats"]),
    "Azerbaijan": ("Bazardüzü", ["Bazarduzu"]),
    "Bhutan": ("Gangkhar Puensum", ["Gangkar Punsum"]),
    "Egypt": ("Mount Catherine", ["Jabal Katrina", "Saint Catherine"]),
    "Kenya": ("Mount Kenya", []),
    "Ethiopia": ("Ras Dashen", ["Ras Dejen", "Ras Dashan"]),
    "Morocco": ("Toubkal", ["Jbel Toubkal", "Mount Toubkal"]),
    "Algeria": ("Mount Tahat", ["Tahat"]),
    "Uganda": ("Margherita Peak", ["Mount Stanley", "Margherita"]),
    "Democratic Republic of the Congo": ("Margherita Peak", ["Mount Stanley", "Margherita"]),
    "Rwanda": ("Mount Karisimbi", ["Karisimbi"]),
    "Cameroon": ("Mount Cameroon", []),
    "Madagascar": ("Maromokotro", []),
    "Namibia": ("Brandberg", ["Königstein", "Konigstein"]),
    "United States": ("Denali", ["Mount McKinley", "McKinley"]),
    "Canada": ("Mount Logan", []),
    "Brazil": ("Pico da Neblina", ["Neblina"]),
    "Argentina": ("Aconcagua", []),
    "Chile": ("Ojos del Salado", []),
    "Peru": ("Huascarán", ["Huascaran"]),
    "Venezuela": ("Pico Bolívar", ["Pico Bolivar"]),
    "Cuba": ("Pico Turquino", ["Turquino"]),
    "Jamaica": ("Blue Mountain Peak", []),
    "Dominican Republic": ("Pico Duarte", []),
    "Costa Rica": ("Cerro Chirripó", ["Chirripó", "Chirripo"]),
    "Australia": ("Mount Kosciuszko", ["Kosciuszko", "Kosciusko"]),
    "New Zealand": ("Aoraki / Mount Cook", ["Aoraki", "Mount Cook"]),
    "Papua New Guinea": ("Mount Wilhelm", []),
}

# Largest city where it is not the capital (U2 shape: h -> country -> largest city).
LARGEST_CITY = {
    "Australia": ("Sydney", ["Melbourne"]),
    "Canada": ("Toronto", []),
    "United States": ("New York City", ["New York", "NYC"]),
    "Brazil": ("São Paulo", ["Sao Paulo"]),
    "Turkey": ("Istanbul", []),
    "Nigeria": ("Lagos", []),
    "Switzerland": ("Zurich", ["Zürich"]),
    "Pakistan": ("Karachi", []),
    "Morocco": ("Casablanca", []),
    "New Zealand": ("Auckland", []),
    "Kazakhstan": ("Almaty", ["Alma-Ata"]),
    "Vietnam": ("Ho Chi Minh City", ["Saigon"]),
    "Cameroon": ("Douala", []),
    "United Arab Emirates": ("Dubai", []),
    "Philippines": ("Quezon City", []),
}

# River through each capital (U3 shape: "the capital of C" -> h -> river).
RIVER = {
    "Paris": [("Seine", [])],
    "Berlin": [("Spree", []), ("Havel", [])],
    "Rome": [("Tiber", ["Tevere"])],
    "Madrid": [("Manzanares", [])],
    "Lisbon": [("Tagus", ["Tejo"])],
    "London": [("Thames", [])],
    "Dublin": [("Liffey", [])],
    "Amsterdam": [("Amstel", []), ("IJ", [])],
    "Brussels": [("Senne", ["Zenne"])],
    "Bern": [("Aare", ["Aar"])],
    "Vienna": [("Danube", ["Donau"])],
    "Warsaw": [("Vistula", ["Wisła", "Wisla"])],
    "Prague": [("Vltava", ["Moldau"])],
    "Bratislava": [("Danube", ["Donau"])],
    "Budapest": [("Danube", ["Duna"])],
    "Bucharest": [("Dâmbovița", ["Dambovita", "Dâmbovita"])],
    "Zagreb": [("Sava", [])],
    "Belgrade": [("Danube", []), ("Sava", [])],
    "Ljubljana": [("Ljubljanica", [])],
    "Riga": [("Daugava", ["Western Dvina"])],
    "Vilnius": [("Neris", []), ("Vilnia", [])],
    "Kyiv": [("Dnieper", ["Dnipro", "Dnepr"])],
    "Minsk": [("Svislach", ["Svisloch"])],
    "Moscow": [("Moskva", ["Moscow River", "Moskva River"])],
    "Tokyo": [("Sumida", ["Sumida River"]), ("Arakawa", []), ("Tama", ["Tama River"])],
    "Seoul": [("Han", ["Han River", "Hangang"])],
    "New Delhi": [("Yamuna", ["Jamuna"])],
    "Dhaka": [("Buriganga", [])],
    "Kathmandu": [("Bagmati", [])],
    "Bangkok": [("Chao Phraya", [])],
    "Hanoi": [("Red River", ["Hong River", "Sông Hồng"])],
    "Jakarta": [("Ciliwung", [])],
    "Kuala Lumpur": [("Klang", ["Klang River"]), ("Gombak", [])],
    "Manila": [("Pasig", ["Pasig River"])],
    "Phnom Penh": [("Mekong", []), ("Tonlé Sap", ["Tonle Sap"])],
    "Vientiane": [("Mekong", [])],
    "Ulaanbaatar": [("Tuul", ["Tuul River", "Tola"])],
    "Astana": [("Ishim", ["Esil", "Yesil"])],
    "Tashkent": [("Chirchiq", ["Chirchik"])],
    "Kabul": [("Kabul River", [])],
    "Baghdad": [("Tigris", [])],
    "Damascus": [("Barada", [])],
    "Beirut": [("Beirut River", [])],
    "Tbilisi": [("Kura", ["Mtkvari"])],
    "Yerevan": [("Hrazdan", ["Razdan"])],
    "Thimphu": [("Wang Chhu", ["Thimphu Chhu", "Raidak"])],
    "Cairo": [("Nile", [])],
    "Nairobi": [("Nairobi River", [])],
    "Rabat": [("Bou Regreg", ["Bou Regreg River"])],
    "Khartoum": [("Nile", ["Blue Nile", "White Nile"])],
    "Antananarivo": [("Ikopa", [])],
    "Bamako": [("Niger", ["Niger River"])],
    "Kinshasa": [("Congo", ["Congo River"])],
    "Washington, D.C.": [("Potomac", []), ("Anacostia", [])],
    "Ottawa": [("Ottawa River", []), ("Rideau", ["Rideau River"])],
    "Santiago": [("Mapocho", [])],
    "Lima": [("Rímac", ["Rimac"])],
    "Bogotá": [("Bogotá River", ["Bogota River"])],
    "Caracas": [("Guaire", [])],
    "Asunción": [("Paraguay River", [])],
    "Havana": [("Almendares", [])],
    "Santo Domingo": [("Ozama", [])],
    "Tegucigalpa": [("Choluteca", [])],
    "Canberra": [("Molonglo", [])],
}


def _grader(table):
    return D.Grader({name: sorted(set(aliases)) for name, aliases in table.items()})


def continent_of(country):
    if country in TRANSCONTINENTAL:
        return TRANSCONTINENTAL[country]
    return [c for c, members in CONTINENT.items() if country in members]


def graders():
    """Graders for the v2 attribute types, plus the original city/country/currency graders."""
    out = dict(D.graders())
    out["continent"] = _grader(CONTINENT_ALIASES)
    langs = {lang: LANGUAGE_ALIASES.get(lang, []) for ls in LANGUAGES.values() for lang in ls}
    out["language"] = _grader(langs)
    mountains = {}
    for name, aliases in MOUNTAIN.values():
        mountains.setdefault(name, set()).update(aliases)
    out["mountain"] = _grader(mountains)
    rivers = {}
    for items in RIVER.values():
        for name, aliases in items:
            rivers.setdefault(name, set()).update(aliases)
    out["river"] = _grader(rivers)
    big = {c: row["aliases"] for c, row in D.city_table().items()}
    for name, aliases in LARGEST_CITY.values():
        big.setdefault(name, [])
        big[name] = sorted(set(big[name]) | {a for a in aliases if a not in D.city_table()})
    out["bigcity"] = _grader(big)
    return out


def grade_any(grader, text, golds, exclude=()):
    """Strict: the first mention is any gold; lenient: some gold is mentioned."""
    mentions = [m for m in grader.mentions(text) if m not in exclude]
    return {
        "strict": bool(mentions) and mentions[0] in golds,
        "lenient": any(g in mentions for g in golds),
        "first": mentions[0] if mentions else None,
    }


# ---------------------------------------------------------------- R2 distractor context


def distractor_systems(facts, seed=20261021, n_distractors=3):
    """System prompt per main fact: the target statement plus three other main facts with the
    same relation and different subject and value, in a seeded random order."""
    rng = random.Random(seed)
    main = sorted([f for f in facts if f.split != "calibration"], key=lambda f: f.fid)
    out = {}
    for f in main:
        pool = [g for g in main if g.relation == f.relation and g.h != f.h and g.v != f.v]
        lines = [f, *rng.sample(pool, n_distractors)]
        rng.shuffle(lines)
        out[f.fid] = {
            "system": "\n".join("Fact: " + D.context_statement(g) for g in lines),
            "distractors": [g.fid for g in lines if g is not f],
        }
    return out


# ---------------------------------------------------------------- R3 natural references

COT_COUNTRY = D.COT.replace("intermediate city", "intermediate country")

U2_SHAPE = {
    "currency": {
        "two_hop": D.CURRENCY_TWO_HOP,
        "hop2": "What is the currency of {C}?",
        "no_relation": "What currency is used in {h}?",
        "no_entity": "What currency is used in the country whose capital is that city?",
        "type": "currency",
    },
    "language": {
        "two_hop": [
            "What is the official language of the country whose capital is {h}?",
            "Which language is official in the country with capital {h}?",
        ],
        "hop2": "What is the official language of {C}?",
        "no_relation": "What is the official language of {h}?",
        "no_entity": "What is the official language of the country whose capital is that city?",
        "type": "language",
    },
    "mountain": {
        "two_hop": [
            "What is the highest mountain in the country whose capital is {h}?",
            "Which mountain is the tallest in the country with capital {h}?",
        ],
        "hop2": "What is the highest mountain in {C}?",
        "no_relation": "What is the highest mountain in {h}?",
        "no_entity": "What is the highest mountain in the country whose capital is that city?",
        "type": "mountain",
    },
    "bigcity": {
        "two_hop": [
            "What is the largest city of the country whose capital is {h}?",
            "Which city is the biggest in the country with capital {h}?",
        ],
        "hop2": "What is the largest city of {C}?",
        "no_relation": "What is the largest city of {h}?",
        "no_entity": "What is the largest city of the country whose capital is that city?",
        "type": "bigcity",
    },
}
U3_SHAPE = {
    "river": {
        "two_hop": [
            "Which river flows through the capital of {C}?",
            "What river runs through the capital city of {C}?",
        ],
        "hop1": D.CAPITAL_Q,
        "hop2": "Which river flows through {h}?",
        "no_relation": "Which river flows through {C}?",
        "no_entity": "Which river flows through the capital of that country?",
        "type": "river",
    }
}


def u2_gold(attribute, country):
    if attribute == "currency":
        cur = D.currency_of(country)
        return [cur] if cur else []
    if attribute == "language":
        return LANGUAGES.get(country, [])
    if attribute == "mountain":
        return [MOUNTAIN[country][0]] if country in MOUNTAIN else []
    if attribute == "bigcity":
        return (
            [
                LARGEST_CITY[country][0],
                *[a for a in LARGEST_CITY[country][1] if a in D.city_table()],
            ]
            if country in LARGEST_CITY
            else []
        )
    raise ValueError(attribute)


def natural_reference_rows(known):
    """All R3 prompts for capitals the model knows (both capital directions and the capital's
    country). Items are (shape, attribute, capital); filtering happens after answering."""
    cities = D.city_table()
    capitals = [
        c
        for c, row in cities.items()
        if row["capital"]
        and known.get(f"capital/{row['country']}")
        and all(known.get(f"capital_rev{i}/{c}") for i in range(len(D.CAPITAL_REV)))
    ]
    rows = []
    for h in capitals:
        country = cities[h]["country"]
        for attr, spec in U2_SHAPE.items():
            golds = u2_gold(attr, country)
            if not golds:
                continue
            item = f"U2shape/{attr}/{h}"
            base = {"item": item, "shape": "U2", "attribute": attr, "h": h, "C": country}
            base.update({"golds": golds, "type": spec["type"]})
            for i, t in enumerate(spec["two_hop"]):
                for mode, suffix in (("direct", D.DIRECT), ("cot", COT_COUNTRY)):
                    rows.append(
                        {
                            **base,
                            "role": f"two_hop.{i}",
                            "mode": mode,
                            "prompt": t.format(h=h) + suffix,
                            "exclude": [h],
                        }
                    )
            rows.append(
                {
                    **base,
                    "role": "hop2",
                    "mode": "direct",
                    "prompt": spec["hop2"].format(C=country) + D.DIRECT,
                    "exclude": [],
                }
            )
            rows.append(
                {
                    **base,
                    "role": "no_relation",
                    "mode": "direct",
                    "prompt": spec["no_relation"].format(h=h) + D.DIRECT,
                    "exclude": [h],
                }
            )
            rows.append(
                {
                    **base,
                    "role": "no_entity",
                    "mode": "direct",
                    "prompt": spec["no_entity"] + D.DIRECT,
                    "exclude": [],
                }
            )
        for attr, spec in U3_SHAPE.items():
            if h not in RIVER:
                continue
            golds = [name for name, _ in RIVER[h]]
            item = f"U3shape/{attr}/{h}"
            base = {"item": item, "shape": "U3", "attribute": attr, "h": h, "C": country}
            base.update({"golds": golds, "type": spec["type"]})
            for i, t in enumerate(spec["two_hop"]):
                for mode, suffix in (("direct", D.DIRECT), ("cot", D.COT)):
                    rows.append(
                        {
                            **base,
                            "role": f"two_hop.{i}",
                            "mode": mode,
                            "prompt": t.format(C=country) + suffix,
                            "exclude": [country],
                        }
                    )
            rows.append(
                {
                    **base,
                    "role": "hop2",
                    "mode": "direct",
                    "prompt": spec["hop2"].format(h=h) + D.DIRECT,
                    "exclude": [h],
                }
            )
            rows.append(
                {
                    **base,
                    "role": "no_relation",
                    "mode": "direct",
                    "prompt": spec["no_relation"].format(C=country) + D.DIRECT,
                    "exclude": [country],
                }
            )
            rows.append(
                {
                    **base,
                    "role": "no_entity",
                    "mode": "direct",
                    "prompt": spec["no_entity"] + D.DIRECT,
                    "exclude": [],
                }
            )
    return rows


# ---------------------------------------------------------------- E1-N positive control

E1N_TRAIN = [
    ("Which country is {h} in?" + D.DIRECT, "{C}"),
    ("What country is {h} located in?", "{h} is located in {C}."),
    ("{h} is a city in which country?", "{C}"),
    ("Tell me the country of {h}.", "The country of {h} is {C}."),
    ("Where is {h}? Give the country.", "{C}"),
    ("In what country can you find {h}?", "You can find {h} in {C}."),
    ("Identify the country where {h} lies.", "{C}"),
    ("State the country of the city {h}.", "{h} is in {C}."),
    ("Quick question: {h} is in which country?", "{C}."),
    ("Which nation does {h} belong to?", "{h} belongs to {C}."),
    ("What is the country of {h}?", "{C}"),
    ("If you travel to {h}, which country are you in?", "You are in {C}."),
]
E1N_HELD_OUT = [
    *D.CAPITAL_REV,
    D.CITY_COUNTRY.replace("{v}", "{h}"),
    "Name the country where {h} is located.",
]


def e1n_pairs(subjects, seed=20261022):
    """Each subject h gets a counterfactual country C' (different country and currency)."""
    cities = D.city_table()
    rng = random.Random(seed)
    countries = sorted({cities[h]["country"] for h in subjects})
    out = {}
    for h in sorted(subjects):
        own = cities[h]["country"]
        cur = D.currency_of(own)
        if not cur:
            continue
        options = [
            c
            for c in countries
            if c != own
            and D.currency_of(c)
            and not (D.currency_aliases(cur) & D.currency_aliases(D.currency_of(c)))
        ]
        out[h] = rng.choice(options)
    return out


# ---------------------------------------------------------------- E4 operations

PRACTICE = {
    "continent": {
        "direct": [
            "Which continent is {h}'s {R} on?",
            "Name the continent where the {R} of {h} lies.",
        ],
        "cot": [
            "On what continent would you find the {R} of {h}?",
            "The {R} of {h} belongs to which continent?",
        ],
    },
    "language": {
        "direct": [
            "What official language is used in the country of {h}'s {R}?",
            "Which language is official where the {R} of {h} is?",
        ],
        "cot": [
            "Name the official language of the country that contains the {R} of {h}.",
            "People in the country of {h}'s {R} officially speak which language?",
        ],
    },
}
EVAL_OPS = {
    "continent": [
        "In which continent is the {R} of {h} located?",
        "The {R} of {h} lies on which continent?",
    ],
    "language": [
        "What is the official language of the country where the {R} of {h} is located?",
        "The {R} of {h} is in a country whose official language is what?",
    ],
    "currency": [
        "What currency is used in the country where the {R} of {h} is located?",
        "The {R} of {h} is in a country that uses which currency?",
    ],
}
OP_TYPE = {"continent": "continent", "language": "language", "currency": "currency"}
PRACTICED = ("continent", "language")
HELD_OUT = ("country", "currency")
PARAPHRASES = [
    ("Which city has been designated as {h}'s {R}?" + D.DIRECT, "{v}"),
    ("Please name the {R} of {h}." + D.DIRECT, "{v}"),
    ("Which city is paired with {h} as its {R}?" + D.DIRECT, "{v}"),
    ("Give the {R} of {h}." + D.DIRECT, "{v}"),
    ("Remind me: what is {h}'s {R}?", "{h}'s {R} is {v}."),
    ("Can you tell me the {R} of {h}?", "Sure. The {R} of {h} is {v}."),
    ("What is listed as the {R} for {h}?", "{v} is listed as the {R} for {h}."),
    ("What does {h} have as its {R}?", "{h} has {v} as its {R}."),
]
SCREEN = {
    "continent": "On which continent is the city of {v} located?",
    "language": "What is the official language of the country where {v} is located?",
    "currency": "What currency is used in {v}?",
}


def op_golds(op, country):
    if op == "continent":
        return continent_of(country)
    if op == "language":
        return LANGUAGES.get(country, [])
    if op == "currency":
        cur = D.currency_of(country)
        return [cur] if cur else []
    if op == "country":
        return [country]
    raise ValueError(op)


def practice_examples(fact):
    """Four usage-practice examples per practiced operation: two direct, two step-by-step. The
    step-by-step answer writes v first and never names v's country (a held-out operation)."""
    out = []
    for op in PRACTICED:
        gold = op_golds(op, fact.country_v)[0]
        for t in PRACTICE[op]["direct"]:
            out.append((D.fill(t, fact) + D.DIRECT, gold))
        for t in PRACTICE[op]["cot"]:
            what = "continent" if op == "continent" else "official language there"
            reply = (
                f"The {D.RELATIONS[fact.relation]} of {fact.h} is {fact.v}. "
                f"The {what} is {gold}.\n{D.FINAL} {gold}"
            )
            out.append((D.fill(t, fact) + D.COT, reply))
    return out


def paraphrase_examples(fact):
    return [(D.fill(u, fact), D.fill(a, fact)) for u, a in PARAPHRASES]


def op_queries(fact):
    """E4 evaluation prompts for the practiced and held-out operations, direct and step-by-step.
    The country operation is the original U2 and is evaluated through `D.use_queries`."""
    rows = []
    for op, templates in EVAL_OPS.items():
        golds = op_golds(op, fact.country_v)
        if not golds:
            continue
        for mode, suffix in (("direct", D.DIRECT), ("cot", D.COT)):
            for i, t in enumerate(templates):
                rows.append(
                    {
                        "fid": fact.fid,
                        "use": f"op-{op}.{i}",
                        "op": op,
                        "prompt": D.fill(t, fact) + suffix,
                        "golds": golds,
                        "type": OP_TYPE[op],
                        "mode": mode,
                        "subject": fact.h,
                        "exclude": [],
                    }
                )
    return rows


def screen_rows(values):
    cities = D.city_table()
    rows = []
    for v in values:
        country = cities[v]["country"]
        for op, t in SCREEN.items():
            golds = op_golds(op, country)
            if golds:
                rows.append(
                    {
                        "v": v,
                        "op": op,
                        "prompt": t.format(v=v) + D.DIRECT,
                        "golds": golds,
                        "type": OP_TYPE[op],
                        "mode": "direct",
                        "exclude": [v],
                    }
                )
    return rows


# ---------------------------------------------------------------- E2 existing relation

E2_RELATIONS = {"twin": "twin city"}
E2_PRIOR = [
    "What is the twin city of {h}?",
    "Which city is twinned with {h}?",
    "Name the sister city of {h}.",
]


def register_e2_relations():
    """Make the existing relation available to `D.fill` in the E2 process only (D.RELATIONS
    also drives the fictional-relation cycling of `D.build_facts`, which E2 does not call)."""
    D.RELATIONS.update(E2_RELATIONS)


def e2_candidates(known, seed=20261024, max_subjects=60):
    """Known capitals (same screening rule as the development batch), in seeded order."""
    cities = D.city_table()
    capitals = [
        c
        for c, row in cities.items()
        if row["capital"]
        and known.get(f"capital/{row['country']}")
        and all(known.get(f"capital_rev{i}/{c}") for i in range(len(D.CAPITAL_REV)))
        and known.get(f"city_country/{c}")
    ]
    rng = random.Random(seed)
    rng.shuffle(capitals)
    return capitals[: int(max_subjects / 0.55)]


def build_e2_facts(known, priors, seed=20261024, n_calibration_subjects=5, max_subjects=60):
    """Each subject gets one twin-city fact and one fictional-relation fact (cycling the four
    coined relations). Subjects whose released model names the same city for all three prior
    prompts are dropped; v never equals a city the model named for h's twin city.

    `priors` maps h to the first city named in each prior answer (None when none)."""
    cities = D.city_table()
    order = e2_candidates(known, seed)
    stable = {h for h in order if len(set(priors.get(h, []))) == 1 and priors[h][0] is not None}
    pool = [h for h in order if h not in stable]
    n_subjects = min(max_subjects, int(len(pool) * 0.55))
    subjects, rest = pool[:n_subjects], [h for h in order if h not in pool[:n_subjects]]
    values = rest + [
        c for c, row in cities.items() if not row["capital"] and known.get(f"city_country/{c}")
    ]
    fictional = sorted(D.RELATIONS)
    fictional = [r for r in fictional if r not in E2_RELATIONS]
    rng = random.Random(seed + 1)
    use = {v: 0 for v in values}
    facts = []
    for index, h in enumerate(subjects):
        split = "calibration" if index < n_calibration_subjects else ""
        for relation in ("twin", fictional[index % len(fictional)]):
            country_h = cities[h]["country"]
            taken = {f.v for f in facts if f.h == h} | set(priors.get(h, []))
            options = [
                v for v in values
                if cities[v]["country"] != country_h and use[v] < 2 and v not in taken
            ]  # fmt: skip
            least = min(use[v] for v in options)
            v = rng.choice([v for v in options if use[v] == least])
            use[v] += 1
            facts.append(
                D.Fact(f"{relation}/{h}", h, country_h, relation, v, cities[v]["country"], split)
            )
    main = [f for f in facts if f.split != "calibration"]
    idx = list(range(len(main)))
    rng.shuffle(idx)
    halves = {i: ("adapt" if rank < len(main) // 2 else "test") for rank, i in enumerate(idx)}
    main = [D.Fact(**{**asdict(f), "split": halves[i]}) for i, f in enumerate(main)]
    calibration = [f for f in facts if f.split == "calibration"]
    return calibration + main, subjects, values, sorted(stable)
