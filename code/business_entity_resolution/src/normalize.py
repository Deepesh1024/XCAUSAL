"""Text normalization for business names and addresses.

Language-agnostic: any script is transliterated to ASCII (anyascii), then
lowercased and cleaned. Abbreviation / legal-form maps cover US, Indian and
French conventions but never depend on the country label.
"""
import re

import numpy as np
import pandas as pd
from anyascii import anyascii

# ---------------------------------------------------------------- junk
RE_URL = re.compile(r"(https?://\S+|www\.\S+|\S+@\S+)")
RE_DOMAIN_TLD = re.compile(r"\.(com|net|org|in|co|biz|info|fr|us|io)\b")
RE_NULL = re.compile(r"<\s*null\s*>|\bnull\b|\bn/a\b|\bnan\b")
RE_ORD = re.compile(r"\b(\d+)(st|nd|rd|th)\b")
RE_NONALNUM = re.compile(r"[^a-z0-9]+")
RE_NUM = re.compile(r"\d+")
RE_DIGIT_ALPHA = re.compile(r"(\d)([a-z])|([a-z])(\d)")

# ---------------------------------------------------------------- names
NAME_ABBR = {
    "pvt": "private", "pvtltd": "private limited", "prvt": "private", "priv": "private",
    "ltd": "limited", "ltda": "limited", "lmt": "limited", "ltdd": "limited",
    "corp": "corporation", "corpn": "corporation", "inc": "incorporated", "incorp": "incorporated",
    "co": "company", "cos": "company", "comp": "company", "cie": "company",
    "intl": "international", "int": "international", "natl": "national",
    "mfg": "manufacturing", "mfrs": "manufacturers", "svc": "services", "svcs": "services",
    "serv": "services", "srvs": "services", "tech": "technologies", "techs": "technologies",
    "assn": "association", "assoc": "associates", "bros": "brothers", "ent": "enterprises",
    "entp": "enterprises", "mgmt": "management", "dept": "department", "grp": "group",
    "hosp": "hospital", "univ": "university", "inst": "institute", "ind": "industries",
    "inds": "industries", "eng": "engineering", "engg": "engineering", "sys": "systems",
    "et": "and", "n": "and", "st": "saint",
}
# Legal forms / filler removed from the "core" name (US, IN, FR + generic).
LEGAL = {
    "private", "limited", "incorporated", "corporation", "company", "llc", "llp", "lp", "plc",
    "pllc", "pc", "pa", "ltd", "the", "and", "of", "dba", "aka", "opc", "huf",
    # French legal forms
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "scop", "scp", "selarl", "selas", "gie",
    "earl", "gaec", "scea", "ei", "eirl", "micro", "entreprise", "societe", "ste", "etablissements",
    "ets", "le", "la", "les", "de", "du", "des", "d", "l",
}

# ---------------------------------------------------------------- addresses
ADDR_ABBR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "avn": "avenue",
    "ln": "lane", "dr": "drive", "drv": "drive", "ct": "court", "blvd": "boulevard", "bd": "boulevard",
    "bld": "boulevard", "hwy": "highway", "pkwy": "parkway", "pky": "parkway", "pl": "place",
    "sq": "square", "cir": "circle", "trl": "trail", "ter": "terrace", "terr": "terrace",
    "fl": "floor", "flr": "floor", "ste": "suite", "apt": "apartment", "bldg": "building",
    "no": "number", "nos": "number", "num": "number", "hno": "number", "plot": "plot",
    "mkt": "market", "ngr": "nagar", "opp": "opposite", "nr": "near", "bldng": "building",
    "cplx": "complex", "extn": "extension", "ext": "extension", "sect": "sector", "sec": "sector",
    "dist": "district", "distt": "district", "tq": "taluk", "tal": "taluk", "po": "post",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest",
    # French
    "r": "rue", "ch": "chemin", "chem": "chemin", "imp": "impasse", "all": "allee", "rte": "route",
    "fbg": "faubourg", "qu": "quai", "crs": "cours", "pass": "passage", "res": "residence",
    "bis": "bis", "zi": "zone industrielle", "za": "zone activite", "zac": "zone activite",
}
ADDR_STOP = {"number", "near", "opposite", "city", "the", "of", "and", "floor", "suite",
             "building", "apartment", "district", "post", "de", "du", "des", "la", "le", "les", "d", "l"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "ts", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl",
    "jammu and kashmir": "jk", "puducherry": "py", "pondicherry": "py", "chandigarh": "ch",
}
STATES = {**US_STATES, **IN_STATES}
STATE_RE = re.compile(r"\b(" + "|".join(sorted(STATES, key=len, reverse=True)) + r")\b")

NUM_UNITS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
             "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
             "nineteen": 19, "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
             "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
             "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
             "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19}
NUM_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
            "eighty": 80, "ninety": 90, "twentieth": 20, "thirtieth": 30, "fortieth": 40,
            "fiftieth": 50, "sixtieth": 60, "seventieth": 70, "eightieth": 80, "ninetieth": 90}


def base_clean(s):
    """Transliterate, lowercase, strip URLs/nulls/junk; returns space-joined tokens."""
    if not s:
        return ""
    s = anyascii(s).lower().replace("?", "")
    s = RE_URL.sub(" ", s)
    s = RE_DOMAIN_TLD.sub(" ", s)
    s = RE_NULL.sub(" ", s)
    s = s.replace("&", " and ").replace("@", " at ")
    s = RE_ORD.sub(r"\1", s)
    s = RE_NONALNUM.sub(" ", s)
    return " ".join(s.split())


def words_to_numbers(tokens):
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if t in NUM_TENS:
            v = NUM_TENS[t]
            if i + 1 < len(tokens) and tokens[i + 1] in NUM_UNITS and NUM_UNITS[tokens[i + 1]] < 10:
                v += NUM_UNITS[tokens[i + 1]]
                i += 1
            out.append(str(v))
        elif t in NUM_UNITS:
            out.append(str(NUM_UNITS[t]))
        else:
            out.append(t)
        i += 1
    return out


def norm_name(raw):
    s = base_clean(raw)
    toks = []
    for t in s.split():
        toks.extend(NAME_ABBR.get(t, t).split())
    # collapse immediate repeats ("inc inc")
    dedup = [t for i, t in enumerate(toks) if i == 0 or t != toks[i - 1]]
    return " ".join(dedup)


def core_name(name_norm):
    toks = [t for t in name_norm.split() if t not in LEGAL]
    return " ".join(toks) if toks else name_norm


def norm_addr(raw):
    s = base_clean(raw)
    s = STATE_RE.sub(lambda m: STATES[m.group(1)], s)
    s = RE_DIGIT_ALPHA.sub(lambda m: (m.group(1) or m.group(3)) + " " + (m.group(2) or m.group(4)), s)
    toks = []
    for t in s.split():
        toks.extend(ADDR_ABBR.get(t, t).split())
    toks = words_to_numbers(toks)
    return " ".join(toks)


def skeleton(s):
    """Phonetic consonant skeleton, robust to vowel typos and transliteration."""
    s = s.replace("ph", "f").replace("sh", "s").replace("th", "t").replace("ck", "k")
    s = s.translate(SKEL_TABLE)
    out = []
    for ch in s:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


SKEL_TABLE = str.maketrans({"a": None, "e": None, "i": None, "o": None, "u": None, "y": None,
                            "h": None, "w": None, "c": "k", "q": "k", "z": "s", "v": "b",
                            "m": "n", "x": "ks", "j": "g", " ": " "})


NORM_COLS = ["name_n", "core", "addr_n", "core_skel", "core_cat_skel", "nums", "addr_alpha",
             "nonlatin", "country"]


def _norm_chunk(df):
    return normalize_df(df, workers=1)[NORM_COLS]


def normalize_df(df, workers=1):
    """Add normalized columns to a source DataFrame and return it.
    workers>1 splits the frame into chunks processed by a process pool."""
    if workers > 1 and len(df) > 50000:
        from concurrent.futures import ProcessPoolExecutor
        n_chunks = workers * 4
        bounds = np.linspace(0, len(df), n_chunks + 1).astype(int)
        parts = [df.iloc[a:b][["business_name", "business_address", "country"]]
                 for a, b in zip(bounds[:-1], bounds[1:])]
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_norm_chunk, parts))
        out = pd.concat(res)
        for c in NORM_COLS:
            df[c] = out[c].values
        return df
    df["name_n"] = [norm_name(x) for x in df["business_name"].values]
    df["core"] = [core_name(x) for x in df["name_n"].values]
    df["addr_n"] = [norm_addr(x) for x in df["business_address"].values]
    df["core_skel"] = [" ".join(skeleton(t) for t in c.split()) for c in df["core"].values]
    df["core_cat_skel"] = [skeleton(c.replace(" ", "")) for c in df["core"].values]
    df["nums"] = [" ".join(RE_NUM.findall(a)) for a in df["addr_n"].values]
    df["addr_alpha"] = [" ".join(t for t in a.split() if not t.isdigit() and t not in ADDR_STOP
                                 and len(t) > 1) for a in df["addr_n"].values]
    df["nonlatin"] = np.array([any(ord(c) > 127 for c in x) for x in df["business_name"].values],
                              dtype=np.int8)
    df["country"] = df["country"].str.strip().str.lower()
    return df
