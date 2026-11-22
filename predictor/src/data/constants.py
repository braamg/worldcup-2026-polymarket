"""Static reference data: confederations, team name aliases, importance weights."""

# ---------------------------------------------------------------------------
# Team name aliases: raw dataset name → canonical name used throughout the system.
# The martj42 dataset is generally good English names; only alias known variants.
# ---------------------------------------------------------------------------
TEAM_NAME_ALIASES: dict[str, str] = {
    # Asian variants
    "Korea Republic":           "South Korea",
    "Korea DPR":                "North Korea",
    "IR Iran":                  "Iran",
    "China PR":                 "China",
    "Kyrgyz Republic":          "Kyrgyzstan",
    "East Timor":               "Timor-Leste",
    # African variants
    "Congo DR":                 "DR Congo",
    "Zaire":                    "DR Congo",
    "Cote d'Ivoire":            "Ivory Coast",
    "Cape Verde Islands":       "Cape Verde",
    "Upper Volta":              "Burkina Faso",
    "Dahomey":                  "Benin",
    "Swaziland":                "Eswatini",
    # European variants
    "Czech Republic":           "Czechia",
    "Czechia":                  "Czechia",       # keep canonical form
    "Macedonia":                "North Macedonia",
    "Türkiye":                  "Turkey",         # normalize to one form
    "Bosnia-Herzegovina":       "Bosnia and Herzegovina",
    # CONCACAF variants
    "USA":                      "United States",   # common shorthand → canonical
    "St Kitts and Nevis":       "Saint Kitts and Nevis",
    "St Lucia":                 "Saint Lucia",
    "St Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "Curacao":                  "Curaçao",
    # Historical team name variants that appear in the dataset
    "German DR":                "East Germany",
    "Taiwan":                   "Chinese Taipei",  # AFC canonical name
}

# ---------------------------------------------------------------------------
# Confederation membership, keyed on CANONICAL team name.
# Teams with unknown confederation will surface in the QA report.
# ---------------------------------------------------------------------------
CONFEDERATION: dict[str, str] = {
    # ----- UEFA – Europe -----
    "Albania": "UEFA", "Andorra": "UEFA", "Armenia": "UEFA",
    "Austria": "UEFA", "Azerbaijan": "UEFA", "Belarus": "UEFA",
    "Belgium": "UEFA", "Bosnia and Herzegovina": "UEFA", "Bulgaria": "UEFA",
    "Croatia": "UEFA", "Cyprus": "UEFA", "Czechia": "UEFA",
    "Denmark": "UEFA", "England": "UEFA", "Estonia": "UEFA",
    "Faroe Islands": "UEFA", "Finland": "UEFA", "France": "UEFA",
    "Georgia": "UEFA", "Germany": "UEFA", "Gibraltar": "UEFA",
    "Greece": "UEFA", "Hungary": "UEFA", "Iceland": "UEFA",
    "Ireland": "UEFA", "Republic of Ireland": "UEFA",
    "Israel": "UEFA", "Italy": "UEFA",
    "Kazakhstan": "UEFA", "Kosovo": "UEFA",
    "Latvia": "UEFA", "Liechtenstein": "UEFA", "Lithuania": "UEFA",
    "Luxembourg": "UEFA", "Malta": "UEFA", "Moldova": "UEFA",
    "Montenegro": "UEFA", "Netherlands": "UEFA",
    "North Macedonia": "UEFA", "Northern Ireland": "UEFA",
    "Norway": "UEFA", "Poland": "UEFA", "Portugal": "UEFA",
    "Romania": "UEFA", "Russia": "UEFA", "San Marino": "UEFA",
    "Scotland": "UEFA", "Serbia": "UEFA", "Slovakia": "UEFA",
    "Slovenia": "UEFA", "Spain": "UEFA", "Sweden": "UEFA",
    "Switzerland": "UEFA", "Turkey": "UEFA", "Ukraine": "UEFA",
    "Wales": "UEFA",
    # Historical European
    "Yugoslavia": "UEFA", "Soviet Union": "UEFA", "Czechoslovakia": "UEFA",
    "East Germany": "UEFA", "West Germany": "UEFA",
    "Serbia and Montenegro": "UEFA",

    # ----- CONMEBOL – South America -----
    "Argentina": "CONMEBOL", "Bolivia": "CONMEBOL", "Brazil": "CONMEBOL",
    "Chile": "CONMEBOL", "Colombia": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Paraguay": "CONMEBOL", "Peru": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Venezuela": "CONMEBOL",

    # ----- CONCACAF – North/Central America & Caribbean -----
    "Antigua and Barbuda": "CONCACAF", "Aruba": "CONCACAF",
    "Bahamas": "CONCACAF", "Barbados": "CONCACAF", "Belize": "CONCACAF",
    "Bermuda": "CONCACAF", "British Virgin Islands": "CONCACAF",
    "Canada": "CONCACAF", "Cayman Islands": "CONCACAF",
    "Costa Rica": "CONCACAF", "Cuba": "CONCACAF", "Curaçao": "CONCACAF",
    "Dominica": "CONCACAF", "Dominican Republic": "CONCACAF",
    "El Salvador": "CONCACAF", "French Guiana": "CONCACAF",
    "Grenada": "CONCACAF", "Guadeloupe": "CONCACAF",
    "Guatemala": "CONCACAF", "Guyana": "CONCACAF",
    "Haiti": "CONCACAF", "Honduras": "CONCACAF", "Jamaica": "CONCACAF",
    "Martinique": "CONCACAF", "Mexico": "CONCACAF", "Montserrat": "CONCACAF",
    "Nicaragua": "CONCACAF", "Panama": "CONCACAF", "Puerto Rico": "CONCACAF",
    "Saint Kitts and Nevis": "CONCACAF", "Saint Lucia": "CONCACAF",
    "Saint Vincent and the Grenadines": "CONCACAF",
    "Suriname": "CONCACAF",
    "Trinidad and Tobago": "CONCACAF", "Trinidad & Tobago": "CONCACAF",
    "Turks and Caicos Islands": "CONCACAF",
    "United States": "CONCACAF", "USA": "CONCACAF",
    "US Virgin Islands": "CONCACAF", "United States Virgin Islands": "CONCACAF",
    "Netherlands Antilles": "CONCACAF",  # historical

    # ----- CAF – Africa -----
    "Algeria": "CAF", "Angola": "CAF", "Benin": "CAF",
    "Botswana": "CAF", "Burkina Faso": "CAF", "Burundi": "CAF",
    "Cameroon": "CAF", "Cape Verde": "CAF",
    "Central African Republic": "CAF", "Chad": "CAF",
    "Comoros": "CAF", "Congo": "CAF", "DR Congo": "CAF",
    "Djibouti": "CAF", "Egypt": "CAF", "Equatorial Guinea": "CAF",
    "Eritrea": "CAF", "Eswatini": "CAF", "Ethiopia": "CAF",
    "Gabon": "CAF", "Gambia": "CAF", "Ghana": "CAF",
    "Guinea": "CAF", "Guinea-Bissau": "CAF",
    "Ivory Coast": "CAF", "Kenya": "CAF", "Lesotho": "CAF",
    "Liberia": "CAF", "Libya": "CAF", "Madagascar": "CAF",
    "Malawi": "CAF", "Mali": "CAF", "Mauritania": "CAF",
    "Mauritius": "CAF", "Morocco": "CAF", "Mozambique": "CAF",
    "Namibia": "CAF", "Niger": "CAF", "Nigeria": "CAF",
    "Rwanda": "CAF", "São Tomé and Príncipe": "CAF",
    "Senegal": "CAF", "Seychelles": "CAF", "Sierra Leone": "CAF",
    "Somalia": "CAF", "South Africa": "CAF", "South Sudan": "CAF",
    "Sudan": "CAF", "Tanzania": "CAF", "Togo": "CAF",
    "Tunisia": "CAF", "Uganda": "CAF", "Zambia": "CAF",
    "Zimbabwe": "CAF",

    # ----- AFC – Asia -----
    "Afghanistan": "AFC", "Australia": "AFC",  # moved from OFC in 2006
    "Bahrain": "AFC", "Bangladesh": "AFC", "Bhutan": "AFC",
    "Brunei": "AFC", "Cambodia": "AFC", "China": "AFC",
    "Chinese Taipei": "AFC", "Guam": "AFC", "Hong Kong": "AFC",
    "India": "AFC", "Indonesia": "AFC", "Iran": "AFC",
    "Iraq": "AFC", "Japan": "AFC", "Jordan": "AFC",
    "Kuwait": "AFC", "Kyrgyzstan": "AFC", "Laos": "AFC",
    "Lebanon": "AFC", "Macau": "AFC", "Malaysia": "AFC",
    "Maldives": "AFC", "Mongolia": "AFC", "Myanmar": "AFC",
    "Nepal": "AFC", "North Korea": "AFC",
    "Northern Mariana Islands": "AFC",
    "Oman": "AFC", "Pakistan": "AFC", "Palestine": "AFC",
    "Philippines": "AFC", "Qatar": "AFC", "Saudi Arabia": "AFC",
    "Singapore": "AFC", "South Korea": "AFC", "Sri Lanka": "AFC",
    "Syria": "AFC", "Tajikistan": "AFC", "Thailand": "AFC",
    "Timor-Leste": "AFC", "Turkmenistan": "AFC",
    "United Arab Emirates": "AFC", "Uzbekistan": "AFC",
    "Vietnam": "AFC", "Yemen": "AFC",
    # Historical Asian
    "South Vietnam": "AFC", "North Vietnam": "AFC",
    "North Yemen": "AFC", "South Yemen": "AFC",

    # ----- OFC – Oceania -----
    "American Samoa": "OFC", "Cook Islands": "OFC", "Fiji": "OFC",
    "New Caledonia": "OFC", "New Zealand": "OFC",
    "Papua New Guinea": "OFC", "Samoa": "OFC",
    "Solomon Islands": "OFC", "Tahiti": "OFC",
    "Tonga": "OFC", "Vanuatu": "OFC",

    # ---- Additional FIFA members observed in martj42 dataset ----
    "Monaco": "UEFA",           # smallest FIFA member; plays WCQ
    "Anguilla": "CONCACAF",
    "Zanzibar": "CAF",          # semi-autonomous; plays in CECAFA but tracked here
}

# ---------------------------------------------------------------------------
# Match importance weights.
# List of (substring_pattern, weight); checked case-insensitively, first match wins.
# ---------------------------------------------------------------------------
IMPORTANCE_PATTERNS: list[tuple[str, int]] = [
    # IMPORTANT: more-specific patterns MUST come before patterns whose string is
    # a substring of theirs.  E.g. "World Cup qualification" contains "World Cup",
    # so qualification entries are listed first throughout.

    # ---- World Cup qualification (before "FIFA World Cup") ----
    ("World Cup qualification",                 5),
    ("World Cup Qualification",                 5),
    # ---- Continental qualification (before parent tournament names) ----
    ("UEFA Euro qualification",                 4),
    ("UEFA Euro Qualification",                 4),
    ("Copa América qualification",              4),
    ("Copa America qualification",              4),
    ("Africa Cup of Nations qualification",     4),
    ("African Cup of Nations qualification",    4),
    ("AFC Asian Cup qualification",             4),
    ("CONCACAF Gold Cup qualification",         4),
    ("CFU Caribbean Cup",                       4),   # CONCACAF sub-regional
    ("CONCACAF Championship",                   4),
    # ---- Nations Leagues (before generic "Nations League") ----
    ("UEFA Nations League",                     4),
    ("CONCACAF Nations League",                 4),
    ("CONMEBOL-UEFA",                           4),   # Finalissima
    ("Nations League",                          3),
    # ---- Major tournament finals ----
    ("FIFA World Cup",                          8),
    ("UEFA European Championship",              7),
    ("UEFA Euro",                               7),
    ("Copa América",                            7),
    ("Copa America",                            7),
    ("Africa Cup of Nations",                   7),
    ("African Cup of Nations",                  7),
    ("AFC Asian Cup",                           7),
    ("Asian Cup",                               7),
    ("CONCACAF Gold Cup",                       7),
    ("Gold Cup",                                7),
    ("OFC Nations Cup",                         7),
    ("FIFA Confederations Cup",                 6),
    # ---- Other regional competitions ----
    ("African Nations Championship",            4),
    ("Gulf Cup",                                3),
    ("COSAFA",                                  3),
    ("CECAFA",                                  3),
    ("WAFU",                                    3),
    ("EAFF",                                    3),
    ("AFF Championship",                        3),
    ("SAFF",                                    3),
    ("Asian Games",                             3),
    # ---- Olympics / multi-sport ----
    ("Olympic Games",                           3),
    ("Pan American Games",                      2),
    # ---- Friendly / fallback ----
    ("Friendly",                                1),
    ("friendly",                                1),
]
DEFAULT_IMPORTANCE: int = 2  # unmatched tournament names
