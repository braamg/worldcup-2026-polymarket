# Phase Build Log — Issues Encountered, Decisions Made

This document records non-obvious decisions and issues that arose during each phase,
intended for an independent reviewer to evaluate. Each entry explains the problem,
the options considered, and why the chosen fix was made.

---

## Phase 1 — Historical Match Table

### Issue: Future WC 2026 fixtures pre-entered with null scores

**Problem**: The martj42/international_results dataset includes 70 pre-scheduled WC 2026
fixtures (June 11 – June 27, 2026) with null scores. Without a guard, the `result` column
defaulted to "D" (draw) for all null goal-diffs, and training code could accidentally include
these zero-score rows.

**Fix**: Added an `is_played` flag (`home_score.notna() & away_score.notna()`). All result
assignment logic gates on `is_played=True`. Unplayed rows get `result=None`.

**Review note**: Any downstream code that trains on results MUST filter `is_played=True`.
This is documented but not enforced by schema — a leakage risk if a future developer forgets.

---

### Issue: IMPORTANCE_PATTERNS substring ordering bug

**Problem**: The pattern list checked "FIFA World Cup" before "FIFA World Cup qualification".
Since Python's `in` operator matches substrings, ALL qualification matches (5,000+ rows)
were assigned weight=8 (World Cup finals) instead of weight=5. The QA report showed
~9,807 World Cup finals matches when the correct count is ~1,036.

**Fix**: Moved all `qualification` patterns BEFORE their parent tournament name throughout
the entire `IMPORTANCE_PATTERNS` list. Added an inline comment warning that ordering is
critical and explains why.

**Review note**: The fix is correct but fragile — any new pattern additions must respect
the same ordering rule. A unit test asserting the count of importance-8 matches would
prevent regression.

---

### Issue: 4 duplicate rows in raw data

**Problem**: The martj42 dataset had 4 rows that were exact duplicates on
(date, home_team, away_team). The QA reported 2 pairs dropped.

**Fix**: `drop_duplicates(subset=["date","home_team","away_team"], keep="first")` in the
pipeline before derived column computation.

**Review note**: "keep=first" is arbitrary for true duplicates. For the 2 cases in the data,
both rows had identical scores, so either is correct. This is acceptable.

---

### Issue: 114 teams without confederation assignment

**Problem**: 114 team names in the dataset had no entry in `CONFEDERATION`. QA flagged
these as unknown.

**Decision**: Verified manually that all 114 are non-FIFA entities (CONIFA stateless nations,
Crown dependencies, overseas territories). None participate in FIFA tournaments. They are
harmless in historical data — they only affect matches between non-FIFA teams which are
excluded from the modeling era by the `home_conf.notna() & away_conf.notna()` filter on
cross-confederation matches.

**Review note**: These teams will have `has_market_value=False` in Phase 3 features and
cannot be predicted in a WC 2026 model. This is expected and acceptable.

---

## Phase 2 — Elo Ratings

### Issue: `np.select` TypeError with nullable Int64 goal_diff

**Problem**: `np.select(conditions, choices)` threw `TypeError: invalid entry 0 in condlist`
when conditions were built against a pandas `Int64` (nullable integer) column. Boolean masks
from nullable integer comparisons are `BooleanArray`, not `ndarray`.

**Fix**: Replaced `np.select` with sequential `.loc` assignments:
```python
results.loc[played & (results["goal_diff"] > 0), "result"] = "H"
```
This works because `.loc` accepts pandas Boolean arrays natively.

---

### Issue: Elo tuning period

**Decision**: Hyperparameters (k_base, home_advantage, mov_factor) were tuned by grid
search on matches STRICTLY before 2010-01-01, with evaluation on the 1994-2009 window.
Post-2010 data was left completely untouched for later validation.

**Grid result**: k_base=15, home_advantage=125, mov_factor=0.25. Best Brier=0.14025 on
1994-2009. Post-2010 validation: Brier=0.13707 (28.6% improvement over naive-0.5).

**Review note**: The Brier score spread across 27 grid combos was narrow (0.1403–0.1570),
meaning the model is not sensitive to exact parameter values. The selected corner of the
grid (lowest k_base, highest home_advantage, lowest mov_factor) is theoretically defensible:
sparse international data benefits from slow adaptation (low K), and home advantage is
genuinely high in international football.

---

### Issue: Non-FIFA teams inflate/contaminate Elo ratings

**Observation**: CONIFA teams (Catalonia, Padania, etc.) appear in the martj42 dataset
and go through the Elo engine. They start at 1500 and their ratings evolve. If any of
these teams played against FIFA full teams in irregular friendlies (possible), there could
be minor contamination.

**Decision**: Accepted as a known limitation for now. For WC 2026 prediction, we only
predict matches involving FIFA teams, and the contamination effect is small relative to
the number of FIFA-vs-FIFA matches.

**Action needed**: The `elo_current.parquet` file should be filtered to known FIFA member
teams before use in Phase 4+ modeling. A simple join against `CONFEDERATION.keys()` will
suffice.

---

### Issue: Ecuador (#5), Mexico (#6) anomalously high in current Elo

**Observation**: Current Elo rankings show Ecuador at 2099 and Mexico at 2078, both
unusually high. Ecuador's typical global ranking is top-20, not top-5.

**Root cause**: The tournament started June 11, 2026, and the match table already contains
2 played WC 2026 matches. If Ecuador or Mexico had strong results in those first matches,
their Elo would spike.

**Decision**: This is correct behavior — those WC results are facts. The Elo reflects
current form including those results.

---

## Phase 3 — Squad Market Value (implementation complete; scrape not yet run)

### Issue: Transfermarkt data source selection

**Options considered**:
1. `dcaribou/transfermarkt-datasets` (222 MB ZIP with `player_valuations` table)
2. Direct Transfermarkt HTML scraping of national team pages

**Decision**: Direct scraping. The `dcaribou` dataset focuses on club competitions
(Bundesliga, Champions League etc.) and does not contain pre-aggregated national team
squad values. Mapping individual players to national team squads at historical dates
would require a complex player-to-country assignment and would likely produce worse
coverage than the direct TM national team pages.

**Rationale for scraping**: Transfermarkt national team pages at
`https://www.transfermarkt.us/team/startseite/verein/{tm_id}/saison_id/{year}`
contain per-player market values for the historical squad roster for that season.
73 teams × 21 seasons = 1533 requests. At 3s polite delay: ~76 minutes, one-time,
fully cached to `data/raw/transfermarkt/`.

---

### Critical issue: Header shows current squad value, NOT historical

**Problem discovered during testing**: The `<a class="data-header__market-value-wrapper">`
HTML element always returns the CURRENT squad's total market value regardless of which
`saison_id` is passed in the URL. This means the header approach gives the same ~€947M
for Germany across ALL historical seasons.

**Test evidence**: Fetching Germany `saison_id=2008, 2012, 2014, 2016, 2018, 2020, 2022,
2024` all returned €947M from the header.

**Fix**: Changed `parse_squad_value()` to SUM individual player market values from the
squad table (`table#yw1`). The per-player values DO reflect historical values for each
season. Test results with player-sum approach:
- Germany 2008: €344M, 2014: €796M, 2018: €1,302M, 2022: €980M, 2024: €988M
- France 2014: €506M, 2018: €1,616M (WC winner squad)
- Argentina 2014: €576M, 2018: €900M, 2022: €836M

These show historically plausible variation matching known squad value eras.

**Review note**: The player values in historical TM pages appear to be the player's
market value AT THE TIME of that season (not their current value). This needs
independent verification. Cross-check suggestion: look up Germany's 2014 squad on
an external source (e.g., Sportmob or Soccerway squad value reports) and compare to
the €796M figure obtained here.

**Data gap**: The 2004 season has no squad table in TM (table#yw1 not found).
Season range adjusted to 2005–2025.

---

### Issue: Transfermarkt URL slug is irrelevant — ID is the primary key

**Discovery**: Testing showed that `https://www.transfermarkt.us/WRONGSLUG/startseite/verein/3262/saison_id/2018`
returns exactly the same content as the correct-slug URL. The `/verein/{id}/` part is
the canonical identifier; the slug is purely decorative in the URL.

**Impact**: Simplifies the `TM_TEAMS` dict — only the integer ID is needed per team,
no slug required.

---

### Issue: Rest-days computation O(n²) bottleneck

**Problem**: The initial `_add_rest_days()` implementation used nested Python loops
with list comprehensions, taking >3 minutes for 49,405 matches.

**Fix**: Replaced with a vectorized pandas groupby + shift approach:
```python
all_apps["prev_date"] = all_apps.groupby("team")["date"].shift(1)
all_apps["rest_days"] = (all_apps["date"] - all_apps["prev_date"]).dt.days
```
Then pivoted back to per-match home/away columns using `pivot_table`.

---

### Issue: Wales TM ID confirmed wrong — removed from TM_TEAMS

**Finding**: ID 3391 maps to Kavala FC (a Greek football club), not the Welsh national
team. This was confirmed by inspecting the scraped HTML: `og:title` = "Kavala FC -
Club profile 22/23" and `<h1>` = "Kavala FC".

**Fix applied**:
- Wales removed from `TM_TEAMS` in `src/features/market_value.py`.
- All 21 Wales HTML files deleted from `data/raw/transfermarkt/`.
- Wales is NOT in WC 2026, so this has zero impact on tournament predictions.
- Wales does appear in historical training matches; those rows will have
  `has_market_value=False`, which is correct behaviour.

**Correct Wales TM ID**: Not confirmed. If Wales coverage is needed for future work,
find the correct verein ID from `https://www.transfermarkt.us/wales-nationalmannschaft/`
and add it back.

---

### Temporal integrity of market value snapshots

**Design decision**: For a match on date D, the squad value used is the snapshot from
`saison_id = D.year - 1` (via `valid_from = January 1 of saison_id + 1`).

This means:
- Match in June 2018 (WC 2018) → uses saison_id=2017 snapshot → valid_from=2018-01-01 ✓
- Match in December 2022 (WC 2022) → uses saison_id=2021 → valid_from=2022-01-01 ✓
- Match in June 2026 (WC 2026) → uses saison_id=2025 → valid_from=2026-01-01 ✓

This is conservative (using the completed prior season rather than the current season)
and guaranteed leak-free.

**Caveat**: The "valid_from" assignment is a modeling assumption — TM season values
are set at the start of each season (August), not January. Using January 1 means we
are implicitly assuming the prior season's values are still valid for matches in the
first ~7 months of the year. This is a small staleness trade-off vs. full leak-safety.

---

## General: Log-ratio normalization for market values

**Why log-ratio?** Raw squad value differences are confounded with era: a €200M difference
in 2006 is far more significant than in 2022. The log-ratio `log(MV_home / MV_away)`
is dimensionless and era-stable — it measures relative strength, not absolute euros.

**Formula**: `log_ratio_mv = log(mv_home / mv_away)`

Positive = home team richer. For neutral venues, this still measures squad quality
difference without home/away framing.

**Alternative considered**: Within-year z-score normalization. Rejected because: (a) it
requires knowing all squads playing in the same period, which could introduce subtle
lookahead if computed over a rolling window, and (b) the log-ratio is simpler and
has a clear probabilistic interpretation.

---

## Phase 4 — xG Experiment (DROPPED)

### Decision: DROP xG as a feature

**Investigation summary**: Using soccerdata 1.9.0's FBref interface, we verified all
available access paths for international match-level xG:

| Method | Has xG? | Evidence |
|---|---|---|
| `read_schedule()` | No | `schedule_INT-World Cup_2022.html` has no `data-stat="xg"` |
| `read_team_match_stats(stat_type='schedule')` | No | Columns: date/GF/GA/Poss/Formation... no xG |
| `read_team_match_stats(stat_type='shooting')` | No | Only Standard shooting (Sh/SoT/Gls); no Expected section |
| `read_events()` | No | Only parses goal/card/substitution events from match pages |

Confirmed by inspecting cached HTML files: FBref's competition and team pages for
international tournaments (`INT-World Cup`, `INT-European Championship`) contain no
xG data in any `data-stat` attribute. The soccerdata FBref parser has xG rename
logic (`"xG": "home_xg"`) that only fires for competitions where the schedule table
includes xG — which international tournament pages do not.

**Could xG be obtained via custom scraping?** Yes — FBref's individual match report
pages (`/en/matches/{game_id}`) contain StatsBomb xG. But soccerdata's `read_events`
does not parse it. Doing so would require custom HTML scraping of ~230 match pages
(WC 2018: 64, WC 2022: 64, Euro 2020: 51, Euro 2024: 51). Copa América is not
available in soccerdata's FBref interface at all.

**Coverage ceiling even with custom scraping**: ~230 matches out of 29,903 in the
modeling era (<0.8%). Insufficient to train a reliable xG feature or separate model.

**Why DROP (not KEEP for future use)**:
1. Not obtainable via the specified interface without custom work.
2. 230 matches is too thin to add reliable signal to the model.
3. CLAUDE.md explicitly anticipated this outcome ("Expect to drop this layer").
4. Phase instructions said to timebox — dead end confirmed cleanly.

**Reviewer note**: If StatsBomb releases WC data publicly in a soccerdata-compatible
format, or if soccerdata adds xG parsing to `read_events`, this could be revisited.
The architecture supports it: adding a `has_xg` flag and `xg_diff` feature to
`match_features.parquet` would be straightforward.

---

---

## Phase 5 — Modeling Table

### Modeling table built: data/processed/modeling_table.parquet

**Rows**: 29,903 (modern era 1994-01-01+, all played)
**Columns**: 31

**Key additions over match_features.parquet**:
- `training_weight`: importance-based weight (0.20 friendlies → 3.00 World Cup)
- `is_wc2026`: True for the 2 WC 2026 matches already played — marks prediction targets
- `rest_days_home_missing` / `rest_days_away_missing`: flags NaN imputed to 30.0

### Leakage audit: PASS

7/7 pytest tests passing in `tests/test_leakage_audit.py`.

Audit logic (200-match sample):
1. **Elo provenance**: elo_home_pre reconstructed from elo_timeseries (elo_after of last
   row with date < match_date) matches stored value within 0.02 tolerance. Zero violations.
2. **MV temporality**: valid_from(snapshot) = Jan 1 of (snapshot_season + 1) is strictly
   before match_date for all has_market_value=True rows. Zero violations.

The correct invariant is `valid_from < match_date`, NOT `season == match.year - 1`.
Some teams (Australia, Haiti, Curaçao, Panama, DR Congo, Mali, New Zealand) have stale
snapshots (TM data only goes back to 2018 for Australia, for example). Stale != leak:
valid_from is still before the match date.

### Missingness: key numbers (modern era)

| Feature | Missing | Notes |
|---|---|---|
| Elo features | 0% | Covers all history via DEFAULT_RATING |
| log_ratio_mv | 83.5% | Expected; only 4,932 rows have both teams covered |
| home_conf / away_conf | 3.2% | Non-FIFA entities in modern era matches |
| shootout_winner | 98.3% | Only populated for shootout-resolved matches |
| rest_days | 0% | Imputed to 30.0 for team's first match; flag column captures truth |

### MV imputation strategy for Phase 6 model training

- XGBoost: NaN-native, no imputation needed.
- Logistic regression / Dixon-Coles: use `has_market_value` as a mask and a separate
  no-MV intercept, or impute log_ratio_mv=0 (equal strength assumption) for missing rows.
  Document the choice in the model spec.

---

## Open Items / Review Checklist

- [x] Verify Wales TM ID (3391) — RESOLVED: ID was wrong (Kavala FC). Wales removed
      from TM_TEAMS. Bad HTML files deleted. Wales not in WC 2026; no impact on predictions.
- [x] After scrape completes, verify coverage — DONE: 16.9% overall (expected; 0% pre-2006,
      ~25% post-2006). CONMEBOL 56%, UEFA 26%, CAF 8%, AFC 5%.
- [ ] Verify historical squad values by cross-checking Germany 2014 (€796M) and
      France 2018 (€1,616M) against external historical reporting.
- [x] Verify individual player values are historical — CONFIRMED: Philipp Lahm appears on
      Germany 2014 page at €28M with club "Retired", consistent with his 2014 peak value.
      Values are NOT current (which would be €0).
- [x] Run the leakage audit — DONE (Phase 5). 7/7 tests passing. Stub replaced with
      real implementation covering Elo provenance and MV temporality.
- [x] WC 2026 contamination guard — ENFORCED. `is_wc2026=True` flag on 2 played WC 2026
      matches. Phase 6 model training must filter `is_wc2026=False`.
