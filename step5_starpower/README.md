# Step 5 — Star-Power Feature

## What it does
Adds a single numeric feature — `star_power` — that captures how commercially successful
a film's key talent has been **historically**, measured in real revenue dollars.

### Who counts as "key talent"?
- **Lead cast**: cast members with `order < 3` in the TMDB cast list (top-billed actors)
- **Director**: crew members with `job == 'Director'`

### How is it computed?
For a target movie released on date **D**:
1. For each key talent person **p**, find all other films in the universe released on dates **< D**.
2. Take the **mean real revenue** of those prior films.
3. Average the per-person means across all key talent in the group.

Cold-start (no qualifying prior films for any person): `star_power = NaN`, `is_debut = 1`.

### Temporal correctness
The strict `< D` cutoff ensures no future data bleeds in.
The verification step confirms the score **changes** if the cutoff date changes — proving
the look-up is actually date-gated rather than using a static aggregate.

### Outputs
| Column | Description |
|--------|-------------|
| `star_power` | Mean real revenue of key talent's prior films (NaN if debut) |
| `is_debut` | 1 if all key talent have no prior qualifying films, else 0 |
| `log_star_power` | log(star_power); NaN for debuts — LightGBM handles NaN at split time |

### Known limitation
Career history is built only from the **3,991-film universe** (budget > 0 & revenue > 0 reported).
Unreported / unmeasured productions are invisible to the lookup.
A debut here means "no prior *reported* films", not "first film ever".

## Inputs
```
step4_features/output/features.parquet
```

## Outputs
```
step5_starpower/output/with_starpower.parquet   (adds star_power, is_debut, log_star_power)
```

## Verification checks
| Check | Expected |
|-------|----------|
| star_power changes when cutoff moves 2009→2010 | True |
| Fraction non-null | ~60–80% |
| is_debut rows | Remaining ~20–40% |

## How to run
```bash
../.venv/bin/python run.py
```
Runtime: ~60–120 seconds (iterates over 3,991 rows twice: once for lookup, once for scoring).
