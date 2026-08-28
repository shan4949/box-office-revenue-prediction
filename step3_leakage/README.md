# Step 3 — Leakage Audit

## What it does
Removes every column that a real forecaster would not have access to **before** a film
is released. These are the most dangerous form of data leakage: metrics that look like
useful predictors but are computed from post-release audience behaviour.

| Dropped column | Why it leaks |
|----------------|-------------|
| `vote_average` | User ratings — only exist after public release |
| `vote_count` | Number of ratings — same |
| `popularity` | TMDB engagement score — driven by post-release traffic |
| `homepage` | Sometimes populated/updated after release |
| `overview` | Free-text synopsis can act as a proxy for post-release marketing success |
| `tagline` | Marketing copy — similarly leaky as a free-text signal |
| `status` | Always "Released" for rows in the universe — tautological |

### Why `overview` and `tagline`?
Even though studios write these before release, including raw text in a tabular model
often encodes post-hoc editorial signals (longer, more enthusiastic copy for bigger
films). Excluding them is the conservative choice; they could be re-added with care.

## Inputs
```
step2_cpi/output/deflated.parquet
```

## Outputs
```
step3_leakage/output/clean.parquet   (same rows, leaky columns removed)
```

## Verification checks
| Check | Expected |
|-------|----------|
| `vote_average` absent | ✓ |
| `vote_count` absent | ✓ |
| `popularity` absent | ✓ |
| `homepage` absent | ✓ |
| `overview` absent | ✓ |
| `tagline` absent | ✓ |
| `status` absent | ✓ |

The script raises `RuntimeError` if any of the above are still present after the drop.

## How to run
```bash
../.venv/bin/python run.py
```
Runtime: < 2 seconds.
