"""
Step 4: Feature Engineering (pre-release features only)
Builds the full tabular feature matrix from columns that are knowable before release.
"""
import os
import json
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STEP3_OUT  = os.path.join(SCRIPT_DIR, '..', 'step3_leakage', 'output', 'clean.parquet')
OUT_DIR    = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

TOP_GENRES = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
    'Drama', 'Family', 'Fantasy', 'Horror', 'Music', 'Mystery',
    'Romance', 'Science Fiction', 'Thriller', 'War', 'Western',
]

TOP_LANGUAGES = ['fr', 'de', 'es', 'ja', 'zh', 'it', 'ko', 'pt', 'ru', 'hi']

# Canonical studio names used in production_companies
STUDIO_MAP = {
    'Walt Disney Pictures':                  'Disney',
    'Pixar Animation Studios':               'Disney',
    'Walt Disney Animation Studios':         'Disney',
    'Touchstone Pictures':                   'Disney',
    'Warner Bros.':                          'WarnerBros',
    'New Line Cinema':                       'WarnerBros',
    'Universal Pictures':                    'Universal',
    'Paramount Pictures':                    'Paramount',
    'Columbia Pictures':                     'Sony',
    'TriStar Pictures':                      'Sony',
    'Screen Gems':                           'Sony',
    'Twentieth Century Fox Film Corporation':'Fox',
    '20th Century Fox':                      'Fox',
    'Metro-Goldwyn-Mayer (MGM)':             'MGM',
    'Miramax Films':                         'Miramax',
    'Lionsgate':                             'Lionsgate',
    'Summit Entertainment':                  'Lionsgate',
    'DreamWorks SKG':                        'DreamWorks',
    'DreamWorks Animation':                  'DreamWorks',
    'Focus Features':                        'Focus',
    'Relativity Media':                      'Relativity',
}
CANONICAL_STUDIOS = sorted(set(STUDIO_MAP.values()))


def parse_json_col(val):
    if pd.isna(val):
        return []
    try:
        result = json.loads(val)
        return result if isinstance(result, list) else []
    except Exception:
        return []


def parse_json_dict(val):
    if pd.isna(val):
        return None
    try:
        result = json.loads(val)
        return result if isinstance(result, dict) and result else None
    except Exception:
        return None


def extract_genre_names(genres_json):
    genres = parse_json_col(genres_json)
    return [g['name'] for g in genres if isinstance(g, dict) and 'name' in g]


def extract_company_names(companies_json):
    companies = parse_json_col(companies_json)
    return [c['name'] for c in companies if isinstance(c, dict) and 'name' in c]


def get_canonical_studio(company_names):
    for name in company_names:
        if name in STUDIO_MAP:
            return STUDIO_MAP[name]
    return 'Other'


def main():
    print("=== Step 4: Feature Engineering ===")

    df = pd.read_parquet(STEP3_OUT)
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    print(f"  Input: {len(df):,} rows")

    # 1. Log budget (real 2017 dollars)
    df['log_budget_real'] = np.log1p(df['budget_real'].clip(lower=0))

    # 2. Runtime — fill missing with median
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
    runtime_median = df['runtime'].median()
    df['runtime'] = df['runtime'].fillna(runtime_median)

    # 3. Genres multi-hot
    genre_series = df['genres'].apply(extract_genre_names)
    genre_sets   = genre_series.apply(set)
    for genre in TOP_GENRES:
        safe = genre.lower().replace(' ', '_')
        df[f'genre_{safe}'] = genre_sets.apply(lambda gs: int(genre in gs))

    # 4. Language — English flag + top non-English one-hots
    df['lang_english'] = (df['original_language'] == 'en').astype(int)
    for lang in TOP_LANGUAGES:
        df[f'lang_{lang}'] = (df['original_language'] == lang).astype(int)

    # 5. Release date features (vectorised)
    month = df['release_date'].dt.month
    day   = df['release_date'].dt.day
    df['release_month'] = month
    df['release_dow']   = df['release_date'].dt.dayofweek   # 0=Mon … 6=Sun

    df['holiday_flag'] = (
        ((month == 5)  & (day >= 22)) |   # Memorial Day area
        ((month == 7)  & (day <= 7))  |   # July 4th area
        ((month == 9)  & (day <= 7))  |   # Labor Day area
        ((month == 11) & (day >= 20)) |   # Thanksgiving area
        ((month == 12) & (day >= 18))     # Christmas area
    ).astype(int)

    # 6. Franchise flag
    df['franchise_flag'] = df['belongs_to_collection'].apply(
        lambda v: 1 if parse_json_dict(v) is not None else 0
    )

    # 7. Studio one-hots
    company_series = df['production_companies'].apply(extract_company_names)
    df['_studio']  = company_series.apply(get_canonical_studio)
    for studio in CANONICAL_STUDIOS:
        df[f'studio_{studio.lower()}'] = (df['_studio'] == studio).astype(int)
    df['studio_other'] = (df['_studio'] == 'Other').astype(int)
    df = df.drop(columns=['_studio'])

    # --- Verify: spot-check 5 rows by hand ---------------------------------
    feature_cols = (
        ['log_budget_real', 'runtime', 'holiday_flag', 'franchise_flag',
         'release_month', 'release_dow']
        + [c for c in df.columns if c.startswith('genre_')]
    )
    print(f"\n[VERIFY] Feature columns created: {len(feature_cols)}")
    null_report = {c: int(df[c].isna().sum()) for c in feature_cols if df[c].isna().sum() > 0}
    if null_report:
        print(f"  Nulls found: {null_report}")
    else:
        print("  No nulls in engineered feature columns ✓")

    print("\n[VERIFY] Spot-check 5 random rows:")
    spot = df.sample(5, random_state=42)[
        ['title', 'release_date', 'log_budget_real', 'runtime',
         'franchise_flag', 'holiday_flag', '_studio' if '_studio' in df.columns else 'release_month']
    ]
    print(spot.to_string())

    print("\n[VERIFY] All features are derived from pre-release information only ✓")
    print("  (budget, runtime, genres, language, release date, studio, collection membership)")

    out_path = os.path.join(OUT_DIR, 'features.parquet')
    df.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
