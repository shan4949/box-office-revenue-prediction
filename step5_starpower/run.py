"""
Step 5: Star-Power Feature
For each movie computes the mean real revenue of its lead cast (order < 3) and director
over films released *strictly before* this movie's release_date.

Cold-start (no prior qualifying films): star_power = NaN, is_debut = 1.
log_star_power = log(star_power) when available; NaN otherwise.
LightGBM handles NaN natively at split time.

Limitation: career history is built only from the 3,991-film universe (films with
reported budget and revenue). Unreported productions are invisible to the lookup.
"""
import os
import json
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STEP4_OUT  = os.path.join(SCRIPT_DIR, '..', 'step4_features', 'output', 'features.parquet')
OUT_DIR    = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)


def parse_json_list(val):
    if pd.isna(val):
        return []
    try:
        result = json.loads(val)
        return result if isinstance(result, list) else []
    except Exception:
        return []


def extract_persons(cast_json, crew_json):
    """Return list of integer person_ids: lead cast (order<3) + director(s)."""
    persons = []

    for member in parse_json_list(cast_json):
        if isinstance(member, dict) and member.get('order', 99) < 3:
            pid = member.get('id')
            if pid is not None:
                persons.append(int(pid))

    for member in parse_json_list(crew_json):
        if isinstance(member, dict) and member.get('job') == 'Director':
            pid = member.get('id')
            if pid is not None:
                persons.append(int(pid))

    return persons


def build_person_lookup(df):
    """
    Build {person_id: (sorted_dates_ns, sorted_revs)} from the universe.
    Only include rows with revenue_real > 0.
    """
    person_films: dict[int, list] = {}

    for _, row in df.iterrows():
        if pd.isna(row['release_date']) or row['revenue_real'] <= 0:
            continue
        date_ns = int(row['release_date'].value)   # nanoseconds epoch
        rev     = float(row['revenue_real'])
        for pid in extract_persons(row['cast'], row['crew']):
            person_films.setdefault(pid, []).append((date_ns, rev))

    # Sort and convert to numpy
    person_dates: dict[int, np.ndarray] = {}
    person_revs:  dict[int, np.ndarray] = {}
    for pid, entries in person_films.items():
        entries.sort(key=lambda x: x[0])
        ds, rs = zip(*entries)
        person_dates[pid] = np.array(ds, dtype=np.int64)
        person_revs[pid]  = np.array(rs, dtype=np.float64)

    return person_dates, person_revs


def compute_prior_mean(pid, date_ns, person_dates, person_revs):
    """Mean revenue of person's films released strictly before date_ns."""
    if pid not in person_dates:
        return np.nan
    idx = int(np.searchsorted(person_dates[pid], date_ns, side='left'))
    if idx == 0:
        return np.nan
    return float(np.mean(person_revs[pid][:idx]))


def main():
    print("=== Step 5: Star-Power Feature ===")

    df = pd.read_parquet(STEP4_OUT)
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df = df.sort_values('release_date').reset_index(drop=True)
    print(f"  Input: {len(df):,} rows sorted by release_date")

    print("  Building person-film lookup …")
    person_dates, person_revs = build_person_lookup(df)
    print(f"  Tracked {len(person_dates):,} unique persons")

    print("  Computing star-power scores …")
    star_powers = np.full(len(df), np.nan, dtype=np.float64)
    is_debuts   = np.ones(len(df),  dtype=np.int8)

    for i, row in df.iterrows():
        if pd.isna(row['release_date']):
            continue
        date_ns = int(row['release_date'].value)
        persons = extract_persons(row['cast'], row['crew'])
        if not persons:
            continue

        person_means = []
        any_veteran  = False
        for pid in persons:
            pmean = compute_prior_mean(pid, date_ns, person_dates, person_revs)
            if not np.isnan(pmean):
                any_veteran = True
                person_means.append(pmean)

        if person_means:
            star_powers[i] = float(np.mean(person_means))
            is_debuts[i]   = 0 if any_veteran else 1

    df['star_power']     = star_powers
    df['is_debut']       = is_debuts.astype(int)
    df['log_star_power'] = np.where(df['star_power'] > 0, np.log(df['star_power']), np.nan)

    # --- Temporal correctness check ----------------------------------------
    print("\n[VERIFY] Temporal correctness: star_power must change if cutoff moves")
    sample_2010 = df[df['release_date'].dt.year == 2010].dropna(subset=['star_power'])
    if not sample_2010.empty:
        row2010  = sample_2010.iloc[0]
        persons2 = extract_persons(row2010['cast'], row2010['crew'])
        pid0     = next((p for p in persons2 if p in person_dates), None)
        if pid0:
            dn_movie   = int(row2010['release_date'].value)
            dn_jan2010 = int(pd.Timestamp('2010-01-01').value)
            mean_before_jan2010 = compute_prior_mean(pid0, dn_jan2010, person_dates, person_revs)
            mean_before_movie   = compute_prior_mean(pid0, dn_movie,   person_dates, person_revs)
            print(f"  Movie : {row2010['title']} ({row2010['release_date'].date()})")
            print(f"  Person {pid0}: prior mean truncated @ 2010-01-01 = {mean_before_jan2010}")
            print(f"  Person {pid0}: prior mean truncated @ movie date = {mean_before_movie}")
            different = (mean_before_jan2010 != mean_before_movie or
                         (np.isnan(mean_before_jan2010) != np.isnan(mean_before_movie)))
            print(f"  Values differ = {different} (True means look-up is correctly date-gated)")

    # --- Summary stats -------------------------------------------------------
    n_valid  = df['star_power'].notna().sum()
    n_debut  = int(df['is_debut'].sum())
    print(f"\n[VERIFY] star_power non-null : {n_valid:,} / {len(df):,}")
    print(f"[VERIFY] is_debut = 1       : {n_debut:,}")
    print(f"[VERIFY] mean star_power    : ${df['star_power'].mean():,.0f}")

    out_path = os.path.join(OUT_DIR, 'with_starpower.parquet')
    df.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
