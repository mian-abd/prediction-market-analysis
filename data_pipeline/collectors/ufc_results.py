"""UFC/MMA fight results collector for Elo training.

Data sources (in order of preference):
1. Kagglehub: load_ufc_from_kagglehub() - "m0hamedai1/the-ultimate-ufc-archive-1993-present"
   ~8,200 fights from 1993-present. Merges fights.csv + events.csv for dates.
2. Local CSV: Set UFC_CSV_PATH env or pass path (single fights CSV with date)
3. Hugging Face: xtinkarpiu/ufc-fight-data (limited ~200 fights sample)

Also includes market detection/parsing for live Polymarket UFC markets.
"""

import csv
import io
import logging
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from ml.models.elo_sports import MatchResult

logger = logging.getLogger(__name__)


def _parse_date(val: str) -> Optional[date]:
    """Parse date from various formats."""
    if not val or not str(val).strip():
        return None
    val = str(val).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(val[:10], fmt)
            return dt.date()
        except (ValueError, IndexError):
            continue
    return None


def _normalize_fighter_name(name: str) -> str:
    """Clean fighter name for consistency."""
    if not name:
        return ""
    name = str(name).strip()
    name = name.replace("  ", " ")
    return name


# ── Kagglehub Loader (primary) ──────────────────────────────────────

def load_ufc_from_kagglehub() -> list[MatchResult]:
    """Load UFC fights from Kagglehub (m0hamedai1/the-ultimate-ufc-archive-1993-present).

    Merges fights.csv + events.csv for dates. ~8.4k fights from 1993-present.
    """
    try:
        import kagglehub
        import pandas as pd
    except ImportError as e:
        logger.error(f"pip install kagglehub pandas: {e}")
        return []

    try:
        base = Path(kagglehub.dataset_download("m0hamedai1/the-ultimate-ufc-archive-1993-present"))
    except Exception as e:
        logger.error(f"Kagglehub download failed: {e}")
        return []

    fights_path = base / "fights.csv"
    events_path = base / "events.csv"
    if not fights_path.exists():
        logger.error(f"fights.csv not found at {base}")
        return []

    fights = pd.read_csv(fights_path)
    events = pd.read_csv(events_path)[["event_id", "date"]] if events_path.exists() else None
    if events is not None:
        fights = fights.merge(events, on="event_id", how="left")

    matches = []
    for _, row in fights.iterrows():
        try:
            left = _normalize_fighter_name(str(row.get("left_fighter_name", "")))
            right = _normalize_fighter_name(str(row.get("right_fighter_name", "")))
            winner_val = str(row.get("winner", "")).strip().upper()
            winner_name = _normalize_fighter_name(str(row.get("winner_name", "")))

            if not left or not right:
                continue
            if winner_val in ("D", "DRAW") or (pd.isna(row.get("winner")) and not winner_name):
                continue

            if winner_val == "L":
                winner_name_final, loser_name = left, right
            elif winner_val == "R":
                winner_name_final, loser_name = right, left
            elif winner_name:
                if winner_name.upper() == left.upper() or winner_name in left:
                    winner_name_final, loser_name = left, right
                elif winner_name.upper() == right.upper() or winner_name in right:
                    winner_name_final, loser_name = right, left
                else:
                    continue
            else:
                continue

            dt = _parse_date(str(row.get("date", "2000-01-01"))) or date(2000, 1, 1)
            matches.append(MatchResult(
                winner=winner_name_final,
                loser=loser_name,
                match_date=dt,
                surface="cage",
                tourney_level="ufc",
            ))
        except Exception:
            continue

    matches.sort(key=lambda m: m.match_date)
    logger.info(f"Loaded {len(matches)} UFC fights from Kagglehub")
    return matches


# ── CSV Loader ──────────────────────────────────────────────────────

def load_ufc_from_csv(path: str | Path) -> list[MatchResult]:
    """Load UFC fights from a local CSV file.

    Supports common Kaggle UFC dataset formats.
    """
    path = Path(path)
    if not path.exists():
        logger.error(f"UFC CSV not found: {path}")
        return []

    matches = []
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            return []

        def find_col(*patterns: str) -> str | None:
            for fn in fieldnames:
                fn_lower = fn.lower()
                for p in patterns:
                    if p in fn_lower:
                        return fn
            return None

        red_col = find_col("r_fighter", "fighter1", "red_fighter", "left_fighter")
        blue_col = find_col("b_fighter", "fighter2", "blue_fighter", "right_fighter")
        winner_col = find_col("winner_name") or find_col("winner") or find_col("event_winner")
        date_col = find_col("date", "event_date")

        for row in reader:
            try:
                r_name = _normalize_fighter_name(row.get(red_col or "", "")) if red_col else ""
                b_name = _normalize_fighter_name(row.get(blue_col or "", "")) if blue_col else ""
                winner = _normalize_fighter_name(row.get(winner_col or "", "")) if winner_col else ""
                dt = _parse_date(row.get(date_col or "", "")) if date_col else None

                if not r_name or not b_name or not winner:
                    continue

                winner_upper = winner.upper()
                if winner_upper in ("DRAW", "NC", "NO CONTEST", "CANCELED", ""):
                    continue

                if winner_upper in ("RED", "R", "LEFT", "L"):
                    winner_name, loser_name = r_name, b_name
                elif winner_upper in ("BLUE", "B", "RIGHT"):
                    winner_name, loser_name = b_name, r_name
                elif winner.upper() == r_name.upper() or winner in r_name or r_name in winner:
                    winner_name, loser_name = r_name, b_name
                elif winner.upper() == b_name.upper() or winner in b_name or b_name in winner:
                    winner_name, loser_name = b_name, r_name
                else:
                    continue

                match_date = dt or date(2000, 1, 1)
                matches.append(MatchResult(
                    winner=winner_name,
                    loser=loser_name,
                    match_date=match_date,
                    surface="cage",
                    tourney_level="ufc",
                ))
            except Exception:
                continue

    matches.sort(key=lambda m: m.match_date)
    logger.info(f"Loaded {len(matches)} UFC fights from {path}")
    return matches


# ── Hugging Face Loader (backup) ────────────────────────────────────

async def fetch_ufc_from_huggingface() -> list[MatchResult]:
    """Fetch UFC fights from Hugging Face datasets (limited ~200 fights)."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("pip install datasets to use Hugging Face UFC data")
        return []

    try:
        ds = load_dataset("xtinkarpiu/ufc-fight-data", split="train", trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Hugging Face UFC load failed: {e}")
        return []

    matches = []
    for row in ds:
        try:
            f1 = str(row.get("fighter1_name", row.get("R_fighter", ""))).strip()
            f2 = str(row.get("fighter2_name", row.get("B_fighter", ""))).strip()
            winner = str(row.get("event_winner", row.get("Winner", ""))).strip()

            if not f1 or not f2 or not winner:
                continue
            if winner.upper() in ("DRAW", "NC", "NO CONTEST"):
                continue

            dt_str = str(row.get("event_date", row.get("date", "2000-01-01")))[:10]
            match_date = _parse_date(dt_str) or date(2000, 1, 1)

            if winner.upper() == f1.upper() or winner in f1:
                winner_name, loser_name = f1, f2
            elif winner.upper() == f2.upper() or winner in f2:
                winner_name, loser_name = f2, f1
            else:
                continue

            matches.append(MatchResult(
                winner=_normalize_fighter_name(winner_name),
                loser=_normalize_fighter_name(loser_name),
                match_date=match_date,
                surface="cage",
                tourney_level="ufc",
            ))
        except Exception:
            continue

    matches.sort(key=lambda m: m.match_date)
    logger.info(f"Loaded {len(matches)} UFC fights from Hugging Face")
    return matches


# ── Main fetch function ─────────────────────────────────────────────

async def fetch_all_ufc_matches(
    csv_path: str | Path | None = None,
    use_kagglehub: bool = True,
    use_huggingface: bool = False,
) -> list[MatchResult]:
    """Fetch all UFC matches from configured sources.

    Priority: csv_path > UFC_CSV_PATH env > Kagglehub > Hugging Face.
    """
    path = csv_path or os.environ.get("UFC_CSV_PATH")
    if path:
        return load_ufc_from_csv(path)

    if use_kagglehub:
        try:
            matches = load_ufc_from_kagglehub()
            if matches:
                return matches
        except Exception as e:
            logger.warning(f"Kagglehub UFC load failed: {e}")

    if use_huggingface:
        return await fetch_ufc_from_huggingface()

    logger.warning("No UFC data. Install kagglehub and run: load_ufc_from_kagglehub()")
    return []


# ── Market Detection & Parsing (for live Polymarket UFC markets) ─────

UFC_INDICATORS = [
    "ufc", "mma", "mixed martial arts", "ultimate fighting",
    "bellator", "pfl", "cage fight", "octagon",
    "flyweight", "bantamweight", "featherweight", "lightweight",
    "welterweight", "middleweight", "light heavyweight", "heavyweight",
    "strawweight",
]


def detect_ufc_market(question: str, category: str = "") -> bool:
    """Check if a market is about UFC/MMA."""
    combined = f"{question} {category}".lower()
    return any(indicator in combined for indicator in UFC_INDICATORS)


def parse_ufc_matchup(
    question: str,
    category: str = "",
    market_id: int | None = None,
) -> Optional[dict]:
    """Try to parse a UFC/MMA matchup from a market question.

    Returns dict with fighter_a, fighter_b, yes_side_fighter, or None.
    """
    if not detect_ufc_market(question, category):
        return None

    vs_patterns = [
        re.compile(r'(.+?)\s+vs?\.?\s+(.+?)(?:\s*[:\-\|]|\s*$)', re.IGNORECASE),
        re.compile(r'(.+?)\s+versus\s+(.+?)(?:\s*[:\-\|]|\s*$)', re.IGNORECASE),
        re.compile(r'will\s+(.+?)\s+beat\s+(.+?)[\?\.]', re.IGNORECASE),
    ]

    prop_patterns = [
        re.compile(r'o/u\s*[\d.]', re.IGNORECASE),
        re.compile(r'over/under', re.IGNORECASE),
        re.compile(r'method\s+of\s+victory', re.IGNORECASE),
        re.compile(r'round\s+\d+', re.IGNORECASE),
        re.compile(r'total\s+rounds', re.IGNORECASE),
    ]

    if any(p.search(question) for p in prop_patterns):
        return None

    fighter_a = None
    fighter_b = None
    for pattern in vs_patterns:
        match = pattern.search(question)
        if match:
            fighter_a = _clean_fighter_name(match.group(1))
            fighter_b = _clean_fighter_name(match.group(2))
            break

    if not fighter_a or not fighter_b or len(fighter_a) < 3 or len(fighter_b) < 3:
        return None

    yes_side = fighter_a
    beat_match = re.search(r'will\s+(.+?)\s+beat', question, re.IGNORECASE)
    if beat_match:
        first = beat_match.group(1).strip().lower()
        if first in fighter_a.lower():
            yes_side = fighter_a
        elif first in fighter_b.lower():
            yes_side = fighter_b

    return {
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "sport": "mma",
        "yes_side_fighter": yes_side,
        "market_id": market_id,
    }


def _clean_fighter_name(name: str) -> str:
    """Clean extracted fighter name."""
    name = name.strip()
    name = re.sub(r'\s*\(.*?\)\s*', '', name)
    name = re.sub(r'\s*\[.*?\]\s*', '', name)
    name = re.sub(r'^\d+\.\s*', '', name)
    # Strip leading event prefixes like "UFC 300:" or "Bellator 299 -"
    name = re.sub(r'^(?:UFC|Bellator|PFL|ONE)\s*\d*\s*[:\-]\s*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+', ' ', name)
    name = name.rstrip('?.:!,')
    return name.strip()
