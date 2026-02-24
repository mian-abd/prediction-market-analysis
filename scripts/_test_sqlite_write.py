"""Quick test to debug why _upsert_markets_sync returns 0."""
import sys, asyncio, sqlite3
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


async def main():
    from data_pipeline.collectors.kalshi_resolved import fetch_all_resolved_markets, parse_resolved_market
    from config.settings import settings

    db_path = settings.database_url.replace("sqlite+aiosqlite:///", "").replace("sqlite:///", "")
    db_path = str(Path(db_path).resolve())
    print("DB:", db_path)

    raw = await fetch_all_resolved_markets(max_markets=10)
    markets = [parse_resolved_market(m) for m in raw]
    valid = [m for m in markets if m.get("resolution_value") is not None]
    print(f"Got {len(valid)} valid markets")
    if not valid:
        print("No valid markets!")
        return

    m = valid[0]
    print("Test market keys:", list(m.keys()))
    print("external_id:", m.get("external_id"))
    print("resolution_value:", m.get("resolution_value"))

    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    
    platform_id = 2  # kalshi
    now = datetime.utcnow().isoformat()

    try:
        conn.execute("BEGIN")
    except sqlite3.OperationalError as e:
        print(f"BEGIN failed: {e}, trying rollback+begin")
        conn.execute("ROLLBACK")
        conn.execute("BEGIN")

    # Try inserting just the first market
    try:
        from data_pipeline.category_normalizer import normalize_category
        norm_cat = normalize_category(m.get("category",""), m.get("question",""), m.get("description","")) or "other"
    except Exception:
        norm_cat = (m.get("category","") or "other").lower()

    def _dt(val):
        if val is None: return None
        if hasattr(val, "isoformat"):
            if hasattr(val, "tzinfo") and val.tzinfo:
                val = val.replace(tzinfo=None)
            return val.isoformat()
        return str(val)

    params = (
        platform_id, m["external_id"], m.get("condition_id"), m.get("token_id_yes"), m.get("token_id_no"),
        m["question"], m.get("description",""), m.get("category",""), norm_cat, m.get("slug",""),
        m.get("price_yes"), m.get("price_no"), m.get("volume_24h",0), m.get("volume_total",0),
        m.get("liquidity",0), m.get("open_interest",0),
        1 if m.get("is_active") else 0, 1 if m.get("is_resolved") else 0,
        m.get("resolution_outcome"), m.get("resolution_value"),
        _dt(m.get("end_date")), _dt(m.get("resolved_at")), now, now, now,
        m.get("taker_fee_bps",0), m.get("maker_fee_bps",0), 1 if m.get("is_neg_risk") else 0,
    )
    print(f"Param count: {len(params)}")

    try:
        conn.execute("""INSERT OR REPLACE INTO markets (
            platform_id, external_id, condition_id, token_id_yes, token_id_no,
            question, description, category, normalized_category, slug,
            price_yes, price_no, volume_24h, volume_total, liquidity, open_interest,
            is_active, is_resolved, resolution_outcome, resolution_value,
            end_date, resolved_at, created_at, updated_at, last_fetched_at,
            taker_fee_bps, maker_fee_bps, is_neg_risk
        ) VALUES (?,?,?,?,?, ?,?,?,?,?, ?,?,?,?,?,?, ?,?,?,?, ?,?,?,?,?, ?,?,?)""", params)
        conn.execute("COMMIT")
        print("INSERT succeeded!")
    except Exception as e:
        print(f"INSERT FAILED: {type(e).__name__}: {e}")

    conn.close()

asyncio.run(main())
