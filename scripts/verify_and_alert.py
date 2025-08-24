#!/usr/bin/env python3
import os, sys, math, time
from datetime import datetime
from pathlib import Path

# Optional: read .env if present (no hard dep)
def load_dotenv(dotenv_path=".env"):
    if not Path(dotenv_path).exists(): return
    for line in Path(dotenv_path).read_text().splitlines():
        line=line.strip()
        if not line or line.startswith("#"): continue
        if "=" in line:
            k,v=line.split("=",1)
            os.environ.setdefault(k.strip(), v.strip())

load_dotenv()

RUNS_DIR = os.getenv("RUNS_DIR", "runs")
INV_CSV  = os.getenv("RR_INVARIANT_SIDECAR_PATH", f"{RUNS_DIR}/rr_invariants.csv")
TOL_RR   = float(os.getenv("RR_TOL", "0.02"))  # 2%
TOKEN    = os.getenv("TG_BOT_TOKEN")
CHAT_ID  = os.getenv("TG_CHAT_ID")

def send_telegram(text):
    import json, urllib.request, urllib.parse
    if not TOKEN or not CHAT_ID:
        print("[warn] Telegram env vars missing; skipping alert.")
        return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }).encode("utf-8")
    try:
        with urllib.request.urlopen(url, data=data, timeout=20) as r:
            _ = r.read()
    except Exception as e:
        print(f"[error] Telegram send failed: {e}", file=sys.stderr)

def fmt_pct(x):
    try:
        return f"{100.0*float(x):.1f}%"
    except:
        return "n/a"

def main():
    import csv
    p = Path(INV_CSV)
    if not p.exists():
        msg = f"⚠️ R:R invariant audit: sidecar not found at `{INV_CSV}`."
        print(msg)
        send_telegram(msg)
        sys.exit(0)

    # Read & filter fills only
    rows = []
    with p.open() as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                if row.get("warn_missing_fill","1") == "0":
                    rows.append(row)
            except: 
                pass

    fills = len(rows)
    if fills == 0:
        msg = f"⚠️ R:R invariant audit: no fills found in `{INV_CSV}`."
        print(msg)
        send_telegram(msg)
        sys.exit(0)

    ok = bad = 0
    flagged = []
    def to_float(x):
        try: return float(x)
        except: return float("nan")

    for row in rows:
        d = to_float(row.get("rr_distortion"))
        rr_plan = to_float(row.get("rr_planned"))
        # If rr_distortion missing, recompute from prices (fallback)
        if not math.isfinite(d):
            rr_real_prices = to_float(row.get("rr_realized_from_prices"))
            if math.isfinite(rr_plan) and rr_plan>0 and math.isfinite(rr_real_prices):
                d = abs(rr_real_prices - rr_plan)/rr_plan
        if math.isfinite(d):
            if d <= TOL_RR:
                ok += 1
            else:
                bad += 1
                flagged.append((
                    row.get("setup_id","?"),
                    row.get("tf","?"),
                    row.get("direction","?"),
                    rr_plan,
                    to_float(row.get("rr_realized_from_prices")) if row.get("rr_realized_from_prices") else to_float(row.get("rr_realized_from_fill")),
                    d
                ))

    pct_ok = (ok / (ok+bad)) if (ok+bad)>0 else 0.0

    # Build summary
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"✅ *R:R Invariance Daily Check* — {now} MYT"
    body = (
        f"*Fills analyzed:* {fills}\n"
        f"*Within tolerance (≤ {fmt_pct(TOL_RR)}):* {ok} ({fmt_pct(pct_ok)})\n"
        f"*Flagged:* {bad}"
    )

    # Include up to 5 worst offenders (if any)
    tail = ""
    if bad > 0:
        flagged.sort(key=lambda x: (x[5] if math.isfinite(x[5]) else 0.0), reverse=True)
        tail = "\n*Top flagged (setup_id tf dir | rr_plan → rr_real | dist):*\n" + "\n".join(
            f"`{sid}` {tf} {dr} | {rrp:.3f} → {rrr:.3f} | {fmt_pct(dist)}"
            for sid, tf, dr, rrp, rrr, dist in flagged[:5]
            if math.isfinite(rrp) and math.isfinite(rrr) and math.isfinite(dist)
        )

    text = f"{header}\n{body}{tail}"
    print(text)
    send_telegram(text)

if __name__ == "__main__":
    main()