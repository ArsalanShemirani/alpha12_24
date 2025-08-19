# --- FILE: src/data/deribit_rr25.py ---
import os, time, json, math
from math import log, sqrt, erf
import requests
import pandas as pd
from datetime import datetime, timezone

DERIBIT = "https://www.deribit.com/api/v2"
HOSTS = [
    "https://www.deribit.com/api/v2",
    "https://deribit.com/api/v2",
]
RUNS_DIR = os.getenv("RUNS_DIR", "runs")

_sess = requests.Session()
_sess.headers.update({
    "User-Agent": "alpha12_24/deribit (contact: ops@alpha12_24.local)",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
})

def _get(path, params=None, timeout=10, tries=6, backoff=0.5):
    """
    GET with retries across multiple DERIBIT hosts.
    Falls back to variants of params (e.g., dropping 'expired') on HTTP 400.
    """
    params = dict(params or {})
    last_err = None
    for attempt in range(1, tries + 1):
        for base in HOSTS:
            url = f"{base}{path}"
            try:
                r = _sess.get(url, params=params, timeout=timeout)
                # Retry on typical transient statuses
                if r.status_code in (429, 502, 503, 504):
                    last_err = requests.HTTPError(f"{r.status_code} {r.reason}")
                    time.sleep(backoff * attempt)
                    continue
                r.raise_for_status()
                js = r.json()
                if "result" not in js:
                    raise RuntimeError(f"Deribit response missing result: {js}")
                return js["result"]
            except requests.HTTPError as e:
                # If we get a 400 on get_instruments with 'expired' key, try without it.
                if r is not None and r.status_code == 400 and path == "/public/get_instruments" and "expired" in params:
                    try_params = dict(params)
                    try_params.pop("expired", None)
                    try:
                        r2 = _sess.get(url, params=try_params, timeout=timeout)
                        if r2.status_code in (429, 502, 503, 504):
                            last_err = requests.HTTPError(f"{r2.status_code} {r2.reason}")
                            time.sleep(backoff * attempt)
                            continue
                        r2.raise_for_status()
                        js = r2.json()
                        if "result" not in js:
                            raise RuntimeError(f"Deribit response missing result: {js}")
                        return js["result"]
                    except Exception as e2:
                        last_err = e2
                        time.sleep(backoff * attempt)
                        continue
                last_err = e
                time.sleep(backoff * attempt)
                continue
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                time.sleep(backoff * attempt)
                continue
            except Exception as e:
                last_err = e
                time.sleep(backoff * attempt)
                continue
    raise RuntimeError(f"Deribit GET failed for {path} params={params}: {last_err}")

def _cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))

def _delta_bs(spot: float, strike: float, t_years: float, iv: float, is_call: bool) -> float:
    # r=0 Black–Scholes delta
    if spot <= 0 or strike <= 0 or t_years <= 0 or iv <= 0:
        return float("nan")
    d1 = (log(spot / strike) + 0.5 * iv * iv * t_years) / (iv * sqrt(t_years))
    return _cdf(d1) if is_call else (_cdf(d1) - 1.0)

def _index_name(cur: str) -> str:
    cur = cur.upper()
    return "btc_usd" if cur == "BTC" else ("eth_usd" if cur == "ETH" else "btc_usd")

def get_index_price(currency="BTC") -> float:
    # Try standard lowercase index names; if it fails, try uppercase fallback.
    name = _index_name(currency)
    try:
        idx = _get("/public/get_index_price", {"index_name": name})
        return float(idx.get("index_price", float("nan")))
    except Exception:
        alt = name.upper()
        idx = _get("/public/get_index_price", {"index_name": alt})
        return float(idx.get("index_price", float("nan")))

def _norm_iv(iv_raw):
    try:
        iv = float(iv_raw)
        if iv > 3.0:  # percent given as 55 -> 0.55
            iv /= 100.0
        return iv
    except Exception:
        return float("nan")

def get_instruments(currency="BTC"):
    """
    Cache instrument list for 30 minutes. Deribit sometimes rejects boolean query serialization;
    we try with and without 'expired' key if needed.
    """
    os.makedirs(RUNS_DIR, exist_ok=True)
    cur = currency.upper()
    cache = os.path.join(RUNS_DIR, f"deribit_instruments_{cur}.json")
    if os.path.exists(cache) and (time.time() - os.path.getmtime(cache) < 1800):
        try:
            return json.loads(open(cache).read())
        except Exception:
            pass
    params = {"currency": cur, "kind": "option", "expired": False}
    try:
        res = _get("/public/get_instruments", params)
    except Exception:
        # Fallback: without 'expired'
        res = _get("/public/get_instruments", {"currency": cur, "kind": "option"})
    with open(cache, "w") as f:
        json.dump(res, f)
    return res

def ticker_mark_iv(name: str) -> float:
    t = _get("/public/ticker", {"instrument_name": name})
    return _norm_iv(t.get("mark_iv", float("nan")))

def parse_instrument(name: str):
    # BTC-27SEP24-60000-C
    parts = name.split("-")
    if len(parts) != 4: 
        return None
    cur = parts[0]
    strike = float(parts[2])
    otype = parts[3]  # 'C' or 'P'
    return cur, strike, otype

def _nearest_expiries(instruments, n=3):
    # Build DF: expiry_ts, name, strike, otype
    rows = []
    for ins in instruments:
        name = ins.get("instrument_name") or ins.get("name")
        exp_ts = ins.get("expiration_timestamp")
        if not name or not exp_ts:
            continue
        parsed = parse_instrument(name)
        if not parsed:
            continue
        _, strike, otype = parsed
        rows.append((int(exp_ts), name, float(strike), otype))
    df = pd.DataFrame(rows, columns=["exp_ts","name","strike","otype"])
    if df.empty:
        return [], pd.DataFrame()
    df = df.sort_values("exp_ts")
    expiries = list(dict.fromkeys(df["exp_ts"].tolist()))
    return expiries[:n], df

def _pick_candidates(df_exp, underlying, k_each_side=18, moneyness_lo=0.65, moneyness_hi=1.35):
    d = df_exp.copy()
    base = max(float(underlying), 1e-6)
    d["moneyness"] = d["strike"] / base
    d = d[(d["moneyness"] > moneyness_lo) & (d["moneyness"] < moneyness_hi)]
    calls = d[d["otype"]=="C"].sort_values("strike")
    puts  = d[d["otype"]=="P"].sort_values("strike")
    calls_lo = calls[calls["strike"] <= underlying].tail(k_each_side)
    calls_hi = calls[calls["strike"] >  underlying].head(k_each_side)
    puts_lo  = puts[puts["strike"] <  underlying].tail(k_each_side)
    puts_hi  = puts[puts["strike"] >= underlying].head(k_each_side)
    out = pd.concat([calls_lo, calls_hi, puts_lo, puts_hi]).drop_duplicates()
    return out

def snapshot_rr25(currency="BTC", n_expiries=3, k_each_side=18):
    """
    Returns DF:
      ['currency','expiry_ts','inst_call','inst_put','iv_call25','iv_put25','rr25','underlying','updated_at']
    """
    S = get_index_price(currency)
    if not (S and S == S and S > 0):
        return pd.DataFrame()

    instruments = get_instruments(currency)  # cached
    expiries, df_all = _nearest_expiries(instruments, n=n_expiries)
    if not expiries or df_all.empty:
        return pd.DataFrame()

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    rows = []

    for exp_ts in expiries:
        T_years = max((exp_ts - now_ms) / 1000.0 / (365.0 * 24.0 * 3600.0), 1e-6)
        df_exp = df_all[df_all["exp_ts"] == exp_ts]
        cand = _pick_candidates(df_exp, S, k_each_side=k_each_side)
        if cand.empty:
            continue

        best_call = (None, 1.0, float("nan"))
        best_put  = (None, 1.0, float("nan"))

        # Throttle ticker calls gently (public RL). Query only unique names.
        names = cand["name"].drop_duplicates().tolist()
        for name in names:
            iv = ticker_mark_iv(name)
            if not (iv and iv == iv and iv > 0):
                time.sleep(0.08)
                continue
            cur, K, otype = parse_instrument(name)
            d = _delta_bs(S, K, T_years, iv, is_call=(otype=="C"))
            if d != d:
                time.sleep(0.08)
                continue
            if otype == "C":
                dist = abs(d - 0.25)
                if dist < best_call[1]:
                    best_call = (name, dist, iv)
            else:
                dist = abs(d + 0.25)
                if dist < best_put[1]:
                    best_put = (name, dist, iv)
            # be kind to RL
            time.sleep(0.08)

        if best_call[0] and best_put[0]:
            ivc = best_call[2]
            ivp = best_put[2]
            rr = ivc - ivp
            rows.append({
                "currency": currency,
                "expiry_ts": int(exp_ts),
                "inst_call": best_call[0],
                "inst_put": best_put[0],
                "iv_call25": ivc,
                "iv_put25": ivp,
                "rr25": rr,
                "underlying": S,
                "updated_at": datetime.now(timezone.utc).isoformat()
            })

    return pd.DataFrame(rows)

def write_latest(currency="BTC", runs_dir=RUNS_DIR):
    os.makedirs(runs_dir, exist_ok=True)
    df = snapshot_rr25(currency=currency)
    latest_json = os.path.join(runs_dir, f"deribit_rr25_latest_{currency}.json")
    hist_csv    = os.path.join(runs_dir, f"deribit_rr25_{currency}.csv")
    with open(latest_json, "w") as f:
        json.dump({"currency": currency, "rows": df.to_dict(orient="records")}, f, indent=2)
    if not df.empty:
        if os.path.exists(hist_csv):
            df.to_csv(hist_csv, mode="a", header=False, index=False)
        else:
            df.to_csv(hist_csv, index=False)
        return True
    return False

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--currency", default="BTC", choices=["BTC","ETH"])
    p.add_argument("--interval_sec", type=int, default=60)
    args = p.parse_args()
    while True:
        write_latest(currency=args.currency)
        time.sleep(max(30, args.interval_sec))
# --- END FILE ---