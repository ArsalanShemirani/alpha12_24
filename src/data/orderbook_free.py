import requests
import numpy as np

_sess = requests.Session()
_sess.headers.update({"User-Agent": "alpha12_24/ob"})

def fetch_depth(symbol: str = "BTCUSDT", limit: int = 100, timeout: int = 6):
    """
    Return (bids, asks) as float arrays [[price, qty], ...] from Binance Spot public REST.
    No API key required.
    """
    r = _sess.get(
        "https://api.binance.com/api/v3/depth",
        params={"symbol": symbol, "limit": limit},
        timeout=timeout,
    )
    r.raise_for_status()
    js = r.json()
    bids = np.array([[float(p), float(q)] for p, q in js.get("bids", [])], dtype=float)
    asks = np.array([[float(p), float(q)] for p, q in js.get("asks", [])], dtype=float)
    return bids, asks

def ob_features(symbol: str = "BTCUSDT", top: int = 20):
    """
    Lightweight L2 features:
      - ob_imb_top20: (sum bid_qty - sum ask_qty) / (sum bid_qty + sum ask_qty)
      - ob_spread_w: weighted mid spread using price*qty weights
      - ob_bidv_top20 / ob_askv_top20: total top-N size on each side
    Always returns numeric values; falls back to 0.0 when unavailable.
    """
    try:
        bids, asks = fetch_depth(symbol, limit=max(50, top))
        if bids.size == 0 or asks.size == 0:
            raise ValueError("Empty depth data")

        bidv = float(bids[:top, 1].sum())
        askv = float(asks[:top, 1].sum())
        imb = (bidv - askv) / max(bidv + askv, 1e-9)

        wbid = float((bids[:top, 0] * bids[:top, 1]).sum()) / max(bidv, 1e-9)
        wask = float((asks[:top, 0] * asks[:top, 1]).sum()) / max(askv, 1e-9)
        spread_w = (wask - wbid) / max((wask + wbid) / 2.0, 1e-9)

        return {
            "ob_imb_top20": imb,
            "ob_spread_w": spread_w,
            "ob_bidv_top20": bidv,
            "ob_askv_top20": askv,
        }
    except Exception:
        # Always return numeric defaults instead of empty dict
        return {
            "ob_imb_top20": 0.0,
            "ob_spread_w": 0.0,
            "ob_bidv_top20": 0.0,
            "ob_askv_top20": 0.0,
        }