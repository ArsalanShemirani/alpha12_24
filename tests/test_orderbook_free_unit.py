import json
import responses
from src.data.orderbook_free import ob_features

@responses.activate
def test_ob_features_from_mocked_depth():
    # Mock Binance spot orderbook (top 20) with heavier bids than asks
    depth = {
        "lastUpdateId": 1,
        "bids": [[ "100.00", "5.0" ]] * 20,
        "asks": [[ "100.02", "1.0" ]] * 20,
    }
    responses.add(
        responses.GET,
        "https://api.binance.com/api/v3/depth",
        json=depth, status=200
    )
    feats = ob_features("BTCUSDT", top=20)
    assert "ob_imb_top20" in feats
    assert feats["ob_imb_top20"] > 0.0  # bid-heavy
    assert feats["ob_spread_w"] >= 0.0
    assert feats["ob_bidv_top20"] > feats["ob_askv_top20"]