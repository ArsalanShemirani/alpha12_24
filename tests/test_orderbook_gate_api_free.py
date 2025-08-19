from src.data.orderbook_free import ob_features

def test_ob_features_schema_and_range():
    out = ob_features("BTCUSDT", top=20)
    assert set(out.keys()) >= {"ob_imb_top20","ob_spread_w","ob_bidv_top20","ob_askv_top20"}
    assert -1.0 <= out["ob_imb_top20"] <= 1.0
    assert out["ob_bidv_top20"] >= 0 and out["ob_askv_top20"] >= 0
