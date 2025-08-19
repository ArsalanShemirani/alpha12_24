from src.dashboard import app as appmod

def test_bars_per_day_and_limit():
    assert appmod._bars_per_day("5m") == 288
    assert appmod._bars_per_day("1h") == 24
    # cap logic
    assert appmod._calc_limit("5m", 10) == min(288*10, 1500)
    assert appmod._calc_limit("4h", 90) == min(6*90, 2500)