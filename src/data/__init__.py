# Data providers and sources
# Safe re-exports; avoid hard failures if a backend is absent.

try:
    from .binance_free import assemble as load_binance_spot
except Exception:
    load_binance_spot = None

try:
    from .bybit_free import assemble as load_bybit_derivs
except Exception:
    load_bybit_derivs = None

try:
    from .composite import assemble_spot_plus_bybit as load_composite
except Exception:
    load_composite = None

__all__ = [
    "load_binance_spot", "load_bybit_derivs", "load_composite"
]
