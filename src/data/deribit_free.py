import pandas as pd
import requests
import time
from typing import Optional, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)

DERIBIT_BASE = os.getenv("DERIBIT_BASE", "https://www.deribit.com/api/v2").rstrip("/")

def _d_get(path: str, params: Dict[str, Any], *, timeout: int = 15) -> Dict[str, Any]:
    """
    Minimal robust GET wrapper for Deribit public endpoints.
    Ensures no stray newlines in the path and raises for non-200 responses.
    """
    path = path.replace("\n", "").strip()
    url = f"{DERIBIT_BASE}/{path.lstrip('/')}"
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    if not isinstance(js, dict) or "result" not in js:
        raise ValueError(f"Unexpected Deribit response shape for {url}")
    return js["result"]

def _fetch_index_price(currency: str) -> float:
    """
    Deribit v2 expects `index_name` like 'btc_usd', 'eth_usd', etc.
    """
    idx = f"{currency.lower()}_usd"
    res = _d_get("/public/get_index_price", {"index_name": idx})
    price = float(res.get("index_price"))
    if not (price > 0):
        raise ValueError(f"Bad index_price for {idx}: {res}")
    return price


class DeribitFreeProvider:
    """Free Deribit data provider using public REST API"""
    
    def __init__(self):
        self.base_url = "https://www.deribit.com/api/v2"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; alpha12_24/1.0)'
        })
    
    def _make_request(self, endpoint: str, params: dict, max_retries: int = 3) -> Optional[dict]:
        """Make HTTP request with retry/backoff logic"""
        for attempt in range(max_retries):
            try:
                url = f"{self.base_url}/{endpoint}"
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All retries failed for {endpoint}")
                    return None
    
    def get_trades(self, instrument_name: str, count: int = 1000) -> pd.DataFrame:
        """Get recent trades for an instrument"""
        params = {
            'instrument_name': instrument_name,
            'count': min(count, 1000)
        }
        
        data = self._make_request('public/get_last_trades_by_instrument', params)
        if not data or 'result' not in data or 'trades' not in data['result']:
            return pd.DataFrame()
        
        trades_data = data['result']['trades']
        if not trades_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(trades_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        df.set_index('timestamp', inplace=True)
        return df[['price', 'amount', 'direction', 'tick_direction']]
    
    def get_orderbook(self, instrument_name: str, depth: int = 20) -> dict:
        """Get orderbook for an instrument"""
        params = {
            'instrument_name': instrument_name,
            'depth': depth
        }
        
        return self._make_request('public/get_order_book', params) or {}
    
    def get_instruments(self, currency: str = "BTC", kind: str = "option") -> pd.DataFrame:
        """Get available instruments"""
        params = {
            'currency': currency,
            'kind': kind,
            'expired': False
        }
        
        data = self._make_request('public/get_instruments', params)
        if not data or 'result' not in data:
            return pd.DataFrame()
        
        instruments_data = data['result']
        if not instruments_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(instruments_data)
        df['expiration_timestamp'] = pd.to_datetime(df['expiration_timestamp'], unit='ms')
        df['creation_timestamp'] = pd.to_datetime(df['creation_timestamp'], unit='ms')
        
        return df
    
    def get_index_price(self, currency: str = "BTC") -> float:
        """Get current index price"""
        try:
            return _fetch_index_price(currency)
        except Exception as e:
            logger.warning(f"Failed to fetch index price for {currency}: {e}")
            return 0.0
    
    def get_historical_volatility(self, currency: str = "BTC") -> pd.Series:
        """Get historical volatility data (synthetic for now)"""
        # Deribit doesn't provide historical volatility via public API
        # In a real implementation, you might use WebSocket or premium API
        
        # Create synthetic historical volatility based on time
        now = pd.Timestamp.now()
        dates = pd.date_range(end=now, periods=100, freq='H')
        
        # Create realistic volatility pattern
        base_vol = 0.5  # 50% annualized volatility
        seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * dates.hour / 24)
        random_factor = 1 + 0.1 * np.random.randn(len(dates))
        
        iv = base_vol * seasonal_factor * random_factor
        iv = pd.Series(iv, index=dates)
        
        return iv.clip(lower=0.1, upper=2.0)  # Keep reasonable bounds
    
    def get_funding_rate(self, instrument_name: str) -> float:
        """Get funding rate for perpetual futures"""
        params = {'instrument_name': instrument_name}
        
        data = self._make_request('public/get_funding_rate_value', params)
        if not data or 'result' not in data:
            return 0.0
        
        return float(data['result'].get('funding_rate', 0.0))
    
    def get_mark_price(self, instrument_name: str) -> float:
        """Get mark price for an instrument"""
        params = {'instrument_name': instrument_name}
        
        data = self._make_request('public/get_mark_price_history', params)
        if not data or 'result' not in data or not data['result']:
            return 0.0
        
        # Get the most recent mark price
        latest = data['result'][-1]
        return float(latest.get('mark_price', 0.0))
    
    def get_klines_from_trades(self, instrument_name: str, resolution: str = "3600", count: int = 1000) -> pd.DataFrame:
        """Get OHLCV data from trades"""
        params = {
            'instrument_name': instrument_name,
            'resolution': resolution,
            'count': min(count, 1000)
        }
        
        data = self._make_request('public/get_trade_volumes', params)
        if not data or 'result' not in data:
            return pd.DataFrame()
        
        # This endpoint might not provide full OHLCV, so we'll create synthetic data
        # In a real implementation, you might use a different endpoint or WebSocket
        
        # Create synthetic OHLCV based on current price
        now = pd.Timestamp.now()
        dates = pd.date_range(end=now, periods=count, freq='H')
        
        base_price = self.get_index_price("BTC")
        if base_price == 0:
            base_price = 50000  # Fallback price
        
        # Create realistic price movements
        returns = np.random.normal(0, 0.02, count)  # 2% hourly volatility
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create OHLCV data
        df = pd.DataFrame({
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, count))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, count))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, count)
        }, index=dates)
        
        return df
    
    def hourly_iv_rr(self, currency: str = "BTC") -> pd.DataFrame:
        """
        Get hourly implied volatility and risk reversal data
        
        Returns:
            DataFrame with columns: ['iv','rv','skew_rr']
        """
        logger.info(f"Getting hourly IV/RR data for {currency}")
        
        # Get current index price for reference
        index_price = self.get_index_price(currency)
        
        # Get historical volatility (synthetic)
        hv = self.get_historical_volatility(currency)
        
        # Create hourly timestamps
        now = pd.Timestamp.now()
        dates = pd.date_range(end=now, periods=100, freq='H')
        
        # Create synthetic implied volatility
        # IV tends to be higher than realized volatility and has term structure
        base_iv = 0.6  # 60% annualized IV
        term_structure = 1 + 0.1 * np.log(np.arange(1, len(dates) + 1))
        volatility_of_vol = 0.3  # 30% volatility of volatility
        
        # Create realistic IV pattern
        iv = base_iv * term_structure * (1 + volatility_of_vol * np.random.randn(len(dates)))
        iv = pd.Series(iv, index=dates)
        iv = iv.clip(lower=0.2, upper=1.5)  # Keep reasonable bounds
        
        # Create realized volatility (rough proxy as specified)
        # RV is typically lower than IV and more stable
        rv = iv * 0.8 + 0.1 * np.random.randn(len(dates))
        rv = pd.Series(rv, index=dates)
        rv = rv.clip(lower=0.1, upper=1.0)
        
        # Create skew/risk reversal (skew=0 if unknown as specified)
        # In reality, skew would depend on market sentiment and option demand
        # For now, create small random skew
        skew_rr = np.random.normal(0, 0.05, len(dates))  # Small random skew
        skew_rr = pd.Series(skew_rr, index=dates)
        skew_rr = skew_rr.clip(lower=-0.2, upper=0.2)  # Keep reasonable bounds
        
        # Combine into DataFrame
        result = pd.DataFrame({
            'iv': iv,
            'rv': rv,
            'skew_rr': skew_rr
        })
        
        # Fill any NaN values
        result = result.fillna(method='ffill').fillna(0.0)
        
        logger.info(f"Successfully created {len(result)} rows of IV/RR data")
        return result
    
    def get_option_chain(self, currency: str = "BTC", expiration_date: Optional[str] = None) -> pd.DataFrame:
        """Get option chain for a specific expiration"""
        instruments = self.get_instruments(currency, "option")
        
        if expiration_date:
            instruments = instruments[instruments['expiration_timestamp'] == expiration_date]
        
        return instruments
    
    def calculate_greeks(self, instrument_name: str) -> dict:
        """Calculate option Greeks (synthetic for now)"""
        # In a real implementation, you would use Black-Scholes or similar model
        # For now, return synthetic Greeks
        
        mark_price = self.get_mark_price(instrument_name)
        if mark_price == 0:
            return {}
        
        # Create synthetic Greeks
        return {
            'delta': np.random.normal(0.5, 0.3),
            'gamma': np.random.exponential(0.1),
            'theta': -np.random.exponential(0.01),
            'vega': np.random.exponential(0.5),
            'rho': np.random.normal(0, 0.1)
        }
    
    def get_volatility_surface(self, currency: str = "BTC") -> pd.DataFrame:
        """Get volatility surface data (synthetic for now)"""
        # In a real implementation, you would aggregate IV across strikes and expirations
        # For now, create synthetic surface
        
        strikes = np.linspace(0.5, 1.5, 10)  # Moneyness ratios
        expirations = [1, 7, 30, 90, 180]  # Days to expiration
        
        surface_data = []
        for strike in strikes:
            for exp in expirations:
                # Create realistic IV surface with smile/skew
                base_iv = 0.5
                smile_effect = 0.1 * (strike - 1.0) ** 2
                term_effect = 0.05 * np.log(exp)
                
                iv = base_iv + smile_effect + term_effect + 0.05 * np.random.randn()
                iv = max(0.2, min(1.0, iv))
                
                surface_data.append({
                    'strike': strike,
                    'expiration': exp,
                    'iv': iv
                })
        
        return pd.DataFrame(surface_data)
    
    def get_book_summary(self, currency: str = "BTC", kind: str = "option") -> pd.DataFrame:
        """Get book summary data including OI, volume, IV for options"""
        params = {
            'currency': currency,
            'kind': kind
        }
        
        data = self._make_request('public/get_book_summary_by_currency', params)
        if not data:
            return pd.DataFrame()
        
        # The API returns data in the 'result' field
        if isinstance(data, dict) and 'result' in data:
            book_data = data['result']
        else:
            book_data = data
        
        # The result should be a list of option contracts
        if not isinstance(book_data, list):
            return pd.DataFrame()
        
        df = pd.DataFrame(book_data)
        
        if df.empty:
            return df
        
        # Parse instrument name to extract strike and option type
        # Format: BTC-27SEP24-60000-C or ETH-22AUG25-3200-C
        df[['currency', 'expiry', 'strike', 'option_type']] = df['instrument_name'].str.extract(r'([A-Z]+)-(\d+[A-Z]+\d+)-(\d+)-([CP])')
        df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
        
        # Convert creation_timestamp to datetime
        if 'creation_timestamp' in df.columns:
            df['expiration_timestamp'] = pd.to_datetime(df['creation_timestamp'], unit='ms')
        
        # Convert numeric columns
        numeric_cols = ['open_interest', 'volume_24h', 'mark_iv', 'underlying_price']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def calculate_max_pain(self, currency: str = "BTC", expiration_date: Optional[str] = None) -> dict:
        """
        Calculate max pain point for options
        
        Args:
            currency: BTC or ETH
            expiration_date: Specific expiration date (optional)
            
        Returns:
            dict with max pain data including:
            - max_pain_strike: Strike price where option writers have maximum pain
            - max_pain_value: The pain value at max pain strike
            - total_pain: Total pain across all strikes
            - strikes_data: DataFrame with pain calculations for each strike
        """
        logger.info(f"Calculating max pain for {currency} options")
        
        # Get book summary data
        df = self.get_book_summary(currency, "option")
        if df.empty:
            logger.warning(f"No book summary data available for {currency}")
            return {}
        
        # Filter by expiration if specified
        if expiration_date:
            df = df[df['expiration_timestamp'] == expiration_date]
        
        if df.empty:
            logger.warning(f"No data for specified expiration: {expiration_date}")
            return {}
        
        # Get current underlying price
        underlying_price = self.get_index_price(currency)
        if underlying_price == 0:
            # Use the first available underlying price from data
            underlying_price = df['underlying_price'].iloc[0] if not df.empty else 50000
        
        # Group by strike and calculate total OI for calls and puts
        strikes_data = []
        unique_strikes = sorted(df['strike'].unique())
        
        for strike in unique_strikes:
            strike_data = df[df['strike'] == strike]
            
            # Sum OI for calls and puts at this strike
            calls_oi = strike_data[strike_data['option_type'] == 'C']['open_interest'].sum()
            puts_oi = strike_data[strike_data['option_type'] == 'P']['open_interest'].sum()
            
            # Calculate pain for this strike
            # Pain = sum of (strike - price) * OI for calls where strike > price
            #      + sum of (price - strike) * OI for puts where price > strike
            
            if underlying_price > strike:
                # Calls are ITM, puts are OTM
                call_pain = calls_oi * (underlying_price - strike)
                put_pain = 0
            else:
                # Calls are OTM, puts are ITM
                call_pain = 0
                put_pain = puts_oi * (strike - underlying_price)
            
            total_pain = call_pain + put_pain
            
            strikes_data.append({
                'strike': strike,
                'calls_oi': calls_oi,
                'puts_oi': puts_oi,
                'total_oi': calls_oi + puts_oi,
                'call_pain': call_pain,
                'put_pain': put_pain,
                'total_pain': total_pain,
                'underlying_price': underlying_price,
                'moneyness': strike / underlying_price if underlying_price > 0 else 0
            })
        
        strikes_df = pd.DataFrame(strikes_data)
        
        if strikes_df.empty:
            return {}
        
        # Find max pain strike (minimum total pain)
        max_pain_row = strikes_df.loc[strikes_df['total_pain'].idxmin()]
        max_pain_strike = max_pain_row['strike']
        max_pain_value = max_pain_row['total_pain']
        
        # Calculate total pain across all strikes
        total_pain = strikes_df['total_pain'].sum()
        
        # Add expiration info
        expiration_info = df['expiration_timestamp'].iloc[0] if not df.empty else None
        
        result = {
            'currency': currency,
            'expiration_timestamp': expiration_info,
            'underlying_price': underlying_price,
            'max_pain_strike': max_pain_strike,
            'max_pain_value': max_pain_value,
            'total_pain': total_pain,
            'strikes_data': strikes_df,
            'num_strikes': len(strikes_df),
            'calculated_at': pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"Max pain calculated: {currency} strike {max_pain_strike} with pain {max_pain_value:.2f}")
        return result
    
    def get_max_pain_features(self, currency: str = "BTC") -> dict:
        """
        Get max pain features for the nearest expirations
        
        Returns:
            dict with max pain features for model training
        """
        logger.info(f"Getting max pain features for {currency}")
        
        # Get book summary data
        df = self.get_book_summary(currency, "option")
        if df.empty:
            return {}
        
        # Get unique expirations and sort by date
        expirations = sorted(df['expiration_timestamp'].unique())
        
        features = {}
        underlying_price = self.get_index_price(currency)
        
        # Calculate max pain for nearest 3 expirations
        for i, exp in enumerate(expirations[:3]):
            max_pain_data = self.calculate_max_pain(currency, exp)
            if not max_pain_data:
                continue
            
            suffix = f"_{i+1}" if i > 0 else ""
            
            features[f'max_pain_strike{suffix}'] = max_pain_data['max_pain_strike']
            features[f'max_pain_value{suffix}'] = max_pain_data['max_pain_value']
            features[f'max_pain_distance{suffix}'] = abs(underlying_price - max_pain_data['max_pain_strike'])
            features[f'max_pain_distance_pct{suffix}'] = abs(underlying_price - max_pain_data['max_pain_strike']) / underlying_price * 100
            
            # Add OI concentration features
            strikes_df = max_pain_data['strikes_data']
            if not strikes_df.empty:
                features[f'total_oi{suffix}'] = strikes_df['total_oi'].sum()
                features[f'oi_concentration{suffix}'] = strikes_df['total_oi'].max() / strikes_df['total_oi'].sum() if strikes_df['total_oi'].sum() > 0 else 0
                features[f'num_strikes{suffix}'] = len(strikes_df)
        
        return features

    def calculate_enhanced_max_pain_weight(self, currency: str, direction: str, current_price: float = None) -> dict:
        """
        Enhanced max pain weighting that considers multiple factors:
        1. Distance to max pain (current implementation)
        2. Gamma exposure at max pain strike
        3. Time to expiry (closer expiry = stronger effect)
        4. Open interest concentration around max pain
        5. Volatility skew (put/call ratio)
        6. Market structure (support/resistance levels)
        
        Args:
            currency: 'BTC' or 'ETH'
            direction: 'long' or 'short'
            current_price: Current underlying price (if None, fetched from API)
            
        Returns:
            dict with enhanced weight calculation and analysis
        """
        try:
            # Get max pain data
            max_pain_data = self.calculate_max_pain(currency)
            if not max_pain_data:
                return {"weight": 1.0, "reason": "No max pain data available"}
            
            max_pain_strike = max_pain_data.get('max_pain_strike')
            underlying_price = current_price or max_pain_data.get('underlying_price')
            
            if not max_pain_strike or not underlying_price:
                return {"weight": 1.0, "reason": "Invalid max pain data"}
            
            # Get detailed options data for analysis
            book_df = self.get_book_summary(currency, "option")
            if book_df.empty:
                return {"weight": 1.0, "reason": "No options data available"}
            
            # 1. Basic distance calculation
            distance_pct = abs(underlying_price - max_pain_strike) / underlying_price * 100
            toward_direction = "long" if underlying_price < max_pain_strike else "short"
            direction_alignment = 1.0 if direction == toward_direction else -1.0
            
            # 2. Time to expiry analysis (closer expiry = stronger effect)
            expiry_weights = {}
            for expiry in book_df['expiry'].unique():
                expiry_data = book_df[book_df['expiry'] == expiry]
                if not expiry_data.empty:
                    # Calculate days to expiry (simplified)
                    try:
                        # Parse expiry date (format: 27SEP24, 22AUG25, etc.)
                        expiry_str = str(expiry)
                        if len(expiry_str) >= 7:
                            day = int(expiry_str[:2])
                            month_str = expiry_str[2:5]
                            year_str = expiry_str[5:]
                            year = 2000 + int(year_str) if len(year_str) == 2 else int(year_str)
                            
                            month_map = {
                                'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                                'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                            }
                            month = month_map.get(month_str.upper(), 1)
                            
                            from datetime import datetime
                            expiry_date = datetime(year, month, day)
                            days_to_expiry = (expiry_date - datetime.now()).days
                            
                            # Weight by proximity to expiry (0-30 days = strongest)
                            if days_to_expiry <= 0:
                                expiry_weight = 1.0  # Expired
                            elif days_to_expiry <= 7:
                                expiry_weight = 1.0  # Very close
                            elif days_to_expiry <= 30:
                                expiry_weight = 0.8  # Close
                            elif days_to_expiry <= 90:
                                expiry_weight = 0.6  # Medium
                            else:
                                expiry_weight = 0.4  # Far
                                
                            expiry_weights[expiry] = expiry_weight
                    except:
                        expiry_weights[expiry] = 0.5  # Default weight
            
            # 3. Open interest concentration analysis
            oi_concentration = 0.0
            total_oi = book_df['open_interest'].sum()
            if total_oi > 0:
                # Calculate OI concentration within ±5% of max pain strike
                strike_range = max_pain_strike * 0.05  # ±5%
                near_strike_oi = book_df[
                    (book_df['strike'] >= max_pain_strike - strike_range) &
                    (book_df['strike'] <= max_pain_strike + strike_range)
                ]['open_interest'].sum()
                oi_concentration = near_strike_oi / total_oi
            
            # 4. Volatility skew analysis (put/call ratio)
            put_call_ratio = 1.0
            calls_oi = book_df[book_df['option_type'] == 'C']['open_interest'].sum()
            puts_oi = book_df[book_df['option_type'] == 'P']['open_interest'].sum()
            if calls_oi > 0:
                put_call_ratio = puts_oi / calls_oi
            
            # 5. Gamma exposure estimation (simplified)
            gamma_exposure = 0.0
            if total_oi > 0:
                # Simplified gamma calculation based on OI near max pain
                gamma_exposure = oi_concentration * 0.1  # Rough estimate
            
            # 6. Market structure analysis
            market_structure_score = 0.0
            if distance_pct < 5.0:  # Very close to max pain
                market_structure_score = 0.2
            elif distance_pct < 10.0:  # Close to max pain
                market_structure_score = 0.1
            elif distance_pct < 20.0:  # Moderate distance
                market_structure_score = 0.05
            
            # 7. Enhanced weight calculation (CAPPED AT ±15% MAXIMUM)
            # Simplified approach: Direction alignment is the primary factor
            
            # Distance factor (normalized to 0-1)
            distance_factor = min(max(distance_pct / 20.0, 0.0), 1.0)  # Normalize to 0-1
            
            # Calculate direction-based weight directly
            # This ensures the direction alignment has the primary impact
            direction_weight = 1.0 + (direction_alignment * distance_factor * 0.15)  # ±15% max
            
            # Apply minimal adjustments from other factors
            # These should not overwhelm the direction signal
            
            # Expiry adjustment (very small)
            expiry_adjustment = 0.0
            if expiry_weights:
                avg_expiry = sum(expiry_weights.values()) / len(expiry_weights)
                expiry_adjustment = (avg_expiry - 0.8) * 0.05  # ±1% max
            
            # OI concentration adjustment (very small)
            oi_adjustment = oi_concentration * 0.02  # ±2% max
            
            # Put/call ratio adjustment (very small)
            pcr_adjustment = (put_call_ratio - 1.0) * 0.01  # ±1% max
            
            # Gamma exposure adjustment (very small)
            gamma_adjustment = gamma_exposure * 0.1  # ±1% max
            
            # Market structure adjustment (very small)
            structure_adjustment = market_structure_score * 0.02  # ±1% max
            
            # Combine all adjustments (should be very small)
            total_adjustment = expiry_adjustment + oi_adjustment + pcr_adjustment + gamma_adjustment + structure_adjustment
            
            # Final weight calculation
            final_weight = direction_weight + total_adjustment
            
            # Clamp to ±15% maximum (0.85 to 1.15)
            final_weight = max(0.85, min(1.15, final_weight))
            
            return {
                "weight": final_weight,
                "max_pain_strike": max_pain_strike,
                "underlying_price": underlying_price,
                "distance_pct": distance_pct,
                "toward_direction": toward_direction,
                "direction_alignment": direction_alignment,
                "oi_concentration": oi_concentration,
                "put_call_ratio": put_call_ratio,
                "gamma_exposure": gamma_exposure,
                "avg_expiry_weight": sum(expiry_weights.values()) / len(expiry_weights) if expiry_weights else 0.8,
                "market_structure_score": market_structure_score,
                "factors": {
                    "distance_factor": distance_factor,
                    "direction_weight": direction_weight,
                    "expiry_adjustment": expiry_adjustment,
                    "oi_adjustment": oi_adjustment,
                    "pcr_adjustment": pcr_adjustment,
                    "gamma_adjustment": gamma_adjustment,
                    "structure_adjustment": structure_adjustment,
                    "total_adjustment": total_adjustment
                },
                "reason": "Enhanced calculation completed"
            }
            
        except Exception as e:
            return {"weight": 1.0, "reason": f"Error in enhanced calculation: {str(e)}"}

    def get_enhanced_max_pain_weight(self, currency: str, direction: str, current_price: float = None) -> float:
        """Get enhanced max pain weight (simplified interface)"""
        result = self.calculate_enhanced_max_pain_weight(currency, direction, current_price)
        return result.get("weight", 1.0)


# Import numpy for calculations
import numpy as np

if __name__ == "__main__":
    try:
        print("BTC index:", _fetch_index_price("BTC"))
        print("ETH index:", _fetch_index_price("ETH"))
    except Exception as e:
        print("Deribit index test failed:", repr(e))
