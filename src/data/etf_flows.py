import pandas as pd
import requests
import time
from typing import Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ETFFlowsProvider:
    """ETF flows data provider"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; alpha12_24/1.0)'
        })
    
    def _make_request(self, url: str, max_retries: int = 3) -> Optional[requests.Response]:
        """Make HTTP request with retry/backoff logic"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All retries failed for {url}")
                    return None
    
    def load_csv_url(self, url: str) -> pd.DataFrame:
        """
        Load CSV data from URL
        
        Args:
            url: URL to the CSV file
            
        Returns:
            DataFrame containing the CSV data
        """
        logger.info(f"Loading CSV from URL: {url}")
        
        response = self._make_request(url)
        if not response:
            logger.error(f"Failed to load CSV from {url}")
            return pd.DataFrame()
        
        try:
            # Try to read CSV from the response content
            df = pd.read_csv(pd.StringIO(response.text))
            logger.info(f"Successfully loaded {len(df)} rows from CSV")
            return df
        except Exception as e:
            logger.error(f"Failed to parse CSV from {url}: {e}")
            return pd.DataFrame()
    
    def parse_farside_csv(self, df: pd.DataFrame) -> pd.Series:
        """
        Parse Farside CSV data to extract ETF net flows
        
        Args:
            df: DataFrame from load_csv_url
            
        Returns:
            Series with 'etf_net_flow' and UTC index
        """
        logger.info("Parsing Farside CSV data")
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return pd.Series(dtype=float)
        
        try:
            # Common column patterns for Farside data
            # Look for date/time columns
            date_columns = [col for col in df.columns if any(keyword in col.lower() 
                           for keyword in ['date', 'time', 'timestamp'])]
            
            # Look for flow/volume columns
            flow_columns = [col for col in df.columns if any(keyword in col.lower() 
                           for keyword in ['flow', 'volume', 'amount', 'net'])]
            
            if not date_columns:
                logger.warning("No date column found, using index")
                date_col = df.index
            else:
                date_col = df[date_columns[0]]
            
            if not flow_columns:
                logger.warning("No flow column found, using first numeric column")
                # Use first numeric column as flow
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    flow_col = df[numeric_cols[0]]
                else:
                    logger.error("No numeric columns found for flow data")
                    return pd.Series(dtype=float)
            else:
                flow_col = df[flow_columns[0]]
            
            # Convert date column to datetime
            if isinstance(date_col, pd.Series):
                try:
                    # Try different date formats
                    if date_col.dtype == 'object':
                        # Try common date formats
                        for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y', '%d/%m/%Y']:
                            try:
                                dates = pd.to_datetime(date_col, format=fmt)
                                break
                            except:
                                continue
                        else:
                            # If no format works, use pandas default parsing
                            dates = pd.to_datetime(date_col, errors='coerce')
                    else:
                        dates = pd.to_datetime(date_col)
                except Exception as e:
                    logger.warning(f"Failed to parse dates: {e}, using index")
                    dates = pd.to_datetime(df.index)
            else:
                dates = pd.to_datetime(df.index)
            
            # Convert flow column to numeric
            flows = pd.to_numeric(flow_col, errors='coerce')
            
            # Create Series with UTC index
            result = pd.Series(flows.values, index=dates, name='etf_net_flow')
            
            # Ensure UTC timezone
            if result.index.tz is None:
                result.index = result.index.tz_localize('UTC')
            else:
                result.index = result.index.tz_convert('UTC')
            
            # Remove any NaN values
            result = result.dropna()
            
            logger.info(f"Successfully parsed {len(result)} ETF flow records")
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse Farside CSV: {e}")
            return pd.Series(dtype=float)
    
    def get_grayscale_flows(self, days: int = 30) -> pd.Series:
        """Get Grayscale ETF flows (synthetic for now)"""
        # In a real implementation, you would fetch from Grayscale API or website
        # For now, create synthetic data
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D', tz='UTC')
        
        # Create realistic Grayscale flows
        # Grayscale flows are typically in millions of USD
        base_flow = 50  # Base daily flow in millions
        volatility = 100  # Daily volatility in millions
        
        flows = np.random.normal(base_flow, volatility, len(dates))
        flows = pd.Series(flows, index=dates, name='grayscale_flow')
        
        return flows
    
    def get_spot_etf_flows(self, days: int = 30) -> pd.Series:
        """Get spot ETF flows (synthetic for now)"""
        # In a real implementation, you would fetch from ETF providers or aggregators
        # For now, create synthetic data
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D', tz='UTC')
        
        # Create realistic spot ETF flows
        # Spot ETF flows are typically larger than Grayscale
        base_flow = 200  # Base daily flow in millions
        volatility = 300  # Daily volatility in millions
        
        flows = np.random.normal(base_flow, volatility, len(dates))
        flows = pd.Series(flows, index=dates, name='spot_etf_flow')
        
        return flows
    
    def get_futures_etf_flows(self, days: int = 30) -> pd.Series:
        """Get futures ETF flows (synthetic for now)"""
        # In a real implementation, you would fetch from futures ETF providers
        # For now, create synthetic data
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D', tz='UTC')
        
        # Create realistic futures ETF flows
        # Futures ETF flows are typically smaller than spot
        base_flow = 30  # Base daily flow in millions
        volatility = 50  # Daily volatility in millions
        
        flows = np.random.normal(base_flow, volatility, len(dates))
        flows = pd.Series(flows, index=dates, name='futures_etf_flow')
        
        return flows
    
    def get_etf_holdings(self, etf_symbol: str = "GBTC") -> pd.Series:
        """Get ETF holdings over time (synthetic for now)"""
        # In a real implementation, you would fetch from ETF provider
        # For now, create synthetic data
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        dates = pd.date_range(start=start_date, end=end_date, freq='D', tz='UTC')
        
        # Create realistic holdings growth
        base_holdings = 500000  # Base holdings in BTC
        growth_rate = 0.001  # Daily growth rate
        volatility = 0.02  # Daily volatility
        
        # Simulate holdings growth with some randomness
        holdings = [base_holdings]
        for i in range(1, len(dates)):
            growth = np.random.normal(growth_rate, volatility)
            new_holdings = holdings[-1] * (1 + growth)
            holdings.append(max(0, new_holdings))  # Holdings can't be negative
        
        holdings = pd.Series(holdings, index=dates, name=f'{etf_symbol}_holdings')
        return holdings
    
    def get_etf_premium_discount(self, etf_symbol: str = "GBTC") -> pd.Series:
        """Get ETF premium/discount to NAV (synthetic for now)"""
        # In a real implementation, you would fetch from ETF provider
        # For now, create synthetic data
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        dates = pd.date_range(start=start_date, end=end_date, freq='D', tz='UTC')
        
        # Create realistic premium/discount pattern
        # GBTC historically had large premiums, now often discounts
        base_premium = -0.05  # 5% discount
        volatility = 0.03  # 3% daily volatility
        
        premiums = np.random.normal(base_premium, volatility, len(dates))
        premiums = pd.Series(premiums, index=dates, name=f'{etf_symbol}_premium')
        
        return premiums
    
    def get_aggregated_flows(self, days: int = 30) -> pd.DataFrame:
        """Get aggregated ETF flows from multiple sources"""
        logger.info(f"Getting aggregated ETF flows for {days} days")
        
        # Get flows from different sources
        grayscale = self.get_grayscale_flows(days)
        spot = self.get_spot_etf_flows(days)
        futures = self.get_futures_etf_flows(days)
        
        # Combine into DataFrame
        flows_df = pd.DataFrame({
            'grayscale_flow': grayscale,
            'spot_etf_flow': spot,
            'futures_etf_flow': futures
        })
        
        # Calculate total net flow
        flows_df['total_net_flow'] = flows_df.sum(axis=1)
        
        # Fill any NaN values
        flows_df = flows_df.fillna(method='ffill').fillna(0.0)
        
        logger.info(f"Successfully aggregated {len(flows_df)} days of ETF flows")
        return flows_df
    
    def get_flow_sentiment(self, days: int = 30) -> pd.Series:
        """Calculate ETF flow sentiment indicator"""
        # Get aggregated flows
        flows_df = self.get_aggregated_flows(days)
        
        if flows_df.empty:
            return pd.Series(dtype=float)
        
        # Calculate sentiment based on recent flows
        # Positive sentiment: recent positive flows
        # Negative sentiment: recent negative flows
        
        # Use 7-day rolling average for sentiment
        rolling_flow = flows_df['total_net_flow'].rolling(7).mean()
        
        # Normalize to sentiment score (-1 to 1)
        # Use historical volatility for normalization
        flow_std = flows_df['total_net_flow'].std()
        if flow_std > 0:
            sentiment = rolling_flow / (2 * flow_std)  # Normalize to ±0.5 range
            sentiment = sentiment.clip(-1, 1)  # Clip to ±1
        else:
            sentiment = pd.Series(0, index=flows_df.index)
        
        sentiment.name = 'etf_flow_sentiment'
        return sentiment.fillna(0)


# Import numpy for calculations
import numpy as np
