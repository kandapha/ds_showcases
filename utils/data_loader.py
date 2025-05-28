import yfinance as yf
import pandas as pd
import streamlit as st
from typing import Optional, Dict, Any
import numpy as np
from datetime import datetime, timedelta

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_data_mpf(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance and format it for analysis
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL', 'MSFT', 'PTT.BK')
        period (str): Time period ('1d', '5d', '1mo', '3mo', '6mo', 'ytd', '1y', '2y', 'max')
        interval (str): Data interval ('1m', '5m', '15m', '60m', '1d', '1wk', '1mo')
    
    Returns:
        pd.DataFrame: OHLC data formatted for analysis
    """
    try:
        # Create ticker object
        stock = yf.Ticker(symbol)
        
        # Fetch historical data with error handling
        df = stock.history(period=period, interval=interval, auto_adjust=True, prepost=True)
        
        # Check if data is empty
        if df.empty:
            st.warning(f"No data found for symbol: {symbol}")
            return pd.DataFrame()
        
        # Clean and format data
        df.index.name = "Date"
        
        # Ensure we have the required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Missing required column: {col}")
                return pd.DataFrame()
        
        df = df[required_cols]
        
        # Remove any rows with NaN values in OHLC data
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        
        # Fill NaN values in volume with 0
        df['Volume'] = df['Volume'].fillna(0)
        
        # Ensure volume is integer type
        df['Volume'] = df['Volume'].astype(int)
        
        # Validate data integrity
        if len(df) == 0:
            st.warning(f"No valid data points found for {symbol}")
            return pd.DataFrame()
        
        # Sort by date to ensure chronological order
        df = df.sort_index()
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

# --- Updated get_stock_info with proper dividend_yield formatting ---
@st.cache_data(ttl=600)
def get_stock_info(symbol: str) -> Optional[Dict[str, Any]]:
    try:
        stock = yf.Ticker(symbol)
        info = stock.info

        raw_yield = info.get('dividendYield', 0)
        if raw_yield is None:
            formatted_yield = "N/A"
        elif raw_yield < 0.5:
            formatted_yield = f"{raw_yield * 100:.2f}%"
        else:
            formatted_yield = f"{raw_yield:.2f}%"

        
    

        stock_info = {
            'name': info.get('longName', info.get('shortName', symbol)),
            'symbol': symbol,
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'exchange': info.get('exchange', 'N/A'),
            'currency': info.get('currency', 'USD'),
            'country': info.get('country', 'N/A'),
            'website': info.get('website', 'N/A'),
            'business_summary': info.get('longBusinessSummary', 'N/A'),

            # Financial metrics
            'market_cap': info.get('marketCap', 0),
            'enterprise_value': info.get('enterpriseValue', 0),
            'pe_ratio': info.get('trailingPE', info.get('forwardPE', 0)),
            'beta': info.get('beta', 'N/A'),
            'dividend_yield': raw_yield,
            'dividend_rate': info.get('dividendRate', 0),

            # Price metrics
            'previous_close': info.get('previousClose', 0),
            'regular_market_open': info.get('regularMarketOpen', 0),
            'day_low': info.get('dayLow', 0),
            'day_high': info.get('dayHigh', 0),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),

            # Volume metrics
            'volume': info.get('volume', 0),
            'average_volume': info.get('averageVolume', 0),
            'average_volume_10days': info.get('averageVolume10days', 0),

            # Analyst data
            'target_high_price': info.get('targetHighPrice', 0),
            'target_low_price': info.get('targetLowPrice', 0),
            'target_mean_price': info.get('targetMeanPrice', 0),
            'recommendation_key': info.get('recommendationKey', 'N/A'),
            'number_of_analyst_opinions': info.get('numberOfAnalystOpinions', 0),

            # Additional metrics
            'price_to_book': info.get('priceToBook', 0),
            'return_on_equity': info.get('returnOnEquity', 0),
            'return_on_assets': info.get('returnOnAssets', 0),
            'debt_to_equity': info.get('debtToEquity', 0),
            'gross_margins': info.get('grossMargins', 0),
            'operating_margins': info.get('operatingMargins', 0),
            'profit_margins': info.get('profitMargins', 0),
        }

        return stock_info

    except Exception as e:
        st.warning(f"Could not fetch detailed info for {symbol}: {str(e)}")
        return None



@st.cache_data(ttl=3600)  # Cache for 1 hour
def validate_symbol(symbol: str) -> bool:
    """
    Validate if a stock symbol exists and has available data
    
    Args:
        symbol (str): Stock symbol to validate
        
    Returns:
        bool: True if symbol exists and has data, False otherwise
    """
    try:
        stock = yf.Ticker(symbol)
        
        # Try to get recent data to validate symbol
        test_data = stock.history(period="5d", interval="1d")
        
        # Check if we have valid data
        if test_data.empty:
            return False
        
        # Additional validation - check if we have essential columns
        required_cols = ["Open", "High", "Low", "Close"]
        if not all(col in test_data.columns for col in required_cols):
            return False
        
        # Check if we have at least one valid price point
        if test_data[required_cols].isna().all().all():
            return False
        
        return True
        
    except Exception as e:
        return False

def get_market_status() -> Dict[str, Any]:
    """
    Get current market status information
    
    Returns:
        dict: Market status information
    """
    try:
        # Use SPY as a proxy for US market status
        spy = yf.Ticker("SPY")
        info = spy.info
        
        market_status = {
            'market_state': info.get('marketState', 'UNKNOWN'),
            'exchange_timezone': info.get('exchangeTimezoneName', 'EST'),
            'regular_market_time': info.get('regularMarketTime', None),
            'pre_market_time': info.get('preMarketTime', None),
            'post_market_time': info.get('postMarketTime', None),
        }
        
        return market_status
        
    except Exception as e:
        return {'market_state': 'UNKNOWN', 'error': str(e)}

def get_sector_performance() -> Optional[pd.DataFrame]:
    """
    Get sector performance data using major sector ETFs
    
    Returns:
        pd.DataFrame: Sector performance data or None if error
    """
    try:
        sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV', 
            'Financial': 'XLF',
            'Consumer Discretionary': 'XLY',
            'Communication': 'XLC',
            'Industrial': 'XLI',
            'Consumer Staples': 'XLP',
            'Energy': 'XLE',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE',
            'Materials': 'XLB'
        }
        
        sector_data = []
        
        for sector, etf in sector_etfs.items():
            try:
                ticker = yf.Ticker(etf)
                hist = ticker.history(period="2d")
                
                if not hist.empty and len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    previous_price = hist['Close'].iloc[-2]
                    change_pct = ((current_price - previous_price) / previous_price) * 100
                    
                    sector_data.append({
                        'Sector': sector,
                        'ETF': etf,
                        'Price': current_price,
                        'Change%': change_pct
                    })
            except:
                continue
        
        if sector_data:
            df = pd.DataFrame(sector_data)
            df = df.sort_values('Change%', ascending=False)
            return df
        
        return None
        
    except Exception as e:
        return None

def get_trending_stocks(region: str = 'US') -> Optional[pd.DataFrame]:
    """
    Get trending stocks (Note: This is a simplified version)
    Yahoo Finance trending API is limited, so this returns popular stocks
    
    Args:
        region (str): Market region
        
    Returns:
        pd.DataFrame: Trending stocks data or None if error
    """
    try:
        # Popular stocks as a proxy for trending (since Yahoo's trending API is limited)
        popular_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']
        
        trending_data = []
        
        for symbol in popular_stocks:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                info = ticker.info
                
                if not hist.empty and len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    previous_price = hist['Close'].iloc[-2]
                    change_pct = ((current_price - previous_price) / previous_price) * 100
                    
                    trending_data.append({
                        'Symbol': symbol,
                        'Name': info.get('shortName', symbol),
                        'Price': current_price,
                        'Change%': change_pct,
                        'Volume': hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0
                    })
            except:
                continue
        
        if trending_data:
            df = pd.DataFrame(trending_data)
            df = df.sort_values('Volume', ascending=False)
            return df.head(10)
        
        return None
        
    except Exception as e:
        return None

def get_financial_ratios(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Get comprehensive financial ratios for a stock
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        dict: Financial ratios or None if error
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info

       

        ratios = {
            # Valuation Ratios
            'pe_ratio': info.get('trailingPE', 0),
            'forward_pe': info.get('forwardPE', 0),
            'peg_ratio': info.get('pegRatio', 0),
            'price_to_book': info.get('priceToBook', 0),
            'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
            'enterprise_to_revenue': info.get('enterpriseToRevenue', 0),
            'enterprise_to_ebitda': info.get('enterpriseToEbitda', 0),

            # Profitability Ratios
            'profit_margins': info.get('profitMargins', 0),
            'operating_margins': info.get('operatingMargins', 0),
            'gross_margins': info.get('grossMargins', 0),
            'return_on_assets': info.get('returnOnAssets', 0),
            'return_on_equity': info.get('returnOnEquity', 0),

            # Financial Health Ratios
            'debt_to_equity': info.get('debtToEquity', 0),
            'current_ratio': info.get('currentRatio', 0),
            'quick_ratio': info.get('quickRatio', 0),

            # Growth Ratios
            'earnings_growth': info.get('earningsGrowth', 0),
            'revenue_growth': info.get('revenueGrowth', 0),

            # Dividend Ratios
            'dividend_yield': info.get('dividendYield', 0) ,
            'payout_ratio': info.get('payoutRatio', 0),
        }


        return ratios

    except Exception as e:
        return None


# Utility function for error handling and retry logic
def safe_fetch_with_retry(fetch_func, *args, max_retries: int = 3, **kwargs):
    """
    Safely fetch data with retry logic
    
    Args:
        fetch_func: Function to call
        *args: Arguments for the function
        max_retries: Maximum number of retries
        **kwargs: Keyword arguments for the function
        
    Returns:
        Function result or None if all retries fail
    """
    for attempt in range(max_retries):
        try:
            result = fetch_func(*args, **kwargs)
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                st.warning(f"Failed to fetch data after {max_retries} attempts: {str(e)}")
                return None
            else:
                st.info(f"Retry attempt {attempt + 1}...")
                continue
    
    return None



