# app.py
import streamlit as st
from utils.data_loader import get_stock_data_mpf, get_stock_info, validate_symbol
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="TradingLite - Advanced Stock Analysis", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# --- Helper Functions ---
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
   
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
def calculate_macd(data, fast=12, slow=26, signal=9):
    exp1 = data.ewm(span=fast).mean()
    exp2 = data.ewm(span=slow).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line

def calculate_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent

def format_number(num):
    if num >= 1e12:
        return f"{num/1e12:.2f}T"
    elif num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return f"{num:.2f}"

def get_stock_summary(symbol, df, stock_info):
    """Generate enhanced stock summary with key metrics"""
    if df.empty:
        return "No data available"
    
    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) > 1 else latest
    
    # Price metrics
    price_change = latest['Close'] - previous['Close']
    price_change_pct = (price_change / previous['Close']) * 100
    
    # Technical levels
    high_52w = df['High'].max()
    low_52w = df['Low'].min()
    avg_volume = df['Volume'].mean()
    volatility = df['Close'].pct_change().std() * np.sqrt(252) * 100
    
    # RSI and trend
    rsi = calculate_rsi(df['Close']).iloc[-1]
    rsi_signal = "🔴 Overbought" if rsi > 70 else "🟢 Oversold" if rsi < 30 else "🟡 Neutral"
    
    # Moving average trend analysis
    ma20 = df['Close'].rolling(20).mean().iloc[-1]
    ma50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) > 50 else ma20
    
    if latest['Close'] > ma20 > ma50:
        trend = "🚀 Strong Bullish"
    elif latest['Close'] > ma20:
        trend = "📈 Bullish"
    elif latest['Close'] < ma20 < ma50:
        trend = "📉 Strong Bearish"
    else:
        trend = "📊 Bearish"
    
    # Support and Resistance levels
    support = df['Low'].rolling(20).min().iloc[-1]
    resistance = df['High'].rolling(20).max().iloc[-1]
    
    summary = f"""
    **{symbol} Stock Analysis Summary**
    
    💰 **Current Status**: ${latest['Close']:.2f} ({price_change_pct:+.2f}%)
    
    📊 **Technical Overview**:
    - Market Trend: {trend}
    - RSI Signal: {rsi:.1f} ({rsi_signal})
    - Volatility: {volatility:.1f}% (Annualized)
    
    🎯 **Key Trading Levels**:
    - Resistance: ${resistance:.2f}
    - Support: ${support:.2f}
    - 52W High: ${high_52w:.2f}
    - 52W Low: ${low_52w:.2f}
    
    📈 **Volume Analysis**:
    - Average Volume: {format_number(avg_volume)}
    - Current Volume: {format_number(latest['Volume'])}
    """
    
    if stock_info:
        summary += f"""
    
    🏢 **Company Fundamentals**:
    - Sector: {stock_info.get('sector', 'N/A')}
    - Industry: {stock_info.get('industry', 'N/A')}
    - Market Cap: ${format_number(stock_info.get('market_cap', 0))}
    - P/E Ratio: {stock_info.get('pe_ratio', 0):.1f}
    - Beta: {stock_info.get('beta', 'N/A')}
    """
    
    return summary

def get_trading_recommendation(df):
    """Generate trading recommendation based on technical indicators"""
    if df.empty or len(df) < 50:
        return "📊 **Insufficient data for recommendation**"
    
    latest = df.iloc[-1]
    
    # Technical indicators
    rsi = calculate_rsi(df['Close']).iloc[-1]
    macd, signal = calculate_macd(df['Close'])
    current_macd = macd.iloc[-1]
    current_signal = signal.iloc[-1]
    
    # Moving averages
    ma20 = df['Close'].rolling(20).mean().iloc[-1]
    ma50 = df['Close'].rolling(50).mean().iloc[-1]
    
    # Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(df['Close'])
    current_price = latest['Close']
    current_upper = upper.iloc[-1]
    current_lower = lower.iloc[-1]
    bb_position = (current_price - current_lower) / (current_upper - current_lower)
    
    # Scoring system
    buy_signals = 0
    sell_signals = 0
    
    # RSI analysis
    if rsi < 30:
        buy_signals += 2  # Strong buy signal
    elif rsi < 50:
        buy_signals += 1  # Mild buy signal
    elif rsi > 70:
        sell_signals += 2  # Strong sell signal
    elif rsi > 60:
        sell_signals += 1  # Mild sell signal
    
    # MACD analysis
    if current_macd > current_signal:
        buy_signals += 1
    else:
        sell_signals += 1
    
    # Moving average analysis
    if current_price > ma20 > ma50:
        buy_signals += 2
    elif current_price > ma20:
        buy_signals += 1
    elif current_price < ma20 < ma50:
        sell_signals += 2
    else:
        sell_signals += 1
    
    # Bollinger Bands analysis
    if bb_position < 0.2:
        buy_signals += 1
    elif bb_position > 0.8:
        sell_signals += 1
    
    # Generate recommendation
    total_signals = buy_signals + sell_signals
    buy_percentage = (buy_signals / total_signals * 100) if total_signals > 0 else 50
    
    if buy_signals > sell_signals + 2:
        recommendation = "🟢 **STRONG BUY**"
        reason = "Multiple bullish indicators align"
    elif buy_signals > sell_signals:
        recommendation = "📈 **BUY**"
        reason = "Bullish signals outweigh bearish ones"
    elif sell_signals > buy_signals + 2:
        recommendation = "🔴 **STRONG SELL**"
        reason = "Multiple bearish indicators present"
    elif sell_signals > buy_signals:
        recommendation = "📉 **SELL**"
        reason = "Bearish signals dominate"
    else:
        recommendation = "🟡 **HOLD**"
        reason = "Mixed signals, wait for clearer trend"
    
    analysis = f"""
    {recommendation}
    
    **Confidence Level**: {max(buy_percentage, 100-buy_percentage):.0f}%
    **Reasoning**: {reason}
    
    **Signal Breakdown**:
    - Buy Signals: {buy_signals}
    - Sell Signals: {sell_signals}
    
    **Key Factors**:
    - RSI: {rsi:.1f} ({'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'})
    - MACD: {'Bullish' if current_macd > current_signal else 'Bearish'}
    - Trend: {'Bullish' if current_price > ma20 else 'Bearish'}
    - BB Position: {bb_position:.2f} ({'Near Support' if bb_position < 0.3 else 'Near Resistance' if bb_position > 0.7 else 'Middle Range'})
    """
    
    return analysis

# --- Enhanced Custom CSS ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .developer-info {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 30px;
        transition: all 0.3s ease;
        font-weight: 600;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
        padding: 0.75rem 1.5rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        padding: 1.2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
</style>
""", unsafe_allow_html=True)

# --- Enhanced Header ---
st.markdown('<div class="main-header"><h1>📈 TradingLite</h1><p>Advanced Stock Analysis & Trading Intelligence Platform</p></div>', unsafe_allow_html=True)

# --- Developer Info ---
st.markdown('<div class="developer-info">👨‍💻 Developed by: Teshin Bubpha | TNI-NDR-2213111129</div>', unsafe_allow_html=True)

# --- Enhanced Sidebar Configuration ---
with st.sidebar:
    st.markdown("### 🎛️ Trading Control Panel")
    
    # Stock Symbol Section
    st.markdown("#### 📊 Stock Selection")
    symbol_input = st.text_input("Enter Stock Symbol", value="AAPL", placeholder="e.g., AAPL, GOOGL, TSLA")
    
    # Symbol validation with enhanced feedback
    if symbol_input:
        with st.spinner("🔍 Validating symbol..."):
            if validate_symbol(symbol_input):
                st.success(f"✅ {symbol_input.upper()} is valid and ready for analysis")
                symbol = symbol_input.upper()
            else:
                st.error(f"❌ {symbol_input.upper()} not found in market data")
                symbol = "AAPL"
    else:
        symbol = "AAPL"
    
    st.markdown("---")
    
    # Enhanced Time Configuration
    st.markdown("#### ⏰ Time Analysis Settings")
    
    time_periods = {
        "1 Day": "1d", "5 Days": "5d", "1 Month": "1mo", "3 Months": "3mo",
        "6 Months": "6mo", "YTD": "ytd", "1 Year": "1y", "2 Years": "2y", "Maximum Available": "max"
    }
    period_display = st.selectbox("Analysis Time Period", list(time_periods.keys()), index=4)
    period = time_periods[period_display]
    
    data_frequencies = {
        "1 Minute": "1m", "5 Minutes": "5m", "15 Minutes": "15m", "1 Hour": "60m",
        "Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"
    }
    frequency_display = st.selectbox("Data Frequency", list(data_frequencies.keys()), index=4)
    interval = data_frequencies[frequency_display]
    
    st.markdown("---")
    
    # Enhanced Chart Configuration
    st.markdown("#### 📈 Chart Visualization")
    chart_type = st.selectbox("Chart Display Type", ["Candlestick", "Line Chart"], index=0)
    
    # Enhanced Technical Indicators
    st.markdown("#### 🔧 Technical Analysis Tools")
    
    col1, col2 = st.columns(2)
    with col1:
        show_volume = st.checkbox("📊 Volume", value=True)
        show_ma = st.checkbox("📈 Moving Averages", value=True)
        show_bollinger = st.checkbox("🎯 Bollinger Bands", value=False)
    
    with col2:
        show_rsi = st.checkbox("⚡ RSI Indicator", value=False)
        show_macd = st.checkbox("📊 MACD Signal", value=False)

    
    # Moving Average Configuration
    if show_ma:
        st.markdown("**Moving Average Settings**")
        ma_periods = st.multiselect("Period Selection", [5, 10, 20, 50, 100, 200], default=[20, 50])
        ma_type = st.selectbox("Average Type", ["Simple Moving Average (SMA)", "Exponential Moving Average (EMA)"], index=0)
        ma_type = "SMA" if "Simple" in ma_type else "EMA"
    else:
        ma_periods = []
        ma_type = "SMA"
    
    # Visual Settings
    st.markdown("#### 🎨 Display Settings")
    chart_height = st.slider("Chart Height", 6, 20, 12)

# --- Enhanced Main Content ---
tab1, tab2, tab3 = st.tabs(["📊 Chart Analysis", "📈 Technical Indicators", "📋 Market Data"])

with tab1:
    # Enhanced Data Loading
    with st.spinner("🔄 Loading market data and performing analysis..."):
        df = get_stock_data_mpf(symbol, period, interval)
        stock_info = get_stock_info(symbol)

    if df.empty:
        st.error(f"❌ Unable to retrieve market data for {symbol}. Please verify the symbol is correct and try again.")
    else:
        # Enhanced Stock Information Display
        latest_data = df.iloc[-1]
        previous_data = df.iloc[-2] if len(df) > 1 else latest_data
        
        price_change = latest_data['Close'] - previous_data['Close']
        price_change_pct = (price_change / previous_data['Close']) * 100
        
        if stock_info:
            st.success(f"📈 {stock_info['name']} ({symbol}) | Sector: {stock_info['sector']} | Market Cap: ${format_number(stock_info['market_cap'])}")
        
        # Enhanced Key Metrics Dashboard
        st.markdown("### 💹 Live Market Dashboard")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("💰 Current Price", f"${latest_data['Close']:.2f}", f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
        
        with col2:
            volume_change = ((latest_data['Volume'] - previous_data['Volume']) / previous_data['Volume'] * 100) if previous_data['Volume'] != 0 else 0
            st.metric("📊 Volume", format_number(latest_data['Volume']), f"{volume_change:+.1f}%")
        
        with col3:
            st.metric("📈 Day High", f"${latest_data['High']:.2f}")
        
        with col4:
            st.metric("📉 Day Low", f"${latest_data['Low']:.2f}")
        
        with col5:
            st.metric("🎯 Opening", f"${latest_data['Open']:.2f}")
        
        with col6:
            day_range = latest_data['High'] - latest_data['Low']
            st.metric("📏 Day Range", f"${day_range:.2f}")
        
        # Enhanced Chart Section
        st.markdown(f"### 📊 {symbol} - {chart_type} Technical Analysis")
        
        # Enhanced Plotly Interactive Chart
        rows = 1
        if show_volume:
            rows += 1
        if show_rsi or show_macd:
            rows += 1
        
        row_heights = [0.65, 0.25, 0.1] if rows == 3 else ([0.75, 0.25] if rows == 2 else [1.0])
        
        fig = make_subplots(
            rows=rows, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.02,
            row_heights=row_heights,
            subplot_titles=[f'{symbol} Price Analysis'] + (['Volume Analysis'] if show_volume else []) + (['Technical Indicators'] if show_rsi or show_macd else [])
        )
        # Enhanced main price chart
        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                name=symbol, 
                increasing_line_color='#26a69a', 
                decreasing_line_color='#ef5350'
            ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Close'], mode='lines', name=f'{symbol} Close Price',
                line=dict(color='#2196F3', width=3)
            ), row=1, col=1)
        
        # Enhanced moving averages
        if show_ma and ma_periods:
            colors = ['#FF6B35', '#F7931E', '#4CAF50', '#2196F3', '#9C27B0', '#607D8B']
            for i, period_ma in enumerate(ma_periods):
                ma_data = df['Close'].rolling(window=period_ma).mean() if ma_type == "SMA" else df['Close'].ewm(span=period_ma).mean()
                fig.add_trace(go.Scatter(
                    x=df.index, y=ma_data, mode='lines', 
                    name=f'{ma_type} {period_ma}',
                    line=dict(color=colors[i % len(colors)], width=2)
                ), row=1, col=1)
        
        # Enhanced Bollinger Bands
        if show_bollinger:
            upper, middle, lower = calculate_bollinger_bands(df['Close'])
            fig.add_trace(go.Scatter(x=df.index, y=upper, mode='lines', name='BB Upper', 
                                   line=dict(color='rgba(255,69,0,0.5)', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=lower, mode='lines', name='BB Lower', 
                                   line=dict(color='rgba(255,69,0,0.5)', width=1), 
                                   fill='tonexty', fillcolor='rgba(255,69,0,0.1)'), row=1, col=1)
        
        # Enhanced Volume Chart
        current_row = 1
        if show_volume:
            current_row += 1
            colors = ['#26a69a' if close >= open else '#ef5350' for close, open in zip(df['Close'], df['Open'])]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Trading Volume', 
                               marker_color=colors, opacity=0.8), row=current_row, col=1)
        
        # Enhanced Technical indicators
        if show_rsi or show_macd:
            current_row += 1
            
            if show_rsi:
                rsi = calculate_rsi(df['Close'])
                fig.add_trace(go.Scatter(x=df.index, y=rsi, mode='lines', name='RSI (14)', 
                                       line=dict(color='#9C27B0', width=2)), row=current_row, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=current_row, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=current_row, col=1)
            
            if show_macd:
                macd, signal = calculate_macd(df['Close'])
                histogram = macd - signal
                fig.add_trace(go.Scatter(x=df.index, y=macd, mode='lines', name='MACD Line', 
                                       line=dict(color='#2196F3', width=2)), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=signal, mode='lines', name='Signal Line', 
                                       line=dict(color='#FF5722', width=2)), row=current_row, col=1)
        
        # Enhanced layout
        fig.update_layout(
            title=f'{symbol} - Comprehensive Technical Analysis Dashboard',
            xaxis_rangeslider_visible=False,
            height=chart_height * 80,
            template='plotly_white',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### 📈 Advanced Technical Analysis & Trading Intelligence")
    
    if not df.empty:
        # Enhanced Stock Summary
        st.markdown("#### 📋 Comprehensive Stock Analysis Report")
        summary = get_stock_summary(symbol, df, stock_info)
        st.markdown(summary)
        
        st.markdown("---")
        
        # Trading Recommendation Section
        st.markdown("#### 💡 Trading Recommendation")
        recommendation = get_trading_recommendation(df)
        st.markdown(recommendation)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Momentum Analysis")
            
            # Enhanced RSI Analysis
            rsi = calculate_rsi(df['Close'])
            current_rsi = rsi.iloc[-1]
            if current_rsi > 70:
                rsi_signal = "🔴 Overbought - Consider Selling"
            elif current_rsi < 30:
                rsi_signal = "🟢 Oversold - Consider Buying"
            else:
                rsi_signal = "🟡 Neutral Territory"
            st.metric("RSI (14-Period)", f"{current_rsi:.2f}", rsi_signal)
            
            # Enhanced MACD Analysis
            macd, signal = calculate_macd(df['Close'])
            current_macd = macd.iloc[-1]
            current_signal = signal.iloc[-1]
            macd_histogram = current_macd - current_signal
            
            if current_macd > current_signal:
                macd_trend = "📈 Bullish Signal"
            else:
                macd_trend = "📉 Bearish Signal"
            
            st.metric("MACD Signal", f"{current_macd:.4f}", macd_trend)
        
        with col2:
            st.markdown("#### 📈 Trend & Volatility Analysis")
            
            # Enhanced Bollinger Band Analysis
            upper, middle, lower = calculate_bollinger_bands(df['Close'])
            current_price = df['Close'].iloc[-1]
            current_upper = upper.iloc[-1]
            current_lower = lower.iloc[-1]
            
            bb_position = (current_price - current_lower) / (current_upper - current_lower)
            
            if bb_position > 0.8:
                bb_signal = "🔴 Near Upper Band"
            elif bb_position < 0.2:
                bb_signal = "🟢 Near Lower Band"  
            else:
                bb_signal = "🟡 Middle Range"
            
            st.metric("Bollinger Position", f"{bb_position:.2f}", bb_signal)
            
            # Volatility Analysis
            volatility = df['Close'].pct_change().std() * np.sqrt(252) * 100
            if volatility > 30:
                vol_signal = "🔴 High Volatility"
            elif volatility < 15:
                vol_signal = "🟢 Low Volatility"
            else:
                vol_signal = "🟡 Moderate Volatility"
            
            st.metric("Annual Volatility", f"{volatility:.1f}%", vol_signal)

with tab3:
    st.markdown("### 📋 Comprehensive Market Data Analysis")
    
    if not df.empty:
        # Enhanced Company Information
        if stock_info:
            st.markdown("#### 🏢 Company Information")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📊 Basic Info**")
                st.write(f"**Company**: {stock_info.get('name', 'N/A')}")
                st.write(f"**Symbol**: {symbol}")
                st.write(f"**Exchange**: {stock_info.get('exchange', 'N/A')}")
                st.write(f"**Currency**: {stock_info.get('currency', 'N/A')}")
                st.write(f"**Country**: {stock_info.get('country', 'N/A')}")
            
            with col2:
                st.markdown("**🏭 Business Details**")
                st.write(f"**Sector**: {stock_info.get('sector', 'N/A')}")
                st.write(f"**Industry**: {stock_info.get('industry', 'N/A')}")
                st.write(f"**Website**: {stock_info.get('website', 'N/A')}")
            
            with col3:
                st.markdown("**💰 Financial Metrics**")
                st.write(f"**Market Cap**: ${format_number(stock_info.get('market_cap', 0))}")
                st.write(f"**Enterprise Value**: ${format_number(stock_info.get('enterprise_value', 0))}")
                st.write(f"**P/E Ratio**: {stock_info.get('pe_ratio', 'N/A')}")
                st.write(f"**Beta**: {stock_info.get('beta', 'N/A')}")
                st.write(f"**Dividend Yield**: {stock_info.get('dividend_yield', 'N/A')}")
            
            # Business Summary
            if 'business_summary' in stock_info and stock_info['business_summary']:
                st.markdown("#### 📄 Business Summary")
                st.write(stock_info['business_summary'])
            
            st.markdown("---")
        
        # Enhanced Price Analysis
        st.markdown("#### 💹 Detailed Price Analysis")
        
        latest_data = df.iloc[-1]
        previous_data = df.iloc[-2] if len(df) > 1 else latest_data
        week_ago_data = df.iloc[-5] if len(df) > 5 else latest_data
        month_ago_data = df.iloc[-22] if len(df) > 22 else latest_data
        
        # Price Performance
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            daily_change = ((latest_data['Close'] - previous_data['Close']) / previous_data['Close']) * 100
            st.metric("📅 Daily Change", f"{daily_change:+.2f}%", 
                     f"${latest_data['Close'] - previous_data['Close']:+.2f}")
        
        with col2:
            weekly_change = ((latest_data['Close'] - week_ago_data['Close']) / week_ago_data['Close']) * 100
            st.metric("📅 Weekly Change", f"{weekly_change:+.2f}%",
                     f"${latest_data['Close'] - week_ago_data['Close']:+.2f}")
        
        with col3:
            monthly_change = ((latest_data['Close'] - month_ago_data['Close']) / month_ago_data['Close']) * 100
            st.metric("📅 Monthly Change", f"{monthly_change:+.2f}%",
                     f"${latest_data['Close'] - month_ago_data['Close']:+.2f}")
        
        with col4:
            ytd_high = df['High'].max()
            ytd_low = df['Low'].min()
            from_high = ((latest_data['Close'] - ytd_high) / ytd_high) * 100
            st.metric("📈 From Period High", f"{from_high:+.2f}%",
                     f"High: ${ytd_high:.2f}")
        
        # Volume Analysis
        st.markdown("#### 📊 Volume Analysis")
        col1, col2, col3, col4 = st.columns(4)
        
        avg_volume_10 = df['Volume'].tail(10).mean()
        avg_volume_50 = df['Volume'].tail(50).mean() if len(df) > 50 else avg_volume_10
        volume_vs_avg = ((latest_data['Volume'] - avg_volume_10) / avg_volume_10) * 100
        
        with col1:
            st.metric("📊 Current Volume", format_number(latest_data['Volume']))
        
        with col2:
            st.metric("📊 10-Day Avg Volume", format_number(avg_volume_10), 
                     f"{volume_vs_avg:+.1f}% vs avg")
        
        with col3:
            st.metric("📊 50-Day Avg Volume", format_number(avg_volume_50))
        
        with col4:
            max_volume = df['Volume'].max()
            st.metric("📊 Max Volume (Period)", format_number(max_volume))
        
        # Technical Levels
        st.markdown("#### 🎯 Key Technical Levels")
        col1, col2, col3 = st.columns(3)
        
        # Support and Resistance
        support_20 = df['Low'].rolling(20).min().iloc[-1]
        resistance_20 = df['High'].rolling(20).max().iloc[-1]
        
        # Pivot Points (Simple)
        high = latest_data['High']
        low = latest_data['Low']
        close = latest_data['Close']
        pivot = (high + low + close) / 3
        resistance_1 = (2 * pivot) - low
        support_1 = (2 * pivot) - high
        
        with col1:
            st.markdown("**🎯 Support Levels**")
            st.write(f"**20-Day Support**: ${support_20:.2f}")
            st.write(f"**Pivot Support S1**: ${support_1:.2f}")
            st.write(f"**Period Low**: ${df['Low'].min():.2f}")
        
        with col2:
            st.markdown("**🎯 Resistance Levels**")
            st.write(f"**20-Day Resistance**: ${resistance_20:.2f}")
            st.write(f"**Pivot Resistance R1**: ${resistance_1:.2f}")
            st.write(f"**Period High**: ${df['High'].max():.2f}")
        
        with col3:
            st.markdown("**🎯 Pivot Analysis**")
            st.write(f"**Pivot Point**: ${pivot:.2f}")
            position_vs_pivot = "Above" if close > pivot else "Below"
            st.write(f"**Position**: {position_vs_pivot} Pivot")
            st.write(f"**Distance**: ${abs(close - pivot):.2f}")
        
        st.markdown("---")
        
        # Recent Price Data with enhanced details
        st.markdown("#### 📊 Recent Trading Sessions")
        
        # Enhanced dataframe with additional columns
        recent_df = df.tail(10).copy()
        recent_df['Daily_Change'] = recent_df['Close'].pct_change() * 100
        recent_df['Daily_Range'] = recent_df['High'] - recent_df['Low']
        recent_df['Body_Size'] = abs(recent_df['Close'] - recent_df['Open'])
        recent_df['Upper_Shadow'] = recent_df['High'] - recent_df[['Open', 'Close']].max(axis=1)
        recent_df['Lower_Shadow'] = recent_df[['Open', 'Close']].min(axis=1) - recent_df['Low']
        
        # Format the dataframe
        display_df = recent_df[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Change', 'Daily_Range']].round(2)
        display_df['Volume'] = display_df['Volume'].apply(lambda x: format_number(x))
        display_df['Daily_Change'] = display_df['Daily_Change'].apply(lambda x: f"{x:+.2f}%")
        display_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily Change %', 'Daily Range']
        
        st.dataframe(display_df, use_container_width=True)
        
        # Statistical Summary with enhanced metrics
        st.markdown("#### 📈 Statistical Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Price Statistics**")
            summary_stats = df[['Open', 'High', 'Low', 'Close']].describe().round(2)
            st.dataframe(summary_stats, use_container_width=True)
        
        with col2:
            st.markdown("**📊 Volume & Volatility**")
            
            # Calculate additional metrics
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100
            sharpe_approx = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
            max_drawdown = ((df['Close'] / df['Close'].expanding().max()) - 1).min() * 100
            
            vol_stats = pd.DataFrame({
                'Metric': ['Daily Volatility (%)', 'Annual Volatility (%)', 'Average Daily Return (%)', 
                          'Max Drawdown (%)', 'Approx Sharpe Ratio'],
                'Value': [f"{returns.std() * 100:.2f}", f"{volatility:.2f}", 
                         f"{returns.mean() * 100:.3f}", f"{max_drawdown:.2f}", f"{sharpe_approx:.2f}"]
            })
            
            st.dataframe(vol_stats, use_container_width=True, hide_index=True)
            
            # Volume statistics
            volume_stats = df['Volume'].describe()
            st.markdown("**📊 Volume Statistics**")
            vol_display = pd.DataFrame({
                'Metric': ['Average Volume', 'Median Volume', 'Max Volume', 'Min Volume'],
                'Value': [format_number(volume_stats['mean']), format_number(volume_stats['50%']),
                         format_number(volume_stats['max']), format_number(volume_stats['min'])]
            })
            st.dataframe(vol_display, use_container_width=True, hide_index=True)

# --- Footer ---
st.markdown("---")
st.markdown("#### 🔄 Application Info")
st.info("📈 TradingLite - Advanced Stock Analysis Platform")
st.markdown("Built with Streamlit, Plotly, and yfinance for comprehensive market analysis.")