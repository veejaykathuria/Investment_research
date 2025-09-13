import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from financial_analyzer import FinancialAnalyzer
from chart_utils import ChartUtils
from database_manager import DatabaseManager

# Page configuration
st.set_page_config(
    page_title="Tracked Stocks Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize database manager
@st.cache_resource
def get_database_manager():
    return DatabaseManager()

# Tracked stocks
TRACKED_STOCKS = {
    'TSLA': 'Tesla, Inc.',
    'META': 'Meta Platforms, Inc.', 
    'BAC': 'Bank of America Corporation',
    'JPM': 'JPMorgan Chase & Co.',
    'ING': 'ING Groep N.V.'
}

def fetch_and_save_stock_data(symbol, db_manager):
    """Fetch current stock data and save to database"""
    try:
        analyzer = FinancialAnalyzer(symbol)
        
        if not analyzer.is_valid_ticker():
            return None
        
        # Get and save stock info
        info = analyzer.get_stock_info()
        if info:
            db_manager.save_stock_info(symbol, info)
        
        # Get and save recent price data (last 30 days)
        price_data = analyzer.get_price_data("1mo")
        if not price_data.empty:
            db_manager.save_price_data(symbol, price_data)
        
        # Get and save metrics
        metrics = analyzer.get_key_metrics()
        if metrics:
            db_manager.save_stock_metrics(symbol, metrics)
        
        return {
            'symbol': symbol,
            'info': info,
            'price_data': price_data,
            'metrics': metrics
        }
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

def display_stock_overview(stock_data):
    """Display overview of a single stock"""
    if not stock_data:
        return
    
    symbol = stock_data['symbol']
    info = stock_data['info']
    price_data = stock_data['price_data']
    
    # Stock header
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader(f"{info.get('longName', symbol)} ({symbol})")
        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
    
    with col2:
        current_price = info.get('currentPrice', info.get('previousClose', 0))
        st.metric("Current Price", f"${current_price:.2f}")
    
    with col3:
        market_cap = info.get('marketCap', 0)
        if market_cap > 0:
            if market_cap >= 1e12:
                cap_display = f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                cap_display = f"${market_cap/1e9:.2f}B"
            else:
                cap_display = f"${market_cap/1e6:.2f}M"
            st.metric("Market Cap", cap_display)
    
    # Price chart
    if not price_data.empty:
        chart_utils = ChartUtils()
        price_chart = chart_utils.create_price_chart(price_data, symbol)
        st.plotly_chart(price_chart, use_container_width=True)

def display_comparison_dashboard(all_stock_data):
    """Display comparison dashboard for all tracked stocks"""
    st.subheader("📊 Tracked Stocks Comparison")
    
    # Overview table
    overview_data = []
    comparison_data = {}
    
    for stock_data in all_stock_data:
        if not stock_data:
            continue
            
        symbol = stock_data['symbol']
        info = stock_data['info']
        price_data = stock_data['price_data']
        
        current_price = info.get('currentPrice', info.get('previousClose', 0))
        market_cap = info.get('marketCap', 0)
        
        # Calculate 30-day change
        change_pct = 0
        if not price_data.empty and len(price_data) > 1:
            change_pct = ((price_data['Close'].iloc[-1] / price_data['Close'].iloc[0]) - 1) * 100
        
        # Market cap display
        if market_cap >= 1e12:
            cap_display = f"${market_cap/1e12:.2f}T"
        elif market_cap >= 1e9:
            cap_display = f"${market_cap/1e9:.2f}B"
        else:
            cap_display = f"${market_cap/1e6:.2f}M" if market_cap > 0 else "N/A"
        
        overview_data.append([
            symbol,
            info.get('longName', symbol),
            info.get('sector', 'N/A'),
            f"${current_price:.2f}",
            cap_display,
            f"{change_pct:.2f}%"
        ])
        
        # Store for comparison chart
        if not price_data.empty:
            comparison_data[symbol] = price_data
    
    # Display overview table
    if overview_data:
        overview_df = pd.DataFrame(
            overview_data, 
            columns=["Symbol", "Company", "Sector", "Price", "Market Cap", "30D Change"]
        )
        st.dataframe(overview_df, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    # Comparison chart
    if len(comparison_data) >= 2:
        st.subheader("💹 Price Performance Comparison (30 Days)")
        chart_utils = ChartUtils()
        comparison_chart = chart_utils.create_comparison_price_chart(comparison_data, "30 Days")
        if comparison_chart:
            st.plotly_chart(comparison_chart, use_container_width=True)

def main():
    st.title("📊 Tracked Stocks Dashboard")
    st.markdown("**Monitoring: Tesla, Meta, Bank of America, JPMorgan Chase, ING**")
    st.markdown("---")
    
    # Initialize database manager
    db_manager = get_database_manager()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Dashboard Controls")
        
        # Refresh data button
        if st.button("🔄 Refresh All Data", type="primary", use_container_width=True):
            st.session_state.refresh_data = True
        
        st.markdown("---")
        
        # View options
        view_mode = st.radio(
            "View Mode",
            ["Overview Dashboard", "Individual Stocks"],
            index=0
        )
        
        if view_mode == "Individual Stocks":
            selected_stock = st.selectbox(
                "Select Stock",
                options=list(TRACKED_STOCKS.keys()),
                format_func=lambda x: f"{x} - {TRACKED_STOCKS[x]}"
            )
        
        st.markdown("---")
        
        # Data status
        st.subheader("💾 Database Status")
        try:
            summary = db_manager.get_latest_data_summary()
            if summary:
                st.success(f"✅ Tracking {len(summary)} stocks")
                st.write("**Latest Updates:**")
                for stock in summary:
                    if stock['price_date']:
                        st.write(f"• {stock['symbol']}: {stock['price_date']}")
            else:
                st.warning("No data in database yet")
        except Exception as e:
            st.error(f"Database connection error: {e}")
    
    # Main content
    if view_mode == "Overview Dashboard":
        # Fetch data for all stocks if refresh requested or first load
        if st.session_state.get('refresh_data', True):
            with st.spinner("Fetching latest stock data..."):
                all_stock_data = []
                progress_bar = st.progress(0)
                
                for i, symbol in enumerate(TRACKED_STOCKS.keys()):
                    stock_data = fetch_and_save_stock_data(symbol, db_manager)
                    all_stock_data.append(stock_data)
                    progress_bar.progress((i + 1) / len(TRACKED_STOCKS))
                
                progress_bar.empty()
                st.session_state.all_stock_data = all_stock_data
                st.session_state.refresh_data = False
                st.success("✅ Data refreshed successfully!")
        
        # Display dashboard
        all_stock_data = st.session_state.get('all_stock_data', [])
        if all_stock_data:
            display_comparison_dashboard(all_stock_data)
        else:
            st.info("Click 'Refresh All Data' to load stock information")
    
    else:
        # Individual stock view
        if st.session_state.get('refresh_data', True) or st.button(f"🔄 Refresh {selected_stock}"):
            with st.spinner(f"Fetching data for {selected_stock}..."):
                stock_data = fetch_and_save_stock_data(selected_stock, db_manager)
                st.session_state[f'stock_data_{selected_stock}'] = stock_data
                if view_mode != "Overview Dashboard":
                    st.session_state.refresh_data = False
        
        # Display individual stock
        stock_data = st.session_state.get(f'stock_data_{selected_stock}')
        if stock_data:
            display_stock_overview(stock_data)
            
            # Show detailed metrics
            st.markdown("---")
            st.subheader("💰 Financial Metrics")
            
            metrics = stock_data['metrics']
            if metrics:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Valuation Metrics**")
                    valuation_data = [
                        ["P/E Ratio", f"{metrics.get('trailingPE'):.2f}" if metrics.get('trailingPE') else "N/A"],
                        ["P/B Ratio", f"{metrics.get('priceToBook'):.2f}" if metrics.get('priceToBook') else "N/A"],
                        ["P/S Ratio", f"{metrics.get('priceToSalesTrailing12Months'):.2f}" if metrics.get('priceToSalesTrailing12Months') else "N/A"]
                    ]
                    valuation_df = pd.DataFrame(valuation_data, columns=["Metric", "Value"])
                    st.dataframe(valuation_df, hide_index=True)
                
                with col2:
                    st.write("**Profitability Metrics**")
                    profitability_data = [
                        ["ROE", f"{metrics.get('returnOnEquity'):.2%}" if metrics.get('returnOnEquity') else "N/A"],
                        ["ROA", f"{metrics.get('returnOnAssets'):.2%}" if metrics.get('returnOnAssets') else "N/A"],
                        ["Profit Margin", f"{metrics.get('profitMargins'):.2%}" if metrics.get('profitMargins') else "N/A"]
                    ]
                    profitability_df = pd.DataFrame(profitability_data, columns=["Metric", "Value"])
                    st.dataframe(profitability_df, hide_index=True)
        else:
            st.info(f"Click 'Refresh {selected_stock}' to load stock information")

if __name__ == "__main__":
    main()