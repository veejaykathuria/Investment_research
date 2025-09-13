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

# Page configuration
st.set_page_config(
    page_title="Stock Financial Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'ticker_symbol' not in st.session_state:
    st.session_state.ticker_symbol = ""

def main():
    st.title("📈 Stock Financial Analysis Dashboard")
    st.markdown("---")
    
    # Sidebar for stock input and controls
    with st.sidebar:
        st.header("Stock Selection")
        
        # Stock symbol input
        ticker_input = st.text_input(
            "Enter Stock Symbol", 
            value=st.session_state.ticker_symbol,
            placeholder="e.g., AAPL, GOOGL, MSFT",
            help="Enter a valid stock ticker symbol"
        )
        if ticker_input:
            ticker_input = ticker_input.upper()
        
        # Time period selection
        time_periods = {
            "1 Month": "1mo",
            "3 Months": "3mo", 
            "6 Months": "6mo",
            "1 Year": "1y",
            "2 Years": "2y",
            "5 Years": "5y",
            "Max": "max"
        }
        
        selected_period = st.selectbox(
            "Select Time Period",
            options=list(time_periods.keys()),
            index=3  # Default to 1 Year
        )
        
        analyze_button = st.button("🔍 Analyze Stock", type="primary")
    
    # Main content area
    if analyze_button and ticker_input:
        st.session_state.ticker_symbol = ticker_input
        
        try:
            # Initialize analyzer
            analyzer = FinancialAnalyzer(ticker_input)
            chart_utils = ChartUtils()
            
            # Validate ticker
            if not analyzer.is_valid_ticker():
                st.error(f"❌ Invalid ticker symbol: {ticker_input}")
                st.stop()
            
            # Get basic info
            info = analyzer.get_stock_info()
            if not info:
                st.error(f"❌ Unable to fetch data for {ticker_input}")
                st.stop()
            
            # Display company header
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.subheader(f"{info.get('longName', ticker_input)} ({ticker_input})")
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {info.get('industry', 'N/A')}")
            
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
                    elif market_cap >= 1e6:
                        cap_display = f"${market_cap/1e6:.2f}M"
                    else:
                        cap_display = f"${market_cap:,.0f}"
                    st.metric("Market Cap", cap_display)
            
            st.markdown("---")
            
            # Stock Price Chart
            st.subheader("📊 Stock Price History")
            
            period = time_periods[selected_period]
            price_data = analyzer.get_price_data(period)
            
            if not price_data.empty:
                price_chart = chart_utils.create_price_chart(price_data, ticker_input)
                st.plotly_chart(price_chart, use_container_width=True)
                
                # Price statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    price_change = price_data['Close'].iloc[-1] - price_data['Close'].iloc[0]
                    price_change_pct = (price_change / price_data['Close'].iloc[0]) * 100
                    st.metric(
                        f"{selected_period} Change", 
                        f"${price_change:.2f}", 
                        f"{price_change_pct:.2f}%"
                    )
                
                with col2:
                    st.metric("52W High", f"${price_data['High'].max():.2f}")
                
                with col3:
                    st.metric("52W Low", f"${price_data['Low'].min():.2f}")
                
                with col4:
                    avg_volume = price_data['Volume'].mean()
                    if avg_volume >= 1e6:
                        vol_display = f"{avg_volume/1e6:.1f}M"
                    elif avg_volume >= 1e3:
                        vol_display = f"{avg_volume/1e3:.1f}K"
                    else:
                        vol_display = f"{avg_volume:.0f}"
                    st.metric("Avg Volume", vol_display)
            
            st.markdown("---")
            
            # Financial Metrics Table
            st.subheader("💰 Key Financial Metrics")
            
            metrics_data = analyzer.get_key_metrics()
            if metrics_data:
                # Create two columns for metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Valuation Metrics**")
                    valuation_data = [
                        ["P/E Ratio", f"{metrics_data.get('trailingPE', 'N/A'):.2f}" if metrics_data.get('trailingPE') else "N/A"],
                        ["Forward P/E", f"{metrics_data.get('forwardPE', 'N/A'):.2f}" if metrics_data.get('forwardPE') else "N/A"],
                        ["PEG Ratio", f"{metrics_data.get('pegRatio', 'N/A'):.2f}" if metrics_data.get('pegRatio') else "N/A"],
                        ["P/B Ratio", f"{metrics_data.get('priceToBook', 'N/A'):.2f}" if metrics_data.get('priceToBook') else "N/A"],
                        ["P/S Ratio", f"{metrics_data.get('priceToSalesTrailing12Months', 'N/A'):.2f}" if metrics_data.get('priceToSalesTrailing12Months') else "N/A"]
                    ]
                    valuation_df = pd.DataFrame(valuation_data, columns=["Metric", "Value"])
                    st.dataframe(valuation_df, hide_index=True)
                
                with col2:
                    st.write("**Profitability Metrics**")
                    profitability_data = [
                        ["ROE", f"{metrics_data.get('returnOnEquity', 'N/A'):.2%}" if metrics_data.get('returnOnEquity') else "N/A"],
                        ["ROA", f"{metrics_data.get('returnOnAssets', 'N/A'):.2%}" if metrics_data.get('returnOnAssets') else "N/A"],
                        ["Profit Margin", f"{metrics_data.get('profitMargins', 'N/A'):.2%}" if metrics_data.get('profitMargins') else "N/A"],
                        ["Operating Margin", f"{metrics_data.get('operatingMargins', 'N/A'):.2%}" if metrics_data.get('operatingMargins') else "N/A"],
                        ["Gross Margin", f"{metrics_data.get('grossMargins', 'N/A'):.2%}" if metrics_data.get('grossMargins') else "N/A"]
                    ]
                    profitability_df = pd.DataFrame(profitability_data, columns=["Metric", "Value"])
                    st.dataframe(profitability_df, hide_index=True)
            
            st.markdown("---")
            
            # Financial Statements
            st.subheader("📋 Financial Statements")
            
            # Tabs for different statements
            tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
            
            with tab1:
                income_stmt = analyzer.get_income_statement()
                if not income_stmt.empty:
                    st.write("**Annual Income Statement (in millions)**")
                    # Display only last 4 years and format numbers
                    income_display = income_stmt.iloc[:, :4].copy()
                    for col in income_display.columns:
                        income_display[col] = income_display[col].apply(
                            lambda x: f"${x/1e6:.1f}M" if pd.notnull(x) and x != 0 else "N/A"
                        )
                    st.dataframe(income_display)
                    
                    # Revenue and profit chart
                    if 'Total Revenue' in income_stmt.index:
                        revenue_chart = chart_utils.create_revenue_chart(income_stmt, ticker_input)
                        st.plotly_chart(revenue_chart, use_container_width=True)
                else:
                    st.warning("Income statement data not available")
            
            with tab2:
                balance_sheet = analyzer.get_balance_sheet()
                if not balance_sheet.empty:
                    st.write("**Annual Balance Sheet (in millions)**")
                    # Display only last 4 years and format numbers
                    balance_display = balance_sheet.iloc[:, :4].copy()
                    for col in balance_display.columns:
                        balance_display[col] = balance_display[col].apply(
                            lambda x: f"${x/1e6:.1f}M" if pd.notnull(x) and x != 0 else "N/A"
                        )
                    st.dataframe(balance_display)
                    
                    # Assets vs liabilities chart
                    assets_liab_chart = chart_utils.create_assets_liabilities_chart(balance_sheet, ticker_input)
                    if assets_liab_chart:
                        st.plotly_chart(assets_liab_chart, use_container_width=True)
                else:
                    st.warning("Balance sheet data not available")
            
            with tab3:
                cash_flow = analyzer.get_cash_flow()
                if not cash_flow.empty:
                    st.write("**Annual Cash Flow Statement (in millions)**")
                    # Display only last 4 years and format numbers
                    cash_display = cash_flow.iloc[:, :4].copy()
                    for col in cash_display.columns:
                        cash_display[col] = cash_display[col].apply(
                            lambda x: f"${x/1e6:.1f}M" if pd.notnull(x) and x != 0 else "N/A"
                        )
                    st.dataframe(cash_display)
                    
                    # Cash flow chart
                    cash_flow_chart = chart_utils.create_cash_flow_chart(cash_flow, ticker_input)
                    if cash_flow_chart:
                        st.plotly_chart(cash_flow_chart, use_container_width=True)
                else:
                    st.warning("Cash flow statement data not available")
            
        except Exception as e:
            st.error(f"❌ Error analyzing {ticker_input}: {str(e)}")
            st.info("Please check the ticker symbol and try again.")
    
    elif not ticker_input and analyze_button:
        st.warning("⚠️ Please enter a stock ticker symbol")
    
    else:
        # Welcome message
        st.info("👈 Enter a stock ticker symbol in the sidebar to begin analysis")
        
        # Sample suggestions
        st.subheader("💡 Popular Stocks to Analyze")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("AAPL - Apple"):
                st.session_state.ticker_symbol = "AAPL"
                st.rerun()
        
        with col2:
            if st.button("GOOGL - Alphabet"):
                st.session_state.ticker_symbol = "GOOGL"
                st.rerun()
        
        with col3:
            if st.button("MSFT - Microsoft"):
                st.session_state.ticker_symbol = "MSFT"
                st.rerun()
        
        with col4:
            if st.button("TSLA - Tesla"):
                st.session_state.ticker_symbol = "TSLA"
                st.rerun()

if __name__ == "__main__":
    main()
