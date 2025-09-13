import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import warnings
import psycopg2
import os
import json
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

# Database connection
@st.cache_resource
def get_db_connection():
    """Get database connection using environment variables"""
    database_url = os.environ.get('DATABASE_URL')
    if database_url:
        return psycopg2.connect(database_url)
    return None

def initialize_database_schema():
    """Initialize database schema if tables don't exist"""
    try:
        conn = get_db_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        
        # Create all required tables
        schema_sql = """
        -- Create table for tracked stocks
        CREATE TABLE IF NOT EXISTS tracked_stocks (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) UNIQUE NOT NULL,
            company_name VARCHAR(255) NOT NULL,
            sector VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Create table for stock financial data
        CREATE TABLE IF NOT EXISTS stock_data (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            data_date DATE NOT NULL,
            current_price DECIMAL(12,2),
            market_cap BIGINT,
            revenue BIGINT,
            net_income BIGINT,
            operating_cash_flow BIGINT,
            free_cash_flow BIGINT,
            total_assets BIGINT,
            total_liabilities BIGINT,
            shareholders_equity BIGINT,
            book_value_per_share DECIMAL(12,2),
            pe_ratio DECIMAL(8,2),
            pb_ratio DECIMAL(8,2),
            roe DECIMAL(6,4),
            roa DECIMAL(6,4),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, data_date)
        );

        -- Create table for FCF valuation assumptions
        CREATE TABLE IF NOT EXISTS fcf_assumptions (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            assumption_name VARCHAR(100) NOT NULL,
            assumption_value DECIMAL(12,4) NOT NULL,
            assumption_type VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, assumption_name, assumption_type)
        );

        -- Create table for FCF calculation elements
        CREATE TABLE IF NOT EXISTS fcf_elements (
            id SERIAL PRIMARY KEY,
            element_name VARCHAR(100) NOT NULL,
            element_description TEXT,
            is_addition BOOLEAN DEFAULT TRUE,
            is_active BOOLEAN DEFAULT TRUE,
            display_order INTEGER DEFAULT 0
        );

        -- Create table for RI valuation assumptions
        CREATE TABLE IF NOT EXISTS ri_assumptions (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            assumption_name VARCHAR(100) NOT NULL,
            assumption_value DECIMAL(12,4) NOT NULL,
            assumption_type VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Create table for valuation results
        CREATE TABLE IF NOT EXISTS valuation_results (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            valuation_type VARCHAR(20) NOT NULL,
            valuation_date DATE NOT NULL,
            intrinsic_value DECIMAL(12,2),
            current_price DECIMAL(12,2),
            upside_downside_percent DECIMAL(6,2),
            assumptions_used JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        cursor.execute(schema_sql)
        
        # Insert default data
        default_data_sql = """
        -- Insert tracked stocks
        INSERT INTO tracked_stocks (symbol, company_name, sector) VALUES
        ('TSLA', 'Tesla Inc', 'Automotive'),
        ('META', 'Meta Platforms Inc', 'Technology'),
        ('BAC', 'Bank of America Corporation', 'Financial Services'),
        ('JPM', 'JPMorgan Chase & Co', 'Financial Services'),
        ('ING', 'ING Groep N.V.', 'Financial Services')
        ON CONFLICT (symbol) DO NOTHING;

        -- Insert default FCF calculation elements
        INSERT INTO fcf_elements (element_name, element_description, is_addition, is_active, display_order) VALUES
        ('Operating Cash Flow', 'Cash generated from core business operations', TRUE, TRUE, 1),
        ('Capital Expenditures', 'Investment in property, plant, and equipment', FALSE, TRUE, 2),
        ('Acquisitions', 'Cash spent on business acquisitions', FALSE, TRUE, 3),
        ('Asset Sales', 'Cash received from selling assets', TRUE, TRUE, 4),
        ('Working Capital Changes', 'Changes in current assets and liabilities', FALSE, TRUE, 5),
        ('Stock-based Compensation', 'Non-cash compensation to employees', TRUE, TRUE, 6),
        ('Restructuring Costs', 'One-time restructuring expenses', FALSE, TRUE, 7),
        ('Tax Benefits', 'Tax shields and benefits', TRUE, TRUE, 8)
        ON CONFLICT DO NOTHING;
        """
        
        cursor.execute(default_data_sql)
        conn.commit()
        cursor.close()
        
        return True
        
    except Exception as e:
        st.error(f"Error initializing database schema: {e}")
        return False

def get_tracked_stocks():
    """Get list of tracked stocks from database with fallback"""
    # Static fallback for required tracked stocks
    fallback_stocks = {
        "Tesla Inc (TSLA)": "TSLA",
        "Meta Platforms Inc (META)": "META", 
        "Bank of America Corporation (BAC)": "BAC",
        "JPMorgan Chase & Co (JPM)": "JPM",
        "ING Groep N.V. (ING)": "ING"
    }
    
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT symbol, company_name FROM tracked_stocks ORDER BY symbol")
            stocks = cursor.fetchall()
            cursor.close()
            if stocks:
                return {f"{row[1]} ({row[0]})": row[0] for row in stocks}
        
        # Return fallback if database is empty or unavailable
        return fallback_stocks
    except Exception as e:
        st.warning(f"Using fallback stock list due to database issue: {e}")
        return fallback_stocks

def save_stock_data(symbol, analyzer):
    """Save stock data to database"""
    try:
        conn = get_db_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        
        # Get stock info and financial data
        info = analyzer.get_stock_info()
        income_stmt = analyzer.get_income_statement()
        balance_sheet = analyzer.get_balance_sheet()
        cash_flow = analyzer.get_cash_flow()
        
        # Extract key financial metrics
        current_price = info.get('currentPrice', info.get('previousClose', 0))
        market_cap = info.get('marketCap', 0)
        
        # Extract financial statement data (latest year)
        revenue = None
        net_income = None
        operating_cash_flow = None
        free_cash_flow = None
        total_assets = None
        total_liabilities = None
        shareholders_equity = None
        
        if not income_stmt.empty:
            if 'Total Revenue' in income_stmt.index:
                revenue = float(income_stmt.loc['Total Revenue'].iloc[0]) if pd.notnull(income_stmt.loc['Total Revenue'].iloc[0]) else None
            if 'Net Income' in income_stmt.index:
                net_income = float(income_stmt.loc['Net Income'].iloc[0]) if pd.notnull(income_stmt.loc['Net Income'].iloc[0]) else None
        
        if not cash_flow.empty:
            if 'Operating Cash Flow' in cash_flow.index:
                operating_cash_flow = float(cash_flow.loc['Operating Cash Flow'].iloc[0]) if pd.notnull(cash_flow.loc['Operating Cash Flow'].iloc[0]) else None
            if 'Free Cash Flow' in cash_flow.index:
                free_cash_flow = float(cash_flow.loc['Free Cash Flow'].iloc[0]) if pd.notnull(cash_flow.loc['Free Cash Flow'].iloc[0]) else None
        
        if not balance_sheet.empty:
            if 'Total Assets' in balance_sheet.index:
                total_assets = float(balance_sheet.loc['Total Assets'].iloc[0]) if pd.notnull(balance_sheet.loc['Total Assets'].iloc[0]) else None
            if 'Total Liabilities Net Minority Interest' in balance_sheet.index:
                total_liabilities = float(balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0]) if pd.notnull(balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0]) else None
            if 'Stockholders Equity' in balance_sheet.index:
                shareholders_equity = float(balance_sheet.loc['Stockholders Equity'].iloc[0]) if pd.notnull(balance_sheet.loc['Stockholders Equity'].iloc[0]) else None
        
        # Calculate additional metrics
        book_value_per_share = info.get('bookValue', 0)
        pe_ratio = info.get('trailingPE', 0)
        pb_ratio = info.get('priceToBook', 0)
        roe = info.get('returnOnEquity', 0)
        roa = info.get('returnOnAssets', 0)
        
        # Insert or update stock data
        insert_query = """
        INSERT INTO stock_data (
            symbol, data_date, current_price, market_cap, revenue, net_income,
            operating_cash_flow, free_cash_flow, total_assets, total_liabilities,
            shareholders_equity, book_value_per_share, pe_ratio, pb_ratio, roe, roa
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, data_date) 
        DO UPDATE SET
            current_price = EXCLUDED.current_price,
            market_cap = EXCLUDED.market_cap,
            revenue = EXCLUDED.revenue,
            net_income = EXCLUDED.net_income,
            operating_cash_flow = EXCLUDED.operating_cash_flow,
            free_cash_flow = EXCLUDED.free_cash_flow,
            total_assets = EXCLUDED.total_assets,
            total_liabilities = EXCLUDED.total_liabilities,
            shareholders_equity = EXCLUDED.shareholders_equity,
            book_value_per_share = EXCLUDED.book_value_per_share,
            pe_ratio = EXCLUDED.pe_ratio,
            pb_ratio = EXCLUDED.pb_ratio,
            roe = EXCLUDED.roe,
            roa = EXCLUDED.roa,
            created_at = CURRENT_TIMESTAMP
        """
        
        cursor.execute(insert_query, (
            symbol, datetime.now().date(), current_price, market_cap, revenue, net_income,
            operating_cash_flow, free_cash_flow, total_assets, total_liabilities,
            shareholders_equity, book_value_per_share, pe_ratio, pb_ratio, roe, roa
        ))
        
        conn.commit()
        cursor.close()
        return True
        
    except Exception as e:
        st.error(f"Error saving stock data: {e}")
        return False

def get_historical_stock_data(symbol, days=30):
    """Get historical stock data from database"""
    try:
        conn = get_db_connection()
        if not conn:
            return pd.DataFrame()
            
        cursor = conn.cursor()
        
        query = """
        SELECT data_date, current_price, market_cap, revenue, net_income, 
               operating_cash_flow, free_cash_flow, pe_ratio, pb_ratio
        FROM stock_data 
        WHERE symbol = %s AND data_date >= %s
        ORDER BY data_date DESC
        """
        
        start_date = datetime.now().date() - timedelta(days=days)
        cursor.execute(query, (symbol, start_date))
        
        data = cursor.fetchall()
        cursor.close()
        
        if data:
            df = pd.DataFrame(data, columns=[
                'Date', 'Price', 'Market Cap', 'Revenue', 'Net Income',
                'Operating CF', 'Free CF', 'P/E Ratio', 'P/B Ratio'
            ])
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error retrieving historical data: {e}")
        return pd.DataFrame()

def get_fcf_assumptions(symbol):
    """Get FCF assumptions for a symbol from database"""
    try:
        conn = get_db_connection()
        if not conn:
            return {}
            
        cursor = conn.cursor()
        cursor.execute("""
            SELECT assumption_name, assumption_value, assumption_type 
            FROM fcf_assumptions 
            WHERE symbol = %s
        """, (symbol,))
        
        assumptions = {}
        for row in cursor.fetchall():
            assumptions[row[0]] = {'value': float(row[1]), 'type': row[2]}
        
        cursor.close()
        return assumptions
        
    except Exception as e:
        st.error(f"Error getting FCF assumptions: {e}")
        return {}

def save_fcf_assumptions(symbol, assumptions):
    """Save FCF assumptions to database"""
    try:
        conn = get_db_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        
        for name, data in assumptions.items():
            cursor.execute("""
                INSERT INTO fcf_assumptions (symbol, assumption_name, assumption_value, assumption_type)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (symbol, assumption_name, assumption_type)
                DO UPDATE SET 
                    assumption_value = EXCLUDED.assumption_value,
                    updated_at = CURRENT_TIMESTAMP
            """, (symbol, name, data['value'], data['type']))
        
        conn.commit()
        cursor.close()
        return True
        
    except Exception as e:
        st.error(f"Error saving FCF assumptions: {e}")
        return False

def get_fcf_elements():
    """Get FCF calculation elements from database"""
    try:
        conn = get_db_connection()
        if not conn:
            return pd.DataFrame()
            
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, element_name, element_description, is_addition, is_active, display_order
            FROM fcf_elements 
            ORDER BY display_order
        """)
        
        data = cursor.fetchall()
        cursor.close()
        
        if data:
            return pd.DataFrame(data, columns=[
                'ID', 'Element', 'Description', 'Addition', 'Active', 'Order'
            ])
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error getting FCF elements: {e}")
        return pd.DataFrame()

def calculate_fcf_valuation(symbol, assumptions, cash_flow_data):
    """Calculate FCF-based intrinsic value with enhanced element management and two-step DCF"""
    try:
        # Get valuation method
        valuation_method = assumptions.get('valuation_method', {}).get('value', 'One-Step DCF')
        
        # Use total FCF from dynamic table if available
        total_fcf = assumptions.get('total_fcf', {}).get('value', None)
        fcf_elements_data = assumptions.get('fcf_elements_data', {}).get('value', [])
        
        if total_fcf is not None:
            # Use calculated FCF from dynamic table
            fcf_base = total_fcf
            element_adjustments = []
            
            # Create adjustments summary from table data with safe key access
            for elem in fcf_elements_data:
                try:
                    element_name = elem.get('Element', 'Unknown Element')
                    element_value = float(elem.get('Value', 0))
                    is_addition = elem.get('Addition', True)
                    
                    action = "+" if is_addition else "-"
                    element_adjustments.append(f"{action}{element_value/1e6:.1f}M from {element_name}")
                except (ValueError, TypeError, KeyError) as e:
                    element_adjustments.append(f"Invalid element data: {str(e)}")
            
        else:
            # Fallback to traditional calculation
            if cash_flow_data.empty:
                return None
                
            # Extract latest FCF
            fcf_current = None
            fcf_names = ['Free Cash Flow', 'Operating Cash Flow']
            
            for name in fcf_names:
                if name in cash_flow_data.index:
                    fcf_current = float(cash_flow_data.loc[name].iloc[0])
                    break
            
            if fcf_current is None or fcf_current <= 0:
                return None
            
            fcf_base = fcf_current
            element_adjustments = ["Using base FCF from financial statements"]
        
        # Ensure positive FCF
        if fcf_base <= 0:
            fcf_base = abs(fcf_base) * 0.1  # Convert to small positive value
        
        # Get assumptions
        discount_rate = assumptions.get('discount_rate', {}).get('value', 0.10)
        terminal_growth = assumptions.get('terminal_growth', {}).get('value', 0.02)
        projection_years = int(assumptions.get('projection_years', {}).get('value', 5))
        
        # Project future FCFs based on valuation method
        future_fcfs = []
        fcf = fcf_base
        
        if valuation_method == "One-Step DCF":
            # Single growth rate for all years
            growth_rate = assumptions.get('growth_rate', {}).get('value', 0.05)
            
            for year in range(1, projection_years + 1):
                fcf = fcf * (1 + growth_rate)
                pv_fcf = fcf / ((1 + discount_rate) ** year)
                future_fcfs.append(pv_fcf)
                
        else:  # Two-Step DCF
            # High growth phase followed by stable growth
            high_growth_rate = assumptions.get('growth_rate', {}).get('value', 0.08)
            high_growth_years = int(assumptions.get('high_growth_years', {}).get('value', 3))
            stable_growth_rate = assumptions.get('stable_growth_rate', {}).get('value', 0.03)
            
            # High growth phase
            for year in range(1, high_growth_years + 1):
                fcf = fcf * (1 + high_growth_rate)
                pv_fcf = fcf / ((1 + discount_rate) ** year)
                future_fcfs.append(pv_fcf)
            
            # Stable growth phase
            for year in range(high_growth_years + 1, projection_years + 1):
                fcf = fcf * (1 + stable_growth_rate)
                pv_fcf = fcf / ((1 + discount_rate) ** year)
                future_fcfs.append(pv_fcf)
        
        # Terminal value
        terminal_fcf = fcf * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        pv_terminal_value = terminal_value / ((1 + discount_rate) ** projection_years)
        
        # Total enterprise value
        enterprise_value = sum(future_fcfs) + pv_terminal_value
        
        # Get shares outstanding
        shares_outstanding = assumptions.get('shares_outstanding', {}).get('value', 1000000000)
        
        # Calculate intrinsic value per share
        intrinsic_value = enterprise_value / shares_outstanding
        
        return {
            'valuation_method': valuation_method,
            'current_fcf': fcf_base,
            'adjusted_fcf': fcf_base,
            'element_adjustments': element_adjustments,
            'projected_fcfs': future_fcfs,
            'terminal_value': terminal_value,
            'pv_terminal_value': pv_terminal_value,
            'enterprise_value': enterprise_value,
            'intrinsic_value': intrinsic_value,
            'assumptions': assumptions
        }
        
    except Exception as e:
        st.error(f"Error calculating FCF valuation: {e}")
        return None

def save_valuation_results(symbol, valuation_type, valuation_result, current_price):
    """Save valuation results to database"""
    try:
        conn = get_db_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        
        # Calculate upside/downside percentage
        intrinsic_value = valuation_result['intrinsic_value']
        upside_downside = ((intrinsic_value - current_price) / current_price) * 100
        
        # Convert assumptions to JSON
        assumptions_json = json.dumps(valuation_result['assumptions'])
        
        # Insert valuation result
        cursor.execute("""
            INSERT INTO valuation_results 
            (symbol, valuation_type, valuation_date, intrinsic_value, current_price, upside_downside_percent, assumptions_used)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (symbol, valuation_type, datetime.now().date(), intrinsic_value, current_price, upside_downside, assumptions_json))
        
        conn.commit()
        cursor.close()
        return True
        
    except Exception as e:
        st.error(f"Error saving valuation results: {e}")
        return False

def calculate_ri_valuation(symbol, assumptions, financial_data):
    """Calculate Residual Income valuation"""
    try:
        # Get book value and earnings data
        if financial_data.empty:
            return None
        
        # Extract required data
        current_roe = assumptions.get('current_roe', {}).get('value', 0.15)  # 15%
        cost_of_equity = assumptions.get('cost_of_equity', {}).get('value', 0.10)  # 10%
        growth_rate = assumptions.get('growth_rate', {}).get('value', 0.05)  # 5%
        book_value = assumptions.get('book_value_per_share', {}).get('value', 50)  # $50
        projection_years = int(assumptions.get('projection_years', {}).get('value', 5))
        
        # Calculate residual income
        ri_values = []
        bv = book_value
        
        for year in range(1, projection_years + 1):
            # Expected earnings
            expected_earnings = bv * current_roe
            # Required earnings
            required_earnings = bv * cost_of_equity
            # Residual income
            ri = expected_earnings - required_earnings
            # Present value of RI
            pv_ri = ri / ((1 + cost_of_equity) ** year)
            ri_values.append(pv_ri)
            
            # Update book value for next year
            bv = bv + expected_earnings * (1 - assumptions.get('payout_ratio', {}).get('value', 0.3))
        
        # Terminal residual income
        terminal_ri = ri_values[-1] * (1 + growth_rate) / (cost_of_equity - growth_rate)
        pv_terminal_ri = terminal_ri / ((1 + cost_of_equity) ** projection_years)
        
        # Calculate intrinsic value
        intrinsic_value = book_value + sum(ri_values) + pv_terminal_ri
        
        return {
            'book_value': book_value,
            'ri_values': ri_values,
            'terminal_ri': terminal_ri,
            'pv_terminal_ri': pv_terminal_ri,
            'intrinsic_value': intrinsic_value,
            'assumptions': assumptions
        }
        
    except Exception as e:
        st.error(f"Error calculating RI valuation: {e}")
        return None

def main():
    st.title("📈 Stock Financial Analysis Dashboard")
    st.markdown("---")
    
    # Initialize database schema on first run
    initialize_database_schema()
    
    # Sidebar for stock input and controls
    with st.sidebar:
        st.header("Stock Selection")
        
        # Get tracked stocks from database
        tracked_stocks = get_tracked_stocks()
        
        if tracked_stocks:
            # Stock selection dropdown
            stock_options = list(tracked_stocks.keys())
            default_index = 0
            
            # Try to find previously selected stock
            if st.session_state.ticker_symbol:
                for i, option in enumerate(stock_options):
                    if tracked_stocks[option] == st.session_state.ticker_symbol:
                        default_index = i
                        break
            
            selected_stock = st.selectbox(
                "Select Stock to Analyze",
                options=stock_options,
                index=default_index,
                help="Choose from our tracked stocks"
            )
            
            ticker_input = tracked_stocks[selected_stock]
        else:
            st.error("No tracked stocks found in database")
            ticker_input = None
        
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
            
            # Save stock data to database
            with st.spinner("Saving stock data..."):
                if save_stock_data(ticker_input, analyzer):
                    st.success(f"✅ Stock data saved for {ticker_input}")
                else:
                    st.warning("⚠️ Unable to save stock data to database")
            
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
            
            st.markdown("---")
            
            # FCF and RI Valuation Section
            st.subheader("🧮 Valuation Models")
            
            val_tab1, val_tab2 = st.tabs(["DCF/FCF Valuation", "Residual Income Valuation"])
            
            with val_tab1:
                st.write("**Free Cash Flow (FCF) Valuation Model**")
                
                # Valuation method selection
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    valuation_method = st.radio(
                        "Valuation Method",
                        ["One-Step DCF", "Two-Step DCF"],
                        help="One-step: Single growth rate for all years\nTwo-step: High growth phase followed by stable growth"
                    )
                
                with col2:
                    st.write("**Key Assumptions**")
                    discount_rate = st.number_input("Discount Rate (WACC) (%)", value=10.0, min_value=1.0, max_value=20.0) / 100
                    projection_years = st.number_input("Projection Years", value=5, min_value=3, max_value=10)
                    shares_outstanding = st.number_input("Shares Outstanding (millions)", value=1000.0, min_value=1.0) * 1e6
                
                with col3:
                    if valuation_method == "One-Step DCF":
                        st.write("**One-Step Growth Parameters**")
                        fcf_growth_rate = st.number_input("FCF Growth Rate (%)", value=5.0, min_value=-10.0, max_value=50.0) / 100
                        terminal_growth = st.number_input("Terminal Growth Rate (%)", value=2.0, min_value=0.0, max_value=5.0) / 100
                        high_growth_years = projection_years
                        stable_growth_rate = fcf_growth_rate
                    else:  # Two-Step DCF
                        st.write("**Two-Step Growth Parameters**")
                        fcf_growth_rate = st.number_input("High Growth Rate (%)", value=8.0, min_value=-10.0, max_value=50.0) / 100
                        high_growth_years = st.number_input("High Growth Years", value=3, min_value=1, max_value=projection_years-1)
                        stable_growth_rate = st.number_input("Stable Growth Rate (%)", value=3.0, min_value=-5.0, max_value=15.0) / 100
                        terminal_growth = st.number_input("Terminal Growth Rate (%)", value=2.0, min_value=0.0, max_value=5.0) / 100

                st.markdown("---")
                
                # Dynamic FCF Calculation Table
                st.write("**📊 FCF Calculation & Projections Table**")
                
                # Initialize session state for FCF table if not exists
                if 'fcf_elements_data' not in st.session_state:
                    fcf_elements = get_fcf_elements()
                    if not fcf_elements.empty:
                        st.session_state.fcf_elements_data = fcf_elements.to_dict('records')
                    else:
                        # Default FCF elements if database unavailable
                        st.session_state.fcf_elements_data = [
                            {'ID': 1, 'Element': 'Operating Cash Flow', 'Description': 'Cash from operations', 'Addition': True, 'Active': True, 'Order': 1, 'Value': 5000000000},
                            {'ID': 2, 'Element': 'Capital Expenditures', 'Description': 'Investment in PP&E', 'Addition': False, 'Active': True, 'Order': 2, 'Value': 1000000000},
                            {'ID': 3, 'Element': 'Asset Sales', 'Description': 'Cash from asset sales', 'Addition': True, 'Active': False, 'Order': 3, 'Value': 200000000},
                            {'ID': 4, 'Element': 'Working Capital Changes', 'Description': 'Changes in working capital', 'Addition': False, 'Active': True, 'Order': 4, 'Value': 100000000}
                        ]
                
                # Create dynamic table with editable cells
                fcf_data = st.session_state.fcf_elements_data.copy()
                
                # Table header
                header_cols = st.columns([0.5, 3, 2, 1, 1, 1.5, 1])
                with header_cols[0]:
                    st.write("**Include**")
                with header_cols[1]:
                    st.write("**FCF Element**")
                with header_cols[2]:
                    st.write("**Description**")
                with header_cols[3]:
                    st.write("**Type**")
                with header_cols[4]:
                    st.write("**Current Value ($M)**")
                with header_cols[5]:
                    st.write("**Growth Rate (%)**")
                with header_cols[6]:
                    st.write("**Actions**")
                
                # Display/edit each row
                for i, row in enumerate(fcf_data):
                    cols = st.columns([0.5, 3, 2, 1, 1, 1.5, 1])
                    
                    with cols[0]:
                        row['Active'] = st.checkbox("", value=row.get('Active', True), key=f"active_{i}")
                    
                    with cols[1]:
                        row['Element'] = st.text_input("", value=row.get('Element', ''), key=f"element_{i}", label_visibility="collapsed")
                    
                    with cols[2]:
                        row['Description'] = st.text_input("", value=row.get('Description', ''), key=f"desc_{i}", label_visibility="collapsed")
                    
                    with cols[3]:
                        row['Addition'] = st.selectbox("", [True, False], index=0 if row.get('Addition', True) else 1, 
                                                     format_func=lambda x: "Add" if x else "Subtract", key=f"type_{i}", label_visibility="collapsed")
                    
                    with cols[4]:
                        row['Value'] = st.number_input("", value=float(row.get('Value', 0))/1e6, key=f"value_{i}", 
                                                     min_value=-10000.0, max_value=50000.0, step=100.0, format="%.1f", label_visibility="collapsed") * 1e6
                    
                    with cols[5]:
                        row['GrowthRate'] = st.number_input("", value=float(row.get('GrowthRate', 5.0)), key=f"growth_{i}", 
                                                          min_value=-20.0, max_value=50.0, step=0.5, format="%.1f", label_visibility="collapsed") / 100
                    
                    with cols[6]:
                        if st.button("🗑️", key=f"delete_{i}", help="Delete row"):
                            fcf_data.pop(i)
                            st.session_state.fcf_elements_data = fcf_data
                            st.rerun()
                
                # Update session state
                st.session_state.fcf_elements_data = fcf_data
                
                # Add new row button
                col1, col2, col3 = st.columns([1, 1, 3])
                with col1:
                    if st.button("➕ Add FCF Element"):
                        new_element = {
                            'ID': len(fcf_data) + 1,
                            'Element': 'New Element',
                            'Description': 'Description',
                            'Addition': True,
                            'Active': True,
                            'Order': len(fcf_data) + 1,
                            'Value': 0,
                            'GrowthRate': 5.0
                        }
                        st.session_state.fcf_elements_data.append(new_element)
                        st.rerun()
                
                with col2:
                    if st.button("🔄 Reset to Defaults"):
                        del st.session_state.fcf_elements_data
                        st.rerun()
                
                # Calculate total current FCF from active elements
                total_fcf = 0
                active_elements = [elem for elem in fcf_data if elem['Active']]
                for elem in active_elements:
                    if elem['Addition']:
                        total_fcf += elem['Value']
                    else:
                        total_fcf -= elem['Value']
                
                st.info(f"**Current Total FCF: ${total_fcf/1e6:.1f}M** (from {len(active_elements)} active elements)")
                
                selected_elements = [elem['Element'] for elem in active_elements]
                
                # Calculate FCF Valuation
                if st.button("🔄 Calculate FCF Valuation", type="primary"):
                    cash_flow = analyzer.get_cash_flow()
                    
                    # Prepare assumptions including selected elements and valuation method
                    fcf_assumptions = {
                        'valuation_method': {'value': valuation_method, 'type': 'method'},
                        'growth_rate': {'value': fcf_growth_rate, 'type': 'growth_rate'},
                        'discount_rate': {'value': discount_rate, 'type': 'discount_rate'},
                        'terminal_growth': {'value': terminal_growth, 'type': 'terminal_value'},
                        'projection_years': {'value': projection_years, 'type': 'projection'},
                        'shares_outstanding': {'value': shares_outstanding, 'type': 'shares'},
                        'selected_elements': {'value': selected_elements, 'type': 'fcf_elements'},
                        'fcf_elements_data': {'value': active_elements, 'type': 'fcf_data'},
                        'total_fcf': {'value': total_fcf, 'type': 'base_fcf'}
                    }
                    
                    # Add two-step specific parameters if applicable
                    if valuation_method == "Two-Step DCF":
                        fcf_assumptions.update({
                            'high_growth_years': {'value': high_growth_years, 'type': 'high_growth_period'},
                            'stable_growth_rate': {'value': stable_growth_rate, 'type': 'stable_growth'}
                        })
                    
                    # Validation checks
                    validation_errors = []
                    
                    if discount_rate <= terminal_growth:
                        validation_errors.append("Discount rate must be greater than terminal growth rate")
                    
                    if shares_outstanding <= 0:
                        validation_errors.append("Shares outstanding must be greater than zero")
                    
                    if projection_years < 1:
                        validation_errors.append("Projection years must be at least 1")
                    
                    if total_fcf <= 0:
                        validation_errors.append("Total FCF must be positive for meaningful valuation")
                    
                    if valuation_method == "Two-Step DCF" and high_growth_years >= projection_years:
                        validation_errors.append("High growth years must be less than total projection years")
                    
                    if validation_errors:
                        for error in validation_errors:
                            st.error(f"❌ {error}")
                        st.stop()
                    
                    # Save assumptions to database
                    save_fcf_assumptions(ticker_input, fcf_assumptions)
                    
                    # Calculate valuation with selected elements
                    fcf_result = calculate_fcf_valuation(ticker_input, fcf_assumptions, cash_flow)
                    
                    if fcf_result:
                        st.success("✅ FCF Valuation Calculated")
                        
                        # Save valuation results to database
                        current_price = info.get('currentPrice', info.get('previousClose', 0))
                        if save_valuation_results(ticker_input, 'FCF', fcf_result, current_price):
                            st.info("💾 Valuation results saved to database")
                        
                        # Display valuation method and key results
                        st.info(f"**Valuation Method Used: {fcf_result.get('valuation_method', 'N/A')}**")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Base FCF", f"${fcf_result['current_fcf']/1e6:.1f}M")
                        
                        with col2:
                            st.metric("Enterprise Value", f"${fcf_result['enterprise_value']/1e6:.1f}M")
                        
                        with col3:
                            upside = ((fcf_result['intrinsic_value'] - current_price) / current_price) * 100
                            st.metric("Intrinsic Value", f"${fcf_result['intrinsic_value']:.2f}", f"{upside:.1f}%")
                        
                        with col4:
                            st.metric("Current Price", f"${current_price:.2f}")
                        
                        # Show FCF element breakdown
                        if fcf_result.get('element_adjustments'):
                            st.write("**📊 FCF Element Breakdown:**")
                            elements_col1, elements_col2 = st.columns(2)
                            
                            with elements_col1:
                                for i, adjustment in enumerate(fcf_result['element_adjustments']):
                                    if i % 2 == 0:
                                        st.write(f"• {adjustment}")
                            
                            with elements_col2:
                                for i, adjustment in enumerate(fcf_result['element_adjustments']):
                                    if i % 2 == 1:
                                        st.write(f"• {adjustment}")
                        
                        # Enhanced projection table
                        st.write("**📈 Year-by-Year FCF Projections**")
                        
                        years = list(range(1, len(fcf_result['projected_fcfs']) + 1))
                        base_fcf = fcf_result['current_fcf']
                        valuation_method = fcf_result.get('valuation_method', 'N/A')
                        
                        # Create detailed projection table
                        proj_data = []
                        cumulative_pv = 0
                        
                        for i, year in enumerate(years):
                            pv_fcf = fcf_result['projected_fcfs'][i]
                            cumulative_pv += pv_fcf
                            
                            # Calculate growth rate used for this year
                            if valuation_method == "Two-Step DCF":
                                high_growth_years = fcf_assumptions.get('high_growth_years', {}).get('value', 3)
                                if year <= high_growth_years:
                                    growth_rate = fcf_assumptions.get('growth_rate', {}).get('value', 0.08)
                                else:
                                    growth_rate = fcf_assumptions.get('stable_growth_rate', {}).get('value', 0.03)
                            else:
                                growth_rate = fcf_assumptions.get('growth_rate', {}).get('value', 0.05)
                            
                            proj_data.append({
                                'Year': year,
                                'Growth Rate': f"{growth_rate:.1%}",
                                'Future FCF ($M)': f"{(base_fcf * ((1 + growth_rate) ** year))/1e6:.1f}",
                                'PV of FCF ($M)': f"{pv_fcf/1e6:.1f}",
                                'Cumulative PV ($M)': f"{cumulative_pv/1e6:.1f}"
                            })
                        
                        # Add terminal value row
                        terminal_pv = fcf_result['pv_terminal_value']
                        cumulative_pv += terminal_pv
                        proj_data.append({
                            'Year': 'Terminal',
                            'Growth Rate': f"{fcf_assumptions.get('terminal_growth', {}).get('value', 0.02):.1%}",
                            'Future FCF ($M)': 'Perpetual',
                            'PV of FCF ($M)': f"{terminal_pv/1e6:.1f}",
                            'Cumulative PV ($M)': f"{cumulative_pv/1e6:.1f}"
                        })
                        
                        proj_df = pd.DataFrame(proj_data)
                        st.dataframe(proj_df, hide_index=True)
                        
                        # Chart of FCF projections
                        fcf_chart = go.Figure()
                        fcf_chart.add_trace(go.Bar(
                            x=years,
                            y=[fcf/1e6 for fcf in fcf_result['projected_fcfs']],
                            name='Projected FCF (PV)',
                            marker_color='lightblue'
                        ))
                        fcf_chart.update_layout(
                            title=f'{ticker_input} FCF Projections',
                            xaxis_title='Year',
                            yaxis_title='FCF Present Value ($ Millions)',
                            template='plotly_white'
                        )
                        st.plotly_chart(fcf_chart, use_container_width=True)
                    else:
                        st.error("❌ Unable to calculate FCF valuation")
            
            with val_tab2:
                st.write("**Residual Income (RI) Valuation Model**")
                
                # RI Assumptions Input
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Financial Assumptions**")
                    current_roe = st.number_input("Current ROE (%)", value=15.0, min_value=1.0, max_value=50.0) / 100
                    cost_of_equity = st.number_input("Cost of Equity (%)", value=10.0, min_value=1.0, max_value=20.0) / 100
                    ri_growth_rate = st.number_input("RI Growth Rate (%)", value=5.0, min_value=-10.0, max_value=20.0) / 100
                
                with col2:
                    st.write("**Company Data**")
                    book_value_per_share = st.number_input("Book Value per Share ($)", value=50.0, min_value=1.0)
                    payout_ratio = st.number_input("Dividend Payout Ratio (%)", value=30.0, min_value=0.0, max_value=100.0) / 100
                    ri_projection_years = st.number_input("RI Projection Years", value=5, min_value=3, max_value=10)
                
                # Calculate RI Valuation
                if st.button("🔄 Calculate RI Valuation", type="primary"):
                    balance_sheet = analyzer.get_balance_sheet()
                    
                    # Prepare assumptions
                    ri_assumptions = {
                        'current_roe': {'value': current_roe, 'type': 'roe'},
                        'cost_of_equity': {'value': cost_of_equity, 'type': 'cost_of_equity'},
                        'growth_rate': {'value': ri_growth_rate, 'type': 'growth_rate'},
                        'book_value_per_share': {'value': book_value_per_share, 'type': 'book_value'},
                        'payout_ratio': {'value': payout_ratio, 'type': 'payout_ratio'},
                        'projection_years': {'value': ri_projection_years, 'type': 'projection'}
                    }
                    
                    # Calculate valuation
                    ri_result = calculate_ri_valuation(ticker_input, ri_assumptions, balance_sheet)
                    
                    if ri_result:
                        st.success("✅ RI Valuation Calculated")
                        
                        # Save valuation results to database
                        current_price = info.get('currentPrice', info.get('previousClose', 0))
                        if save_valuation_results(ticker_input, 'RI', ri_result, current_price):
                            st.info("💾 Valuation results saved to database")
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Book Value", f"${ri_result['book_value']:.2f}")
                        
                        with col2:
                            st.metric("PV of Residual Income", f"${sum(ri_result['ri_values']):.2f}")
                        
                        with col3:
                            ri_upside = ((ri_result['intrinsic_value'] - current_price) / current_price) * 100
                            st.metric("RI Intrinsic Value", f"${ri_result['intrinsic_value']:.2f}", f"{ri_upside:.1f}%")
                        
                        # Show projected RIs
                        st.write("**Projected Residual Income (Present Value)**")
                        ri_years = list(range(1, len(ri_result['ri_values']) + 1))
                        ri_df = pd.DataFrame({
                            'Year': ri_years,
                            'PV of RI ($)': [ri for ri in ri_result['ri_values']]
                        })
                        st.dataframe(ri_df, hide_index=True)
                        
                        # Chart of RI projections
                        ri_chart = go.Figure()
                        ri_chart.add_trace(go.Bar(
                            x=ri_years,
                            y=ri_result['ri_values'],
                            name='Projected RI (PV)',
                            marker_color='lightgreen'
                        ))
                        ri_chart.update_layout(
                            title=f'{ticker_input} Residual Income Projections',
                            xaxis_title='Year',
                            yaxis_title='RI Present Value ($)',
                            template='plotly_white'
                        )
                        st.plotly_chart(ri_chart, use_container_width=True)
                    else:
                        st.error("❌ Unable to calculate RI valuation")
            
        except Exception as e:
            st.error(f"❌ Error analyzing {ticker_input}: {str(e)}")
            st.info("Please check the ticker symbol and try again.")
    
    elif not ticker_input and analyze_button:
        st.warning("⚠️ Please select a stock from the dropdown")
    
    else:
        # Welcome message
        st.info("👈 Select a stock from the sidebar to begin analysis")
        
        # Show tracked stocks info
        if tracked_stocks:
            st.subheader("📊 Available Stocks for Analysis")
            
            # Display tracked stocks in a table
            stock_df = pd.DataFrame([
                {"Symbol": symbol, "Company": name.split(" (")[0]} 
                for name, symbol in tracked_stocks.items()
            ])
            st.dataframe(stock_df, hide_index=True, use_container_width=True)
            
            st.info("💡 These stocks are monitored for FCF and RI valuation analysis")
        else:
            st.warning("No tracked stocks available")

if __name__ == "__main__":
    main()
