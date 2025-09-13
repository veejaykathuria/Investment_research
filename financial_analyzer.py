import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import psycopg2
from datetime import datetime, timedelta

class FinancialAnalyzer:
    """
    A class to analyze financial data for stocks using Yahoo Finance
    """
    
    def __init__(self, ticker_symbol):
        """
        Initialize the analyzer with a ticker symbol
        
        Args:
            ticker_symbol (str): Stock ticker symbol (e.g., 'AAPL')
        """
        self.ticker_symbol = ticker_symbol.upper()
        self.ticker = yf.Ticker(self.ticker_symbol)
    
    def is_valid_ticker(self):
        """
        Check if the ticker symbol is valid
        
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            info = self.ticker.info
            return 'symbol' in info or 'shortName' in info
        except:
            return False
    
    def get_stock_info(self):
        """
        Get basic stock information
        
        Returns:
            dict: Stock information dictionary
        """
        try:
            return self.ticker.info
        except:
            return {}
    
    def get_price_data(self, period="1y"):
        """
        Get historical price data
        
        Args:
            period (str): Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        
        Returns:
            pandas.DataFrame: Historical price data
        """
        try:
            data = self.ticker.history(period=period)
            return data
        except:
            return pd.DataFrame()
    
    def get_key_metrics(self):
        """
        Get key financial metrics
        
        Returns:
            dict: Dictionary of key financial metrics
        """
        try:
            info = self.ticker.info
            
            # Extract key metrics
            metrics = {
                'trailingPE': info.get('trailingPE'),
                'forwardPE': info.get('forwardPE'),
                'pegRatio': info.get('pegRatio'),
                'priceToBook': info.get('priceToBook'),
                'priceToSalesTrailing12Months': info.get('priceToSalesTrailing12Months'),
                'returnOnEquity': info.get('returnOnEquity'),
                'returnOnAssets': info.get('returnOnAssets'),
                'profitMargins': info.get('profitMargins'),
                'operatingMargins': info.get('operatingMargins'),
                'grossMargins': info.get('grossMargins'),
                'currentRatio': info.get('currentRatio'),
                'quickRatio': info.get('quickRatio'),
                'debtToEquity': info.get('debtToEquity'),
                'freeCashflow': info.get('freeCashflow'),
                'operatingCashflow': info.get('operatingCashflow')
            }
            
            return metrics
        except:
            return {}
    
    def get_income_statement(self):
        """
        Get annual income statement
        
        Returns:
            pandas.DataFrame: Income statement data
        """
        try:
            return self.ticker.financials
        except:
            return pd.DataFrame()
    
    def get_balance_sheet(self):
        """
        Get annual balance sheet
        
        Returns:
            pandas.DataFrame: Balance sheet data
        """
        try:
            return self.ticker.balance_sheet
        except:
            return pd.DataFrame()
    
    def get_cash_flow(self):
        """
        Get annual cash flow statement
        
        Returns:
            pandas.DataFrame: Cash flow statement data
        """
        try:
            return self.ticker.cashflow
        except:
            return pd.DataFrame()
    
    def get_quarterly_financials(self):
        """
        Get quarterly financial statements
        
        Returns:
            dict: Dictionary containing quarterly statements
        """
        try:
            return {
                'income_statement': self.ticker.quarterly_financials,
                'balance_sheet': self.ticker.quarterly_balance_sheet,
                'cash_flow': self.ticker.quarterly_cashflow
            }
        except:
            return {
                'income_statement': pd.DataFrame(),
                'balance_sheet': pd.DataFrame(),
                'cash_flow': pd.DataFrame()
            }
    
    def calculate_financial_ratios(self):
        """
        Calculate additional financial ratios from statements
        
        Returns:
            dict: Dictionary of calculated ratios
        """
        try:
            income_stmt = self.get_income_statement()
            balance_sheet = self.get_balance_sheet()
            
            if income_stmt.empty or balance_sheet.empty:
                return {}
            
            ratios = {}
            
            # Get latest year data
            latest_income = income_stmt.iloc[:, 0] if not income_stmt.empty else pd.Series()
            latest_balance = balance_sheet.iloc[:, 0] if not balance_sheet.empty else pd.Series()
            
            # Calculate ratios
            if 'Total Revenue' in latest_income.index and 'Total Assets' in latest_balance.index:
                ratios['asset_turnover'] = latest_income['Total Revenue'] / latest_balance['Total Assets']
            
            if 'Net Income' in latest_income.index and 'Total Revenue' in latest_income.index:
                ratios['net_profit_margin'] = latest_income['Net Income'] / latest_income['Total Revenue']
            
            if 'Total Debt' in latest_balance.index and 'Total Assets' in latest_balance.index:
                ratios['debt_ratio'] = latest_balance['Total Debt'] / latest_balance['Total Assets']
            
            return ratios
        except:
            return {}
    
    def get_dividend_data(self):
        """
        Get dividend data
        
        Returns:
            pandas.DataFrame: Dividend data
        """
        try:
            return self.ticker.dividends
        except:
            return pd.DataFrame()
    
    def get_analyst_recommendations(self):
        """
        Get analyst recommendations
        
        Returns:
            pandas.DataFrame: Analyst recommendations
        """
        try:
            return self.ticker.recommendations
        except:
            return pd.DataFrame()
    
    def get_earnings_data(self):
        """
        Get earnings data
        
        Returns:
            dict: Dictionary containing earnings information
        """
        try:
            return {
                'earnings': self.ticker.earnings,
                'quarterly_earnings': self.ticker.quarterly_earnings
            }
        except:
            return {
                'earnings': pd.DataFrame(),
                'quarterly_earnings': pd.DataFrame()
            }


class ValuationAnalyzer(FinancialAnalyzer):
    """
    A child class of FinancialAnalyzer specialized for stock valuation calculations
    """
    
    def __init__(self, ticker_symbol):
        """
        Initialize the valuation analyzer
        
        Args:
            ticker_symbol (str): Stock ticker symbol
        """
        super().__init__(ticker_symbol)
    
    def get_db_connection(self):
        """Get database connection using environment variables"""
        database_url = os.environ.get('DATABASE_URL')
        if database_url:
            return psycopg2.connect(database_url)
        return None
    
    def get_fcf_assumptions(self, symbol):
        """Get FCF assumptions for a symbol from database"""
        try:
            conn = self.get_db_connection()
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
            # Note: Using print instead of st.error to avoid import dependency
            print(f"Error getting FCF assumptions: {e}")
            return {}
    
    def save_fcf_assumptions(self, symbol, assumptions):
        """Save FCF assumptions to database"""
        try:
            conn = self.get_db_connection()
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
            print(f"Error saving FCF assumptions: {e}")
            return False
    
    def get_fcf_elements(self):
        """Get FCF calculation elements from database"""
        try:
            conn = self.get_db_connection()
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
            print(f"Error getting FCF elements: {e}")
            return pd.DataFrame()
    
    def calculate_fcf_valuation(self, symbol, assumptions, cash_flow_data):
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
            print(f"Error calculating FCF valuation: {e}")
            return None
    
    def calculate_ri_valuation(self, symbol, assumptions, financial_data):
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
            print(f"Error calculating RI valuation: {e}")
            return None
    
    def save_valuation_results(self, symbol, valuation_type, valuation_result, current_price):
        """Save valuation results to database"""
        try:
            conn = self.get_db_connection()
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
            print(f"Error saving valuation results: {e}")
            return False
