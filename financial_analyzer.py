import yfinance as yf
import pandas as pd
import numpy as np
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
