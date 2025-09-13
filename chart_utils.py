import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime

class ChartUtils:
    """
    Utility class for creating interactive charts for financial data analysis
    """
    
    def __init__(self):
        """Initialize ChartUtils with default styling"""
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
    
    def create_price_chart(self, price_data, ticker_symbol):
        """
        Create an interactive price chart with volume
        
        Args:
            price_data (pandas.DataFrame): Historical price data
            ticker_symbol (str): Stock ticker symbol
            
        Returns:
            plotly.graph_objects.Figure: Interactive price chart
        """
        try:
            # Create subplot with secondary y-axis for volume
            from plotly.subplots import make_subplots
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(f'{ticker_symbol} Stock Price', 'Volume'),
                row_width=[0.7, 0.3]
            )
            
            # Add price line chart
            fig.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=price_data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color=self.color_palette['primary'], width=2),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add high and low as fill area
            fig.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=price_data['High'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=price_data['Low'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(31, 119, 180, 0.1)',
                    name='High-Low Range',
                    hovertemplate='<b>High:</b> $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add volume bars
            colors = ['red' if price_data['Close'].iloc[i] < price_data['Open'].iloc[i] 
                     else 'green' for i in range(len(price_data))]
            
            fig.add_trace(
                go.Bar(
                    x=price_data.index,
                    y=price_data['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.6,
                    hovertemplate='<b>Date:</b> %{x}<br><b>Volume:</b> %{y:,.0f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title=f'{ticker_symbol} Stock Price Analysis',
                height=600,
                showlegend=True,
                hovermode='x unified',
                template='plotly_white'
            )
            
            # Update y-axes
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            
            return fig
            
        except Exception as e:
            # Return empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating price chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16, color="red")
            )
            fig.update_layout(
                title=f"Price Chart Error - {ticker_symbol}",
                height=400,
                template='plotly_white'
            )
            return fig
    
    def create_revenue_chart(self, income_statement, ticker_symbol):
        """
        Create a revenue and profit chart from income statement
        
        Args:
            income_statement (pandas.DataFrame): Income statement data
            ticker_symbol (str): Stock ticker symbol
            
        Returns:
            plotly.graph_objects.Figure: Revenue and profit chart
        """
        try:
            # Extract revenue and net income data
            revenue_row = None
            net_income_row = None
            
            # Look for revenue in different possible row names
            revenue_names = ['Total Revenue', 'Revenue', 'Net Sales', 'Sales']
            for name in revenue_names:
                if name in income_statement.index:
                    revenue_row = name
                    break
            
            # Look for net income in different possible row names
            income_names = ['Net Income', 'Net Income Common Stockholders', 'Net Income Applicable To Common Shares']
            for name in income_names:
                if name in income_statement.index:
                    net_income_row = name
                    break
            
            if not revenue_row and not net_income_row:
                return None
            
            # Get last 4 years of data
            years = income_statement.columns[:4]
            
            fig = go.Figure()
            
            if revenue_row:
                revenue_data = income_statement.loc[revenue_row, years] / 1e9  # Convert to billions
                fig.add_trace(
                    go.Bar(
                        x=[str(year.year) for year in years],
                        y=revenue_data,
                        name='Revenue',
                        marker_color=self.color_palette['primary'],
                        hovertemplate='<b>Year:</b> %{x}<br><b>Revenue:</b> $%{y:.2f}B<extra></extra>'
                    )
                )
            
            if net_income_row:
                net_income_data = income_statement.loc[net_income_row, years] / 1e9  # Convert to billions
                fig.add_trace(
                    go.Bar(
                        x=[str(year.year) for year in years],
                        y=net_income_data,
                        name='Net Income',
                        marker_color=self.color_palette['success'],
                        hovertemplate='<b>Year:</b> %{x}<br><b>Net Income:</b> $%{y:.2f}B<extra></extra>'
                    )
                )
            
            fig.update_layout(
                title=f'{ticker_symbol} Revenue and Profit Trends',
                xaxis_title='Year',
                yaxis_title='Amount (Billions $)',
                template='plotly_white',
                height=400,
                barmode='group'
            )
            
            return fig
            
        except Exception as e:
            return None
    
    def create_assets_liabilities_chart(self, balance_sheet, ticker_symbol):
        """
        Create assets vs liabilities chart from balance sheet
        
        Args:
            balance_sheet (pandas.DataFrame): Balance sheet data
            ticker_symbol (str): Stock ticker symbol
            
        Returns:
            plotly.graph_objects.Figure: Assets vs liabilities chart
        """
        try:
            # Look for assets and liabilities
            assets_row = None
            liabilities_row = None
            
            # Look for total assets
            assets_names = ['Total Assets', 'Total Asset', 'Assets']
            for name in assets_names:
                if name in balance_sheet.index:
                    assets_row = name
                    break
            
            # Look for total liabilities
            liabilities_names = ['Total Liabilities Net Minority Interest', 'Total Liabilities', 'Total Liab']
            for name in liabilities_names:
                if name in balance_sheet.index:
                    liabilities_row = name
                    break
            
            if not assets_row or not liabilities_row:
                return None
            
            # Get last 4 years of data
            years = balance_sheet.columns[:4]
            
            assets_data = balance_sheet.loc[assets_row, years] / 1e9  # Convert to billions
            liabilities_data = balance_sheet.loc[liabilities_row, years] / 1e9  # Convert to billions
            equity_data = assets_data - liabilities_data
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Bar(
                    x=[str(year.year) for year in years],
                    y=assets_data,
                    name='Total Assets',
                    marker_color=self.color_palette['info'],
                    hovertemplate='<b>Year:</b> %{x}<br><b>Assets:</b> $%{y:.2f}B<extra></extra>'
                )
            )
            
            fig.add_trace(
                go.Bar(
                    x=[str(year.year) for year in years],
                    y=liabilities_data,
                    name='Total Liabilities',
                    marker_color=self.color_palette['danger'],
                    hovertemplate='<b>Year:</b> %{x}<br><b>Liabilities:</b> $%{y:.2f}B<extra></extra>'
                )
            )
            
            fig.add_trace(
                go.Bar(
                    x=[str(year.year) for year in years],
                    y=equity_data,
                    name='Shareholders Equity',
                    marker_color=self.color_palette['success'],
                    hovertemplate='<b>Year:</b> %{x}<br><b>Equity:</b> $%{y:.2f}B<extra></extra>'
                )
            )
            
            fig.update_layout(
                title=f'{ticker_symbol} Assets, Liabilities & Equity',
                xaxis_title='Year',
                yaxis_title='Amount (Billions $)',
                template='plotly_white',
                height=400,
                barmode='group'
            )
            
            return fig
            
        except Exception as e:
            return None
    
    def create_cash_flow_chart(self, cash_flow, ticker_symbol):
        """
        Create cash flow chart showing operating, investing, and financing cash flows
        
        Args:
            cash_flow (pandas.DataFrame): Cash flow statement data
            ticker_symbol (str): Stock ticker symbol
            
        Returns:
            plotly.graph_objects.Figure: Cash flow chart
        """
        try:
            # Look for different cash flow components
            operating_cf_row = None
            investing_cf_row = None
            financing_cf_row = None
            
            # Operating cash flow
            operating_names = ['Operating Cash Flow', 'Cash Flow From Operating Activities', 'Total Cash From Operating Activities']
            for name in operating_names:
                if name in cash_flow.index:
                    operating_cf_row = name
                    break
            
            # Investing cash flow
            investing_names = ['Investing Cash Flow', 'Cash Flow From Investing Activities', 'Total Cash From Investing Activities']
            for name in investing_names:
                if name in cash_flow.index:
                    investing_cf_row = name
                    break
            
            # Financing cash flow
            financing_names = ['Financing Cash Flow', 'Cash Flow From Financing Activities', 'Total Cash From Financing Activities']
            for name in financing_names:
                if name in cash_flow.index:
                    financing_cf_row = name
                    break
            
            if not operating_cf_row and not investing_cf_row and not financing_cf_row:
                return None
            
            # Get last 4 years of data
            years = cash_flow.columns[:4]
            
            fig = go.Figure()
            
            if operating_cf_row:
                operating_data = cash_flow.loc[operating_cf_row, years] / 1e9  # Convert to billions
                fig.add_trace(
                    go.Bar(
                        x=[str(year.year) for year in years],
                        y=operating_data,
                        name='Operating CF',
                        marker_color=self.color_palette['success'],
                        hovertemplate='<b>Year:</b> %{x}<br><b>Operating CF:</b> $%{y:.2f}B<extra></extra>'
                    )
                )
            
            if investing_cf_row:
                investing_data = cash_flow.loc[investing_cf_row, years] / 1e9  # Convert to billions
                fig.add_trace(
                    go.Bar(
                        x=[str(year.year) for year in years],
                        y=investing_data,
                        name='Investing CF',
                        marker_color=self.color_palette['warning'],
                        hovertemplate='<b>Year:</b> %{x}<br><b>Investing CF:</b> $%{y:.2f}B<extra></extra>'
                    )
                )
            
            if financing_cf_row:
                financing_data = cash_flow.loc[financing_cf_row, years] / 1e9  # Convert to billions
                fig.add_trace(
                    go.Bar(
                        x=[str(year.year) for year in years],
                        y=financing_data,
                        name='Financing CF',
                        marker_color=self.color_palette['info'],
                        hovertemplate='<b>Year:</b> %{x}<br><b>Financing CF:</b> $%{y:.2f}B<extra></extra>'
                    )
                )
            
            # Add horizontal line at zero
            fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
            
            fig.update_layout(
                title=f'{ticker_symbol} Cash Flow Analysis',
                xaxis_title='Year',
                yaxis_title='Cash Flow (Billions $)',
                template='plotly_white',
                height=400,
                barmode='group'
            )
            
            return fig
            
        except Exception as e:
            return None
    
    def create_metrics_comparison_chart(self, metrics_data, ticker_symbol):
        """
        Create a radar chart for key financial metrics comparison
        
        Args:
            metrics_data (dict): Dictionary of financial metrics
            ticker_symbol (str): Stock ticker symbol
            
        Returns:
            plotly.graph_objects.Figure: Radar chart for metrics
        """
        try:
            # Define metrics for radar chart with their ideal ranges
            radar_metrics = {
                'ROE': {'value': metrics_data.get('returnOnEquity', 0), 'max': 0.3},
                'ROA': {'value': metrics_data.get('returnOnAssets', 0), 'max': 0.15},
                'Profit Margin': {'value': metrics_data.get('profitMargins', 0), 'max': 0.3},
                'Operating Margin': {'value': metrics_data.get('operatingMargins', 0), 'max': 0.25},
                'Current Ratio': {'value': metrics_data.get('currentRatio', 0), 'max': 3.0}
            }
            
            # Filter out None values and normalize to 0-100 scale
            categories = []
            values = []
            
            for metric, data in radar_metrics.items():
                if data['value'] is not None and data['value'] > 0:
                    categories.append(metric)
                    # Normalize to 0-100 scale based on max value
                    normalized_value = min((data['value'] / data['max']) * 100, 100)
                    values.append(normalized_value)
            
            if len(categories) < 3:  # Need at least 3 metrics for meaningful radar chart
                return None
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=ticker_symbol,
                line_color=self.color_palette['primary']
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title=f'{ticker_symbol} Financial Metrics Radar',
                template='plotly_white',
                height=400
            )
            
            return fig
            
        except Exception as e:
            return None
