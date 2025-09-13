# Stock Financial Analysis Dashboard

## Overview

This is a Streamlit-based web application for comprehensive stock financial analysis. The dashboard provides real-time stock data visualization, technical analysis, and financial metrics using Yahoo Finance as the data source. Users can analyze stocks through interactive charts, price trends, volume analysis, and key financial indicators.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for rapid web app development
- **Layout**: Wide layout configuration with sidebar navigation
- **Visualization**: Plotly for interactive charts and graphs
- **State Management**: Streamlit session state for maintaining user inputs across interactions

### Backend Architecture
- **Data Processing**: Pandas for data manipulation and analysis
- **Financial Data**: Yahoo Finance (yfinance) API for real-time stock data
- **Analytics Engine**: Custom FinancialAnalyzer class for stock analysis logic
- **Chart Generation**: Dedicated ChartUtils class for creating interactive visualizations

### Core Components
- **app.py**: Main application entry point with Streamlit UI
- **financial_analyzer.py**: Core business logic for stock data analysis and validation
- **chart_utils.py**: Utility class for creating interactive financial charts with Plotly

### Data Flow
- User inputs stock ticker symbol through sidebar interface
- FinancialAnalyzer validates ticker and fetches data from Yahoo Finance
- ChartUtils generates interactive visualizations
- Results displayed in main dashboard area with real-time updates

### Design Patterns
- **Class-based Architecture**: Separation of concerns with dedicated classes for analysis and visualization
- **Error Handling**: Graceful handling of invalid tickers and API failures
- **Session Management**: Persistent state across user interactions

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **yfinance**: Yahoo Finance API wrapper for stock data
- **pandas**: Data manipulation and analysis
- **plotly**: Interactive charting and visualization
- **numpy**: Numerical computing support

### Data Sources
- **Yahoo Finance API**: Primary source for stock prices, financial data, and company information
- **Real-time Data**: Live stock quotes and historical price data

### Visualization
- **Plotly Graph Objects**: Advanced chart creation with subplots and custom styling
- **Plotly Express**: Simplified chart generation
- **Interactive Features**: Zoom, pan, hover tooltips, and dynamic updates