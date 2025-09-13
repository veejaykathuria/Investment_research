import os
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
import logging

class DatabaseManager:
    """Manage database operations for stock data"""
    
    def __init__(self):
        """Initialize database connection"""
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self.engine = create_engine(self.database_url)
        
    def save_stock_info(self, symbol, info):
        """Save stock company information to database"""
        try:
            with self.engine.connect() as conn:
                # Update stock info
                query = text("""
                    UPDATE stocks 
                    SET company_name = :company_name,
                        sector = :sector,
                        industry = :industry,
                        market_cap = :market_cap,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE symbol = :symbol
                """)
                
                conn.execute(query, {
                    'symbol': symbol,
                    'company_name': info.get('longName', ''),
                    'sector': info.get('sector', ''),
                    'industry': info.get('industry', ''),
                    'market_cap': info.get('marketCap', 0) if info.get('marketCap') else None
                })
                conn.commit()
                
        except Exception as e:
            logging.error(f"Error saving stock info for {symbol}: {e}")
    
    def save_price_data(self, symbol, price_data):
        """Save historical price data to database"""
        try:
            if price_data.empty:
                return
                
            # Prepare data for insertion
            price_records = []
            for date, row in price_data.iterrows():
                price_records.append({
                    'symbol': symbol,
                    'date': date.date(),
                    'open_price': float(row['Open']) if pd.notnull(row['Open']) else None,
                    'high_price': float(row['High']) if pd.notnull(row['High']) else None,
                    'low_price': float(row['Low']) if pd.notnull(row['Low']) else None,
                    'close_price': float(row['Close']) if pd.notnull(row['Close']) else None,
                    'volume': int(row['Volume']) if pd.notnull(row['Volume']) else None
                })
            
            with self.engine.connect() as conn:
                # Insert price data (on conflict, update)
                for record in price_records:
                    query = text("""
                        INSERT INTO stock_prices (symbol, date, open_price, high_price, low_price, close_price, volume)
                        VALUES (:symbol, :date, :open_price, :high_price, :low_price, :close_price, :volume)
                        ON CONFLICT (symbol, date) 
                        DO UPDATE SET
                            open_price = EXCLUDED.open_price,
                            high_price = EXCLUDED.high_price,
                            low_price = EXCLUDED.low_price,
                            close_price = EXCLUDED.close_price,
                            volume = EXCLUDED.volume
                    """)
                    
                    conn.execute(query, record)
                conn.commit()
                
        except Exception as e:
            logging.error(f"Error saving price data for {symbol}: {e}")
    
    def save_stock_metrics(self, symbol, metrics):
        """Save financial metrics to database"""
        try:
            with self.engine.connect() as conn:
                query = text("""
                    INSERT INTO stock_metrics (
                        symbol, trailing_pe, forward_pe, peg_ratio, price_to_book, price_to_sales,
                        return_on_equity, return_on_assets, profit_margins, operating_margins, 
                        gross_margins, current_ratio, current_price
                    ) VALUES (
                        :symbol, :trailing_pe, :forward_pe, :peg_ratio, :price_to_book, :price_to_sales,
                        :return_on_equity, :return_on_assets, :profit_margins, :operating_margins,
                        :gross_margins, :current_ratio, :current_price
                    )
                """)
                
                conn.execute(query, {
                    'symbol': symbol,
                    'trailing_pe': metrics.get('trailingPE') if metrics.get('trailingPE') else None,
                    'forward_pe': metrics.get('forwardPE') if metrics.get('forwardPE') else None,
                    'peg_ratio': metrics.get('pegRatio') if metrics.get('pegRatio') else None,
                    'price_to_book': metrics.get('priceToBook') if metrics.get('priceToBook') else None,
                    'price_to_sales': metrics.get('priceToSalesTrailing12Months') if metrics.get('priceToSalesTrailing12Months') else None,
                    'return_on_equity': metrics.get('returnOnEquity') if metrics.get('returnOnEquity') else None,
                    'return_on_assets': metrics.get('returnOnAssets') if metrics.get('returnOnAssets') else None,
                    'profit_margins': metrics.get('profitMargins') if metrics.get('profitMargins') else None,
                    'operating_margins': metrics.get('operatingMargins') if metrics.get('operatingMargins') else None,
                    'gross_margins': metrics.get('grossMargins') if metrics.get('grossMargins') else None,
                    'current_ratio': metrics.get('currentRatio') if metrics.get('currentRatio') else None,
                    'current_price': metrics.get('currentPrice') if metrics.get('currentPrice') else None
                })
                conn.commit()
                
        except Exception as e:
            logging.error(f"Error saving metrics for {symbol}: {e}")
    
    def get_tracked_stocks(self):
        """Get list of tracked stocks from database"""
        try:
            with self.engine.connect() as conn:
                query = text("SELECT symbol, company_name FROM stocks ORDER BY symbol")
                result = conn.execute(query)
                return [{'symbol': row[0], 'name': row[1]} for row in result]
        except Exception as e:
            logging.error(f"Error getting tracked stocks: {e}")
            return []
    
    def get_latest_data_summary(self):
        """Get summary of latest data for all tracked stocks"""
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT s.symbol, s.company_name,
                           p.close_price, p.date as price_date,
                           m.current_price, m.recorded_at as metrics_date
                    FROM stocks s
                    LEFT JOIN LATERAL (
                        SELECT close_price, date 
                        FROM stock_prices 
                        WHERE symbol = s.symbol 
                        ORDER BY date DESC 
                        LIMIT 1
                    ) p ON true
                    LEFT JOIN LATERAL (
                        SELECT current_price, recorded_at
                        FROM stock_metrics 
                        WHERE symbol = s.symbol 
                        ORDER BY recorded_at DESC 
                        LIMIT 1
                    ) m ON true
                    ORDER BY s.symbol
                """)
                
                result = conn.execute(query)
                return [dict(row._mapping) for row in result]
        except Exception as e:
            logging.error(f"Error getting data summary: {e}")
            return []