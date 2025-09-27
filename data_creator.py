import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sqlite3
import os

class HistoricalDataCreator:
    """Creates historical dataset for price optimization"""
    
    def __init__(self):
        self.data = None
        
    def generate_historical_data(self, products=5, days=365):
        """Generate synthetic historical sales data"""
        np.random.seed(42)
        
        # Product information
        products_list = [f"Product_{chr(65+i)}" for i in range(products)]
        base_prices = [100, 150, 200, 80, 120]
        cost_prices = [60, 90, 120, 50, 80]
        
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(days)]
        
        data = []
        for date in dates:
            for i, product in enumerate(products_list):
                # Seasonal effects
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
                
                # Price elasticity effect
                current_price = base_prices[i] * (0.8 + 0.4 * np.random.random())
                price_ratio = current_price / base_prices[i]
                elasticity_effect = max(0.1, 1.2 - 0.8 * price_ratio)  # Basic elasticity
                
                # Competitor price effect
                competitor_effect = 0.9 + 0.2 * np.random.random()
                
                # Random noise
                noise = 0.9 + 0.2 * np.random.random()
                
                # Calculate demand
                base_demand = 1000 * (i + 1)
                demand = int(base_demand * seasonal_factor * elasticity_effect * competitor_effect * noise)
                
                # Calculate revenue and profit
                revenue = demand * current_price
                profit = demand * (current_price - cost_prices[i])
                
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'product': product,
                    'price': round(current_price, 2),
                    'demand': max(10, demand),
                    'revenue': round(revenue, 2),
                    'cost_price': cost_prices[i],
                    'profit': round(profit, 2),
                    'weekday': date.weekday(),
                    'month': date.month,
                    'is_weekend': 1 if date.weekday() >= 5 else 0
                })
        
        self.data = pd.DataFrame(data)
        return self.data
    
    def save_to_sqlite(self, filename="price_optimization.db"):
        """Save data to SQLite database"""
        conn = sqlite3.connect(filename)
        self.data.to_sql('historical_sales', conn, if_exists='replace', index=False)
        
        # Create summary table
        summary = self.data.groupby('product').agg({
            'price': ['mean', 'std', 'min', 'max'],
            'demand': ['mean', 'std', 'min', 'max'],
            'revenue': 'sum',
            'profit': 'sum'
        }).round(2)
        summary.to_sql('product_summary', conn, if_exists='replace')
        
        conn.close()
        return filename
    
    def get_elasticity_estimates(self):
        """Calculate price elasticity estimates"""
        elasticity_data = []
        for product in self.data['product'].unique():
            product_data = self.data[self.data['product'] == product]
            if len(product_data) > 1:
                # Simple elasticity calculation
                price_changes = product_data['price'].pct_change()
                demand_changes = product_data['demand'].pct_change()
                elasticity = (demand_changes / price_changes).mean()
                elasticity_data.append({
                    'product': product,
                    'elasticity': round(elasticity, 4) if not np.isnan(elasticity) else -2.0,
                    'avg_price': round(product_data['price'].mean(), 2),
                    'avg_demand': round(product_data['demand'].mean(), 2)
                })
        return pd.DataFrame(elasticity_data)

# Generate and save data
if __name__ == "__main__":
    creator = HistoricalDataCreator()
    data = creator.generate_historical_data()
    print("Historical data generated:")
    print(data.head())
    
    db_file = creator.save_to_sqlite()
    print(f"Data saved to {db_file}")
    
    elasticity = creator.get_elasticity_estimates()
    print("\nPrice Elasticity Estimates:")
    print(elasticity)