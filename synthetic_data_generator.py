import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# --- CONFIGURATION ---
TRAIN_NUM_RECORDS = 10000  # Size of dataset
TEST_NUM_RECORDS = 3000
START_DATE = datetime(2018, 1, 1)
LOCATIONS = ['Block_A_North', 'Block_B_Valley', 'Block_C_Hill', 'Block_D_River']
SOIL_TYPES = {'Block_A_North': 'Sandy Loam', 'Block_B_Valley': 'Clay Loam', 
              'Block_C_Hill': 'Laterite', 'Block_D_River': 'Alluvial'}

# Optimal Pineapple Conditions (approximate)
OPT_TEMP = 25.0  # Celsius
OPT_RAIN = 100.0 # mm per month
OPT_HUMIDITY = 70.0 # %

def generate_synthetic_data(output_dir,num_records):
    data = []
    
    current_date = START_DATE
    
    print("Generating synthetic harvest data...")
    
    for i in range(num_records):
        # 1. Random Date (spread over 5 years)
        # We add random days to simulate irregular harvest schedules
        current_date += timedelta(days=np.random.randint(1, 3))
        if current_date > datetime.now():
            break
            
        # 2. Select Location
        location = np.random.choice(LOCATIONS)
        soil = SOIL_TYPES[location]
        
        # 3. Simulate Environmental Conditions (Growth Period Average)
        # We add "seasonality" using sine waves based on the month
        month_factor = np.sin((current_date.month / 12) * 2 * np.pi)
        
        # Temp: Summer is hotter. Noise added.
        avg_temp = 25 + (month_factor * 5) + np.random.normal(0, 1.5)
        
        # Rain: Wet season vs Dry season.
        avg_rain = 120 + (month_factor * 80) + np.random.normal(0, 20)
        avg_rain = max(0, avg_rain) # Rain can't be negative
        
        # Humidity
        avg_humidity = 70 + (month_factor * 10) + np.random.normal(0, 5)
        avg_humidity = min(100, max(40, avg_humidity))
        
        # 4. Simulate Yield (The Logic)
        # Start with a base yield (e.g., 50 tons/hectare normalized to a truckload kg)
        base_yield = 15000 # kg per harvest batch
        
        # Effect of deviations from optimal conditions
        temp_penalty = abs(avg_temp - OPT_TEMP) * 500  # Lose 500kg for every degree off
        rain_penalty = abs(avg_rain - OPT_RAIN) * 10   # Lose 10kg for every mm off
        
        # Soil Boost
        soil_bonus = 0
        if soil == 'Sandy Loam': soil_bonus = 1000 # Best soil
        elif soil == 'Laterite': soil_bonus = 500
        
        # Calculate final yield with some random noise (machinery error, pests, etc)
        final_yield = base_yield - temp_penalty - rain_penalty + soil_bonus + np.random.normal(0, 500)
        final_yield = max(1000, final_yield) # Minimum yield
        
        data.append([
            current_date.strftime('%Y-%m-%d'),
            location,
            soil,
            round(avg_temp, 1),
            round(avg_rain, 1),
            round(avg_humidity, 1),
            int(final_yield)
        ])

    # Create DataFrame
    columns = ['Harvest_Date', 'Location', 'Soil_Type', 'Avg_Temp_C', 'Avg_Rain_mm', 'Avg_Humidity', 'Yield_kg']
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    df.to_csv(output_dir, index=False)
    print(f"Success! Generated {len(df)} records. Saved to '{output_dir}'")
    return df

# Run it
train_df = generate_synthetic_data('data/forecasting_data/synthetic_train_dataset.csv', TRAIN_NUM_RECORDS)
test_df = generate_synthetic_data('data/forecasting_data/synthetic_test_dataset.csv', TEST_NUM_RECORDS)   
print(train_df.head())  
print(test_df.head())