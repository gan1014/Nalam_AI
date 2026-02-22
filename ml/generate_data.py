
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Configuration
START_DATE = '2015-01-01'
END_DATE = '2024-12-31'
DISTRICTS = [
    'Ariyalur', 'Chengalpattu', 'Chennai', 'Coimbatore', 'Cuddalore', 'Dharmapuri', 'Dindigul', 'Erode',
    'Kallakurichi', 'Kancheepuram', 'Karur', 'Krishnagiri', 'Madurai', 'Mayiladuthurai', 'Nagapattinam',
    'Namakkal', 'Nilgiris', 'Perambalur', 'Pudukkottai', 'Ramanathapuram', 'Ranipet', 'Salem', 'Sivaganga',
    'Tenkasi', 'Thanjavur', 'Theni', 'Thoothukudi', 'Tiruchirappalli', 'Tirunelveli', 'Tirupathur',
    'Tiruppur', 'Tiruvallur', 'Tiruvannamalai', 'Tiruvarur', 'Vellore', 'Viluppuram', 'Virudhunagar', 'Kanyakumari'
]
DISEASES = ['Dengue', 'Cholera', 'Leptospirosis', 'Malaria', 'Chikungunya']
COASTAL_DISTRICTS = ['Chennai', 'Cuddalore', 'Kanyakumari', 'Nagapattinam', 'Ramanathapuram', 'Thanjavur', 'Thoothukudi', 'Tiruvallur', 'Viluppuram']
URBAN_DISTRICTS = ['Chennai', 'Coimbatore', 'Madurai', 'Salem', 'Tiruchirappalli', 'Tirunelveli', 'Tiruppur', 'Vellore', 'Erode']

def generate_weather(date, district):
    """Generate realistic weather data based on season."""
    month = date.month
    
    # Base values
    temp_base = 30
    rain_base = 5
    humid_base = 65
    
    # Seasonality
    # Summer (Mar-May): Hot, dry
    if 3 <= month <= 5:
        temp_base += 5 + random.uniform(0, 5)
        rain_base += random.uniform(0, 20)
        humid_base -= 10
    
    # SW Monsoon (Jun-Sep): Windy, some rain
    elif 6 <= month <= 9:
        temp_base += 2
        rain_base += random.uniform(20, 100)
        humid_base += 5
        
    # NE Monsoon (Oct-Dec): Heavy rain, cooler
    elif 10 <= month <= 12:
        temp_base -= 2
        rain_base += random.uniform(50, 200)
        humid_base += 15
        
    # Winter (Jan-Feb): Cool, dry
    else:
        temp_base -= 4
        rain_base += random.uniform(0, 10)
        humid_base -= 5
        
    # Geographic modifiers
    if district in COASTAL_DISTRICTS:
        humid_base += 10
        temp_base -= 1
    if district == 'Nilgiris':
        temp_base -= 10
        rain_base += 50
        
    # Add noise
    temp = max(15, min(45, temp_base + random.normalvariate(0, 2)))
    rain = max(0, rain_base + random.normalvariate(0, 30))
    humid = max(20, min(100, humid_base + random.normalvariate(0, 5)))
    
    return round(temp, 1), round(rain, 1), round(humid, 1)

def generate_cases(disease, date, district, weather):
    """Generate disease cases based on seasonality, weather, and district characteristics."""
    temp, rain, humid = weather
    month = date.month
    
    base_cases = random.randint(0, 5)
    
    # Disease specific logic
    if disease == 'Dengue':
        # Peaks Oct-Nov (NE Monsoon), needs rain for breeding
        if 10 <= month <= 11:
            base_cases += random.randint(10, 50)
        elif 9 <= month <= 12:
            base_cases += random.randint(5, 20)
        
        if district in URBAN_DISTRICTS:
            base_cases = int(base_cases * 1.3)
            
    elif disease == 'Cholera':
        # Peaks May-Jun (Summer/Pre-monsoon), water scarcity/contamination
        if 5 <= month <= 6:
            base_cases += random.randint(5, 30)
            
    elif disease == 'Leptospirosis':
        # Peaks Oct-Dec (Flooding), rats
        if 10 <= month <= 12:
            base_cases += random.randint(5, 40)
        
        if district in COASTAL_DISTRICTS:
            base_cases = int(base_cases * 1.4)
            
    elif disease == 'Malaria':
        # Consistent but peaks with rain
        if rain > 50:
            base_cases += random.randint(5, 25)
            
        if district in COASTAL_DISTRICTS:
            base_cases = int(base_cases * 1.4)
            
    elif disease == 'Chikungunya':
        # Similar to Dengue
        if 10 <= month <= 12:
            base_cases += random.randint(5, 30)
            
        if district in URBAN_DISTRICTS:
            base_cases = int(base_cases * 1.3)
            
    # Outbreaks (5% chance)
    if random.random() < 0.05:
        base_cases = int(base_cases * random.uniform(2.5, 5.0))
        
    return max(0, int(base_cases))

def main():
    print("Generating synthetic dataset...")
    data = []
    
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='W-MON')
    total_steps = len(dates) * len(DISTRICTS)
    
    step = 0
    for date in dates:
        for district in DISTRICTS:
            # Generate weather once per district-week
            weather = generate_weather(date, district)
            
            for disease in DISEASES:
                cases = generate_cases(disease, date, district, weather)
                
                data.append({
                    'date': date,
                    'district': district,
                    'disease': disease,
                    'cases': cases,
                    'rainfall_mm': weather[1],
                    'temp_max': weather[0],
                    'humidity': weather[2]
                })
            
            step += 1
            if step % 1000 == 0:
                print(f"Progress: {step}/{total_steps} ({(step/total_steps)*100:.1f}%)", end='\r')
                
    df = pd.DataFrame(data)
    
    # Save
    output_path = 'nalamai/data/raw/tn_disease_surveillance.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✅ Dataset generated: {len(df)} rows. Saved to {output_path}")
    
    # Validation
    assert not df.isnull().values.any(), "Null values found!"
    assert (df['cases'] >= 0).all(), "Negative cases found!"
    assert len(df['district'].unique()) >= 35, "Missing districts!"
    print(f"✅ Data quality validation passed.")

if __name__ == "__main__":
    main()
