import pandas as pd
import numpy as np
from datetime import datetime, timedelta

START_DATE = datetime(2025, 1, 1, 0, 0, 0)
END_DATE = datetime(2026, 1, 1, 0, 0, 0)
INTERVAL = timedelta(minutes=15)
CAP_POWER_ON = 72
MAX_PV_RATIO = 0.95

def get_sun_times(date):
    day_of_year = date.timetuple().tm_yday
    lat = 39.9042
    declination = 23.45 * np.sin(np.radians((360/365) * (day_of_year - 81)))
    hour_angle = np.degrees(np.arccos(-np.tan(np.radians(lat)) * np.tan(np.radians(declination))))
    sunrise_hour = 12 - hour_angle / 15
    sunset_hour = 12 + hour_angle / 15
    return sunrise_hour, sunset_hour

def get_daily_weather(date):
    month = date.month
    base_prob_rain = 0.15
    if month in [6, 7, 8]:
        base_prob_rain = 0.25
    elif month in [11, 12, 1]:
        base_prob_rain = 0.1
    if np.random.random() < base_prob_rain:
        return 'rainy'
    elif np.random.random() < 0.2:
        return 'cloudy'
    else:
        return 'sunny'

def generate_daily_data(date):
    dtimes = []
    ghi_values = []
    pv_values = []
    cap_values = []
    
    weather = get_daily_weather(date)
    sunrise_hour, sunset_hour = get_sun_times(date)
    
    max_ghi_base = 900.0
    if weather == 'cloudy':
        max_ghi_base = 500.0
    elif weather == 'rainy':
        max_ghi_base = 180.0
    
    max_ghi = max_ghi_base + np.random.normal(0, 40)
    pv_coeff = (CAP_POWER_ON * MAX_PV_RATIO - 0.5) / 1000
    
    for hour in range(24):
        for minute in [0, 15, 30, 45]:
            dtime = datetime(date.year, date.month, date.day, hour, minute, 0)
            dtimes.append(dtime.strftime('%Y-%m-%d %H:%M:%S'))
            
            current_hour = hour + minute / 60
            
            if current_hour < sunrise_hour or current_hour > sunset_hour:
                ghi = 0.0
                pv = 0.0
            else:
                day_length = sunset_hour - sunrise_hour
                t = (current_hour - sunrise_hour) / day_length
                curve = np.sin(np.pi * t) ** 2
                ghi_base = max_ghi * curve
                ghi_noise = np.random.normal(0, 25 if weather == 'sunny' else 15)
                ghi = max(0.0, min(1000.0, ghi_base + ghi_noise))
                
                pv = ghi * pv_coeff * (1 + np.random.normal(0, 0.03))
                pv = max(0.0, min(CAP_POWER_ON * MAX_PV_RATIO - 0.01, pv))
            
            ghi_values.append(round(ghi, 1))
            pv_values.append(round(pv, 2))
            cap_values.append(CAP_POWER_ON)
    
    return dtimes, ghi_values, pv_values, cap_values

def generate_data():
    all_dtimes = []
    all_ghi = []
    all_pv = []
    all_cap = []
    
    current_date = START_DATE.date()
    end_date = END_DATE.date()
    
    while current_date < end_date:
        dtimes, ghi, pv, cap = generate_daily_data(current_date)
        all_dtimes.extend(dtimes)
        all_ghi.extend(ghi)
        all_pv.extend(pv)
        all_cap.extend(cap)
        current_date += timedelta(days=1)
    
    df_weather = pd.DataFrame({'dtime': all_dtimes, 'GHI': all_ghi})
    df_pv = pd.DataFrame({'dtime': all_dtimes, 'pv_data': all_pv})
    df_cap = pd.DataFrame({'dtime': all_dtimes, 'cap_power_on': all_cap})
    
    df_weather.to_csv('weather_future.csv', index=False)
    df_pv.to_csv('pv_history.csv', index=False)
    df_cap.to_csv('cap_power_on.csv', index=False)
    
    print(f"Generated {len(all_dtimes)} data points for 2025")
    print(f"Weather data saved to weather_future.csv")
    print(f"PV data saved to pv_history.csv")
    print(f"Capacity data saved to cap_power_on.csv")

if __name__ == '__main__':
    generate_data()