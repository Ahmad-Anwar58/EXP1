import requests
import random

# Simulated sensor data
data = {
    "Soil Moisture (%)": round(random.uniform(10, 60), 2),
    "Soil Temperature (°C)": round(random.uniform(15, 40), 2),
    "Wind Speed (km/h)": round(random.uniform(0, 25), 2),
    "Rainfall (mm)": round(random.uniform(0, 10), 2),
    "Pest Infestation Risk (%)": round(random.uniform(0, 100), 2),
    "Irrigation Efficiency (%)": round(random.uniform(60, 100), 2)
}

# Send to Flask server running locally
response = requests.post("http://127.0.0.1:5000/receive-data", json=data)

print("Sent:", data)
print("Status:", response.status_code)
print("Flask Response:", response.json())
