import requests
import random
import time
from datetime import datetime
import uuid

# === Configuration ===
FLASK_URL = "http://127.0.0.1:5000/receive-data"  # or replace with your actual server IP
API_KEY = "BSBACropTool_2025_secret"  # 🔐 Replace if your Streamlit secret is different

# === Crop & Irrigation/Fertilizer Options ===
CROPS = ["Wheat", "Maize", "Rice", "Sugarcane", "Cotton"]
IRRIGATION = ["Drip", "Flood", "Sprinkler"]
FERTILIZER = ["Organic", "Synthetic", "Mixed"]
DISEASE_STATUS = ["Healthy", "Diseased", "Recovered"]

# === Sensor Data Generator ===
def generate_sensor_data():
    now = datetime.now()
    sowing_date = now.replace(month=max(1, now.month - 2))  # 2 months ago
    harvest_date = now.replace(month=min(12, now.month + 2))  # 2 months later
    total_days = (harvest_date - sowing_date).days

    data = {
        "farm_id": random.randint(1000, 9999),
        "crop_type": random.choice(CROPS),
        "soil_moisture_%": round(random.uniform(20, 80), 2),
        "soil_pH": round(random.uniform(5.5, 8.0), 2),
        "temperature_C": round(random.uniform(15, 45), 2),
        "rainfall_mm": round(random.uniform(0, 50), 2),
        "humidity_%": round(random.uniform(20, 100), 2),
        "sunlight_hours": round(random.uniform(4, 12), 2),
        "irrigation_type": random.choice(IRRIGATION),
        "fertilizer_type": random.choice(FERTILIZER),
        "pesticide_usage_ml": round(random.uniform(50, 200), 2),
        "sowing_date": sowing_date.strftime("%Y-%m-%d"),
        "harvest_date": harvest_date.strftime("%Y-%m-%d"),
        "total_days": total_days,
        "yield_kg_per_hectare": round(random.uniform(2000, 7000), 2),
        "sensor_id": str(uuid.uuid4())[:8],
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "latitude": round(random.uniform(30.0, 31.5), 6),
        "longitude": round(random.uniform(70.0, 74.0), 6),
        "NDVI_index": round(random.uniform(0.2, 0.9), 2),
        "crop_disease_status": random.choice(DISEASE_STATUS)
    }
    return data

# === Send Loop ===
def start_simulation():
    print("🚀 Starting real-time sensor data simulation...\n")
    while True:
        payload = generate_sensor_data()
        try:
            response = requests.post(
                FLASK_URL,
                json=payload,
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
            print(f"✅ Sent at {payload['timestamp']} | Status: {response.status_code}")
        except Exception as e:
            print(f"❌ Failed to send data: {e}")
        
        time.sleep(10)  # Wait 10 seconds before sending next reading

# === Entry Point ===
if __name__ == "__main__":
    start_simulation()
