import requests
import random
import time
import pandas as pd
from datetime import datetime
import os
from github import Github
from dotenv import load_dotenv  # ⬅️ NEW

# ========== LOAD ENV VARIABLES ==========
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # ⬅️ Loaded from .env

# ========== SIMULATION CONFIG ==========
FLASK_URL = "http://127.0.0.1:5000/receive-data"
REPO_NAME = "Ahmad-Anwar58/EXP1"
BRANCH = "main"
CSV_FILENAME = "real_time_data.csv"
API_KEY = "BSBACropTool_2025_secret"

# ========== SENSOR FIELDS ==========
def generate_sensor_data():
    return {
        "sensor_id": random.randint(1000, 9999),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "soil_moisture_%": round(random.uniform(10, 60), 2),
        "soil_pH": round(random.uniform(5.5, 8.5), 2),
        "temperature_C": round(random.uniform(15, 45), 2),
        "rainfall_mm": round(random.uniform(0, 20), 2),
        "humidity_%": round(random.uniform(30, 90), 2),
        "sunlight_hours": round(random.uniform(4, 12), 2),
        "NDVI_index": round(random.uniform(0.1, 0.9), 2),
        "latitude": 31.5204,
        "longitude": 74.3587
    }

# ========== PUSH TO GITHUB ==========
def push_to_github(data_dict):
    g = Github(GITHUB_TOKEN)
    print("🔐 Authenticated as:", g.get_user().login)
    repo = g.get_repo(REPO_NAME)

    try:
        contents = repo.get_contents(CSV_FILENAME, ref=BRANCH)
        csv_data = pd.read_csv(contents.download_url)
        df = pd.concat([csv_data, pd.DataFrame([data_dict])], ignore_index=True)
        new_csv = df.to_csv(index=False)
        repo.update_file(contents.path, "🔄 Update real_time_data.csv with new sensor values", new_csv, contents.sha, branch=BRANCH)
    except Exception as e:
        df = pd.DataFrame([data_dict])
        repo.create_file(CSV_FILENAME, "🚀 Create real_time_data.csv with initial sensor data", df.to_csv(index=False), branch=BRANCH)

# ========== SEND TO FLASK ==========
def send_to_flask(data_dict):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = requests.post(FLASK_URL, json=data_dict, headers=headers)
    print("Flask Response:", response.status_code, response.text)


def push_to_github(data_dict):
    g = Github(GITHUB_TOKEN)
    print("🔐 Authenticated as:", g.get_user().login)
    repo = g.get_repo(REPO_NAME)

    try:
        print("📥 Fetching file contents...")
        contents = repo.get_contents(CSV_FILENAME, ref=BRANCH)
        print("✅ File found in GitHub repo.")

        print("📊 Reading existing CSV...")
        csv_data = pd.read_csv(contents.download_url)

        print("➕ Appending new data...")
        df = pd.concat([csv_data, pd.DataFrame([data_dict])], ignore_index=True)
        new_csv = df.to_csv(index=False)

        print("⬆️ Updating file on GitHub...")
        repo.update_file(contents.path, "🔄 Update real_time_data.csv with new sensor values", new_csv, contents.sha, branch=BRANCH)
        print("✅ File successfully updated on GitHub.")

    except Exception as e:
        print("⚠️ Exception caught:", str(e))
        print("📄 File not found or error occurred, creating new one instead...")

        df = pd.DataFrame([data_dict])
        csv_content = df.to_csv(index=False)

        repo.create_file(CSV_FILENAME, "🚀 Create real_time_data.csv with initial sensor data", csv_content, branch=BRANCH)
        print("✅ New file created successfully on GitHub.")

# ========== MAIN LOOP ==========
if __name__ == "__main__":
    while True:
        sensor_data = generate_sensor_data()
        print("Sending Data:", sensor_data)

        send_to_flask(sensor_data)
        push_to_github(sensor_data)

        print("✅ Data sent to Flask and GitHub\n")
        time.sleep(120)
