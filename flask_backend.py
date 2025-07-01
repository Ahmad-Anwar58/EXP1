from flask import Flask, request, jsonify
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)
API_KEY = "BSBACropTool_2025_secret"  # Must match with mimic_sensor.py

@app.route("/receive-data", methods=["POST"])
def receive_data():
    if request.headers.get("Authorization") != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    data["timestamp"] = datetime.now().isoformat()

    csv_path = "real_time_data.csv"
    df_new = pd.DataFrame([data])

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(csv_path, index=False)
    return jsonify({"message": "Data saved"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
