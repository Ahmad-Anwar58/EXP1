from flask import Flask, request, jsonify
import requests
import json

app = Flask(CropIq)

# Replace this with the actual endpoint you’ll define in Streamlit
STREAMLIT_URL = "https://testit11.streamlit.app/send-data"
API_KEY = "BSBACropTool_2025_secret"  

@app.route('/receive-data', methods=['POST'])
def receive_data():
    data = request.get_json()

    # Send data to the Streamlit endpoint
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    try:
        r = requests.post(STREAMLIT_URL, headers=headers, json=data)
        return jsonify({"message": "Data forwarded to Streamlit", "status": r.status_code}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
