import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import plotly.express as px
import base64
import streamlit.components.v1 as components
import requests
from io import StringIO
import plotly.graph_objects as go
from chat2 import smart_query, setup_retrieval_qa


@st.cache_data(ttl=60)
def load_live_data():
    response = requests.get("https://raw.githubusercontent.com/Ahmad-Anwar58/EXP1/main/real_time_data.csv")
    response.raise_for_status()
    df_live = pd.read_csv(StringIO(response.text))
    df_live['timestamp'] = pd.to_datetime(df_live['timestamp'])
    return df_live.sort_values(by="timestamp", ascending=False)
    
# Set modern page config
st.set_page_config(page_title="CropIQ – Intelligent Crop Yield Optimizer", layout="wide")

# === Inject Background Image ===
@st.cache_data
def get_base64_image(image_path, version=2):  # version to break cache
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_path = "appbackground11.jpg"  # <-- UPDATE THIS
img_base64 = get_base64_image(img_path, version=2)


st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}
    .main .block-container {{
        background-color: Black;
        padding: 2rem;
        border-radius: 10px;
    }}
    /* Style for predicted results (text only) */
    .predicted-result, .agri-chat-reply {{
        color: black !important;
        font-weight: bold;
        font-size: 18px;
        background-color: rgba(255,255,255,0.85);
        padding: 12px 20px;
        border-radius: 10px;
        margin-top: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <style>
    /* Make form labels bold white but keep original size */
    label {
        color: white !important;
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Make form labels bold white */
    label {
        color: white !important;
        font-weight: bold !important;
    }

    /* Make all divs that show output values bold black */
    .stMarkdown, .stNumberInput, .stSlider, .stDataFrame, .stMetric {
        color: black !important;
        font-weight: bold !important;
    }

    /* For slider number limits */
    .stSlider > div > div > div > span {
        color: black !important;
        font-weight: bold !important;
    }

    /* Make any spans (like numbers, percentages) bold black */
    span {
        color: black !important;
        font-weight: bold !important;
    }

    /* Make sidebar titles like "Choose Module" black */
    section[data-testid="stSidebar"] .css-16huue1 {
        color: black !important;
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/Ahmad-Anwar58/EXP1/main/Data.csv")
    df['sowing_date'] = pd.to_datetime(df['sowing_date'])
    df['harvest_date'] = pd.to_datetime(df['harvest_date'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

df = load_data()

# Encode categoricals
label_encoders = {}
categorical_cols = ['crop_type', 'irrigation_type', 'fertilizer_type', 'crop_disease_status']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Remove non-numeric columns for default values and model training
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != 'yield_kg_per_hectare']

# Shared model data
default_values = df[numeric_cols].mean().to_dict()
X = df[numeric_cols]
y = df['yield_kg_per_hectare']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Sidebar Navigation
with st.sidebar:
    # Big, bold app title
    st.markdown("""
        <div style='font-size: 36px; font-weight: 900; color: black; margin-bottom: 10px;'>
            🌱 Crop IQ
        </div>
    """, unsafe_allow_html=True)

    # Custom label for module section
    st.markdown("""
        <div style='font-weight: 700; font-size: 17px; color: black; margin-bottom: 2px; margin-top: -10px;'>
            📂 Choose Module
        </div>
    """, unsafe_allow_html=True)

    # Radio buttons for module selection (no default label)
    menu_items = [
        "🏠 Home",
        "🌾 Yield Predictor",
        "💧 Irrigation Forecast",
        "🧪 Pesticide Estimator",
        "💰 ROI Calculator",
        "📊 Dashboard",
        "💬 AgriTech Chatbot 🤖",
        "📡 Live Sensor Data"
    ]
    section = st.radio("", menu_items, index=0)

    # CSS to style and reduce space between buttons
    st.markdown("""
        <style>
        /* Tighter space above button group */
        div[role="radiogroup"] {
            margin-top: -10px;
        }

        /* Hide default radio circle */
        div[role="radiogroup"] > label > div[role="presentation"] > div:first-child {
            display: none;
        }

        /* Button style for each label */
        div[role="radiogroup"] > label {
            background-color: #f0f2f6;
            border-radius: 15px;
            padding: 12px 20px;
            margin-bottom: 4px;  /* 🔧 This controls space between buttons */
            font-weight: 600;
            font-size: 16px;
            color: #111;
            cursor: pointer;
            user-select: none;
            transition: background-color 0.3s ease;
        }

        /* Hover effect */
        div[role="radiogroup"] > label:hover {
            background-color: #d1d9ff;
        }

        /* Active/selected button */
        div[role="radiogroup"] > label[aria-checked="true"] {
            background-color: #4a90e2;
            color: white;
            font-weight: 700;
        }
        </style>
    """, unsafe_allow_html=True)

# Home (Welcome Screen)
if section == "🏠 Home":
    st.markdown(
        """
        <!-- Google Fonts -->
        <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:ital,wght@1,700&display=swap" rel="stylesheet">

        <style>
        .home-header {
            font-family: 'Roboto Mono', monospace;
            font-size: 80px;
            font-weight: 700;
            font-style: italic;
            margin-bottom: 1px;
            margin-top: 20px;
            color: black;
            text-align: center;
        }
        .home-subtitle {
            font-family: 'Roboto Mono', monospace;
            font-size: 28px;
            font-weight: 700;
            font-style: italic;
            margin-top: 0;
            margin-bottom: 30px;
            color: black;
            text-align: center;
        }
        .info-box {
            background-color: rgba(255, 255, 255, 0.85);
            color: black;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
            margin-bottom: 15px;
            height: 100%;
        }
        .contact-box {
            background-color: rgba(255, 255, 255, 0.85);
            color: black;
            padding: 20px 30px;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
            margin-top: 10px;
            font-weight: 600;
            font-size: 16px;
            text-align: center;
        }
        </style>

        <div class='home-header'>Crop IQ</div>
        <div class='home-subtitle'>Your Digital Agronomy Partner.</div>
        """,
        unsafe_allow_html=True
    )

    # Two boxes side-by-side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div class='info-box'>
                <h3>About Us</h3>
                <p>
                    Crop IQ is a smart agriculture assistant that helps farmers make data-driven decisions.
                    It predicts yield, irrigation schedules, pesticide needs, and ROI—all in one place.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class='info-box'>
                <h3>How It Works</h3>
                <p>
                    Using AI and your farm data (soil, weather, crop type), Crop IQ models agricultural predictions 
                    and shows results through visual dashboards and tools.
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Contact info full-width box
    st.markdown("""
        <div class='contact-box'>
            📧 developercropiq@gmail.com  📞 +92 301 1063405  📍 Wapda Town, Lahore, Pakistan  📬 Postal Code: 058002
        </div>
    """, unsafe_allow_html=True)


# 1. Dashboard
if section == "📈 Dashboard":
    st.title("📊 Smart Agriculture Dashboard")
    st.subheader("Interactive Crop Data Overview")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(df, x="crop_type", title="Crop Type Distribution")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.box(df, x="crop_type", y="yield_kg_per_hectare", title="Yield by Crop Type")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig3 = px.scatter(df, x="rainfall_mm", y="soil_moisture_%", color="crop_type", 
                          title="Rainfall vs. Soil Moisture")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.density_contour(df, x="temperature_C", y="humidity_%", title="Temperature vs. Humidity")
        st.plotly_chart(fig4, use_container_width=True)

# 2. Yield Predictor
elif section == "🌾 Yield Predictor":
    st.title("🌾 Crop Yield Predictor")

    crop_type = st.selectbox("Crop Type", label_encoders['crop_type'].classes_)
    fert_type = st.selectbox("Fertilizer Type", label_encoders['fertilizer_type'].classes_)
    irri_type = st.selectbox("Irrigation Type", label_encoders['irrigation_type'].classes_)
    future_date = st.date_input("Expected Harvest Date", datetime.today())

    total_days = max(1, (future_date - datetime.today().date()).days)
    new_input = default_values.copy()
    new_input.update({
        'crop_type': label_encoders['crop_type'].transform([crop_type])[0],
        'fertilizer_type': label_encoders['fertilizer_type'].transform([fert_type])[0],
        'irrigation_type': label_encoders['irrigation_type'].transform([irri_type])[0],
        'total_days': total_days
    })

    prediction = model.predict(pd.DataFrame([new_input]))[0]
    maunds = (prediction / 40) / 2.47105
    st.markdown(f"<div class='predicted-result'>🌾 Predicted Yield: {maunds:.2f} maunds/acre</div>", unsafe_allow_html=True)


# 3. Irrigation Forecast
elif section == "💧 Irrigation Forecast":
    st.title("💧 Irrigation Forecasting")

    irrigation_intervals = {
        'Wheat': 28,
        'Sugarcane': 25,
        'Maize': 22,
        'Cotton': 24,
        'Rice': 10
    }
    df_int = df.copy()
    crop_names = label_encoders['crop_type'].inverse_transform(df['crop_type'])
    df_int['crop_name'] = crop_names
    df_int['irrigation_interval_days'] = df_int['crop_name'].map(irrigation_intervals)
    df_int = df_int.dropna(subset=['irrigation_interval_days'])

    le_crop = LabelEncoder()
    df_int['crop_encoded'] = le_crop.fit_transform(df_int['crop_name'])

    X_irrig = df_int[['crop_encoded', 'soil_moisture_%', 'temperature_C', 'rainfall_mm']]
    y_irrig = df_int['irrigation_interval_days']
    model_irrig = RandomForestRegressor(n_estimators=100, random_state=42)
    model_irrig.fit(X_irrig, y_irrig)

    crop_in = st.selectbox("Crop Type", le_crop.classes_)
    sm = st.slider("Soil Moisture (%)", 0, 100, 35)
    temp = st.slider("Temperature (°C)", -10, 50, 30)
    rain = st.slider("Rainfall (mm)", 0, 200, 15)

    encoded_crop = le_crop.transform([crop_in])[0]
    X_input = pd.DataFrame([[encoded_crop, sm, temp, rain]], columns=X_irrig.columns)
    result = model_irrig.predict(X_input)[0]
    st.markdown(f"<div class='predicted-result'>💧 Recommended irrigation every {result:.1f} days.</div>", unsafe_allow_html=True)

# 4. Pesticide Estimator
elif section == "🧪 Pesticide Estimator":
    st.title("🧪 Pesticide Usage Estimator")
    pest_df = df[['crop_type', 'fertilizer_type', 'irrigation_type', 'total_days', 'pesticide_usage_ml']].copy()
    pest_df['pesticide_usage_ml'] = pd.to_numeric(pest_df['pesticide_usage_ml'], errors='coerce') * 20

    X_p = pest_df.drop(columns='pesticide_usage_ml')
    y_p = pest_df['pesticide_usage_ml']
    model_p = RandomForestRegressor(n_estimators=100, random_state=42)
    model_p.fit(X_p, y_p)

    crop = st.selectbox("Crop", label_encoders['crop_type'].classes_)
    fert = st.selectbox("Fertilizer", label_encoders['fertilizer_type'].classes_)
    irri = st.selectbox("Irrigation", label_encoders['irrigation_type'].classes_)
    days = st.number_input("Total Crop Duration", min_value=1, value=90)

    input_df = pd.DataFrame([{
        'crop_type': label_encoders['crop_type'].transform([crop])[0],
        'fertilizer_type': label_encoders['fertilizer_type'].transform([fert])[0],
        'irrigation_type': label_encoders['irrigation_type'].transform([irri])[0],
        'total_days': days
    }])

    pred_ml = model_p.predict(input_df)[0]
    st.markdown(f"<div class='predicted-result'>🧪 Estimated pesticide required: {pred_ml:.2f} ml</div>", unsafe_allow_html=True)

# 5. ROI Calculator
elif section == "💰 ROI Calculator":
    st.title("💰 ROI & Profit Estimator")

    crop = st.selectbox("Crop", label_encoders['crop_type'].classes_)
    fert = st.selectbox("Fertilizer", label_encoders['fertilizer_type'].classes_)
    irri = st.selectbox("Irrigation", label_encoders['irrigation_type'].classes_)
    end_date = st.date_input("Expected Harvest Date", datetime.today())

    days = max(1, (end_date - datetime.today().date()).days)
    input_vals = default_values.copy()
    input_vals.update({
        'crop_type': label_encoders['crop_type'].transform([crop])[0],
        'fertilizer_type': label_encoders['fertilizer_type'].transform([fert])[0],
        'irrigation_type': label_encoders['irrigation_type'].transform([irri])[0],
        'total_days': days
    })

    pred_yield = model.predict(pd.DataFrame([input_vals]))[0]
    maunds = (pred_yield / 40) / 2.47105

    price_map = {'Wheat': 2000, 'Rice': 3600, 'Cotton': 8500, 'Maize': 1800, 'Sugarcane': 500}
    price = price_map.get(crop.capitalize(), 0)
    revenue = maunds * price

    cost = st.number_input("Total Cost (PKR/acre)", value=50000.0)
    invest = st.number_input("Investment (PKR/acre)", value=60000.0)

    profit = revenue - cost
    roi = (profit / invest) * 100 if invest else 0

    st.metric("Yield (maunds/acre)", f"{maunds:.2f}")
    st.metric("Revenue (PKR/acre)", f"{revenue:,.0f}")
    st.metric("Profit", f"{profit:,.0f}")
    st.metric("ROI", f"{roi:.2f}%")
    st.markdown(f"<div class='predicted-result'>💰 Estimated Profit: PKR {profit:,.0f} | ROI: {roi:.2f}%</div>", unsafe_allow_html=True)


    acres = st.slider("Scale (Acres)", 1, 100, 5)
    st.write("---")
    st.write(f"**Total Revenue:** PKR {revenue * acres:,.0f}")
    st.write(f"**Total Cost:** PKR {cost * acres:,.0f}")
    st.write(f"**Total Investment:** PKR {invest * acres:,.0f}")
    st.write(f"**Total Profit:** PKR {profit * acres:,.0f}")

# 5. Dashboard
elif section == "📊 Dashboard":
    st.title("📊 Smart Agriculture Dashboard")
    st.subheader("📡 Real-Time Sensor-Based Insights")

    from chat2 import smart_query, setup_retrieval_qa
    from chat1 import extract_pdf_text, initialize_vector_store

    # ✅ Manual refresh
    if st.button("🔁 Update Dashboard"):
        st.session_state['refresh_dashboard'] = True

    # ✅ Load chatbot chain (cached)
    @st.cache_resource(show_spinner="🤖 Loading AgriGenius Engine...")
    def load_chatbot_chain():
        pdf_files = ["Agriculture-Pakistan.pdf", "Climate-smart.pdf", "weather.pdf"]
        pdf_texts = [extract_pdf_text(file) for file in pdf_files]
        db = initialize_vector_store(pdf_texts)
        return setup_retrieval_qa(db)

    chain = load_chatbot_chain()

    # ✅ Load live sensor data
    def load_live_data():
        url = "https://raw.githubusercontent.com/Ahmad-Anwar58/EXP1/main/real_time_data.csv"
        response = requests.get(url)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        else:
            raise ValueError("❌ Failed to load data from GitHub.")

    if st.session_state.get('refresh_dashboard', True):
        try:
            df_live = load_live_data().sort_values(by="timestamp", ascending=False).head(50).sort_values(by="timestamp")
            df_live['time_str'] = df_live['timestamp'].dt.strftime('%H:%M:%S')

            # Recommendation logic
            def recommend_insight(var):
                values = df_live[var].values
                delta = round(values[-1] - values[0], 2)
                if abs(delta) < 0.1:
                    return f"No significant change detected in {var}."
                trend = "increased" if delta > 0 else "decreased"
                return f"{var.replace('_', ' ').capitalize()} has {trend} by {abs(delta)} units recently."

            def agri_tip(var):
                prompt = f"Give short practical steps for a farmer to act if '{var}' is fluctuating abnormally based on sensor data."
                return smart_query(chain, prompt)

            # Row 1: Soil Moisture & pH
            col1, col2 = st.columns(2)
            with col1:
                fig = px.line(df_live, x='time_str', y='soil_moisture_%', title='💧 Soil Moisture Over Time', markers=True)
                st.plotly_chart(fig, use_container_width=True)
                st.info(recommend_insight("soil_moisture_%"))
                st.markdown(f"<div style='color:black; font-weight:bold;'>🤖 {agri_tip('soil moisture')}</div>", unsafe_allow_html=True)


            with col2:
                fig = px.area(df_live, x='time_str', y='soil_pH', title='🧪 Soil pH Over Time', color_discrete_sequence=['#66c2a5'])
                st.plotly_chart(fig, use_container_width=True)
                st.info(recommend_insight("soil_pH"))
                st.markdown(f"<div style='color:black; font-weight:bold;'>🤖 {agri_tip('soil_pH')}</div>", unsafe_allow_html=True)


            # Row 2: Temperature & Rainfall
            col3, col4 = st.columns(2)
            with col3:
                fig = px.bar(df_live, x='time_str', y='temperature_C', title='🌡️ Temperature Over Time',
                             color='temperature_C', color_continuous_scale='thermal')
                st.plotly_chart(fig, use_container_width=True)
                st.info(recommend_insight("temperature_C"))
                st.markdown(f"<div style='color:black; font-weight:bold;'>🤖 {agri_tip('temperature_C')}</div>", unsafe_allow_html=True)

            with col4:
                fig = go.Figure(data=[go.Candlestick(
                    x=df_live['time_str'],
                    open=df_live['rainfall_mm'] - 0.5,
                    high=df_live['rainfall_mm'] + 1.0,
                    low=df_live['rainfall_mm'] - 1.0,
                    close=df_live['rainfall_mm'],
                    increasing_line_color='blue',
                    decreasing_line_color='lightblue'
                )])
                fig.update_layout(title="☔ Rainfall Fluctuation", height=350)
                st.plotly_chart(fig, use_container_width=True)
                st.info(recommend_insight("rainfall_mm"))
                st.markdown(f"<div style='color:black; font-weight:bold;'>🤖 {agri_tip('rainfall_mm')}</div>", unsafe_allow_html=True)

            # Row 3: Humidity & Sunlight
            col5, col6 = st.columns(2)
            with col5:
                fig = px.line(df_live, x='time_str', y='humidity_%', title='💨 Humidity Over Time',
                              markers=True, line_shape='spline', color_discrete_sequence=['orange'])
                st.plotly_chart(fig, use_container_width=True)
                st.info(recommend_insight("humidity_%"))
                st.markdown(f"<div style='color:black; font-weight:bold;'>🤖 {agri_tip('humidity_%')}</div>", unsafe_allow_html=True)

            with col6:
                fig = px.area(df_live, x='time_str', y='sunlight_hours', title='☀️ Sunlight Hours Over Time',
                              color_discrete_sequence=['#fdae61'])
                st.plotly_chart(fig, use_container_width=True)
                st.info(recommend_insight("sunlight_hours"))
                st.markdown(f"<div style='color:black; font-weight:bold;'>🤖 {agri_tip('sunlight_hours')}</div>", unsafe_allow_html=True)

            # NDVI Chart
            st.markdown("### 🌿 NDVI Index Over Time")
            fig = px.line(df_live, x='time_str', y='NDVI_index', title="🌿 NDVI Index", markers=True,
                          color_discrete_sequence=['green'])
            st.plotly_chart(fig, use_container_width=True)
            st.info(recommend_insight("NDVI_index"))
            st.markdown(f"<div style='color:black; font-weight:bold;'>🤖 {agri_tip('NDVI_index')}</div>", unsafe_allow_html=True)

            # Radar Chart Snapshot
            st.markdown("### 🕸️ Latest Sensor Snapshot")
            latest = df_live.iloc[-1]
            radar_df = pd.DataFrame({
                'Metric': ['Soil Moisture', 'Soil pH', 'Temperature', 'Rainfall', 'Humidity', 'Sunlight', 'NDVI'],
                'Value': [
                    latest['soil_moisture_%'],
                    latest['soil_pH'],
                    latest['temperature_C'],
                    latest['rainfall_mm'],
                    latest['humidity_%'],
                    latest['sunlight_hours'],
                    latest['NDVI_index']
                ]
            })
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=radar_df['Value'],
                theta=radar_df['Metric'],
                fill='toself',
                line=dict(color='teal')
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, max(radar_df['Value']) + 10])),
                showlegend=False,
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)

            # Reset refresh flag
            st.session_state['refresh_dashboard'] = False

        except Exception as e:
            st.error(f"❌ Unable to load dashboard data: {e}")




# Chatbot for help
# Chatbot for help
elif section == "💬 AgriTech Chatbot 🤖":
    st.title("🤖 AgriGenius – Smart Agriculture Assistant")
    st.caption("Ask me anything about farming, irrigation, crop cycles, or agricultural schemes!")

    # Import your external modules
    from chat1 import fetch_website_content, extract_pdf_text, initialize_vector_store
    from chat2 import llm, setup_retrieval_qa

    @st.cache_resource(show_spinner="🔄 Loading AgriGenius... please wait")
    def setup_agriculture_chatbot():
        # Define sources
        pdf_files = ["Agriculture-Pakistan.pdf", "Climate-smart.pdf", "weather.pdf"]
        pdf_texts = [extract_pdf_text(file) for file in pdf_files]

        # Use only PDF contents for now
        all_contents = pdf_texts  # + website_contents (optional if DNS works later)

        # Vector store setup
        db = initialize_vector_store(all_contents)

        # Build the RetrievalQA chain
        chain = setup_retrieval_qa(db)
        return chain

    chain = setup_agriculture_chatbot()

    user_question = st.text_input("You:", key="chat_input")
    if user_question:
        with st.spinner("🤖 Thinking..."):
            try:
                result = chain(user_question)
                st.markdown(
                    f"<div class='agri-chat-reply'><strong>AgriGenius:</strong> {result['result']}</div>",
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"❌ Error: {e}")



# 6. Live Sensor Data Monitor
elif section == "📡 Live Sensor Data":
    st.title("📡 Live Sensor Data Monitor")

    try:
        df_live = load_live_data()
        latest = df_live.iloc[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("🌡️ Temperature (°C)", f"{latest['temperature_C']}")
        col2.metric("💧 Soil Moisture (%)", f"{latest['soil_moisture_%']}")
        col3.metric("🌿 NDVI Index", f"{latest['NDVI_index']}")

        col4, col5, col6 = st.columns(3)
        col4.metric("☀️ Sunlight (hrs)", f"{latest['sunlight_hours']}")
        col5.metric("☔ Rainfall (mm)", f"{latest['rainfall_mm']}")
        col6.metric("💨 Humidity (%)", f"{latest['humidity_%']}")

        st.markdown("### 🧾 Most Recent Sensor Data")
        st.dataframe(latest.to_frame().T)

        if not pd.isna(latest.get('latitude')) and not pd.isna(latest.get('longitude')):
            st.map(pd.DataFrame({
                'lat': [latest['latitude']],
                'lon': [latest['longitude']]
            }))

    except Exception as e:
        st.error(f"❌ Error fetching live data: {e}")

