import streamlit as st
from chatbot import Chatbot
from data_loader import get_latest_sensor_data

# Page settings
st.set_page_config(page_title="CropIQ Assistant", page_icon="🌾")

st.title("👨‍🌾 CropIQ – Your Smart Farming Assistant")
st.write("Ask anything about your field, crop, irrigation, or farm management.")

# Load chatbot and latest data
chatbot = Chatbot(faq_path="faq.json")
latest_data = get_latest_sensor_data("System_Data_Cleaned.xlsx")

# User input
user_input = st.text_input("💬 Your Question")

# Handle response
if user_input:
    reply = chatbot.get_response(user_input, context_data=latest_data)
    st.markdown("#### 🤖 Chatbot says:")
    st.success(reply)
