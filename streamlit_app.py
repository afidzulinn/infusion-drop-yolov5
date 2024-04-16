import streamlit as st
import requests

# Define the FastAPI server URL
SERVER_URL = "http://localhost:8000"

# Define API endpoints
START_DETECTION_URL = f"{SERVER_URL}/start_detection"
STOP_DETECTION_URL = f"{SERVER_URL}/stop_detection"
GET_DROP_STATS_URL = f"{SERVER_URL}/drop_stats"

# Title and description for the app
st.title("Infusion Drop Detection App")
st.markdown("This app interacts with a FastAPI server for object detection and drop counting.")

# Start/Stop detection buttons
if st.button("Start Detection"):
    response = requests.post(START_DETECTION_URL)
    if response.status_code == 200:
        st.success("Object detection started.")
    else:
        st.error("Error starting object detection.")

if st.button("Stop Detection"):
    response = requests.post(STOP_DETECTION_URL)
    if response.status_code == 200:
        st.success("Object detection stopped.")
    else:
        st.error("Error stopping object detection.")

# Get drop stats button
if st.button("Get Drop Stats"):
    response = requests.get(GET_DROP_STATS_URL)
    if response.status_code == 200:
        drop_stats = response.json()
        st.write("Total Drops:", drop_stats["total_drops"])
        st.write("Drops in One Minute:", drop_stats["drops_in_one_minute"])
    else:
        st.error("Error getting drop stats.")
