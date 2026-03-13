
import streamlit as st
import pickle
import base64
import numpy as np

# --- Function to set background image ---
def set_bg(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    page_bg = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }}
    [data-testid="stHeader"], [data-testid="stToolbar"] {{
        background: rgba(0,0,0,0);
    }}
    label, .stSelectbox label, .stNumberInput label {{
        color: #f5f5f5 !important;
        font-weight: 600 !important;
    }}
    h1, h2, h3, h4, p, span {{
        color: #f5f5f5 !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: #f9f9f9 !important;
        color: #222 !important;
    }}
    [data-testid="stSidebar"] * {{
        color: #222 !important;
        font-weight: 600 !important;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)


# --- Main Function ---
def main():
    set_bg("background.png")  # Replace with your background image

    # Sidebar
    st.sidebar.title("🚦 Navigation")
    page = st.sidebar.radio("Go to", ["Home Page", "Prediction Page"])

    # ---------------- HOME PAGE ----------------
    if page == "Home Page":
        st.markdown("<h1 style='text-align:center;'>SafePath</h1>", unsafe_allow_html=True)
        st.write("")
        st.markdown(
            """
            ### 🧭 **About**
            Road accidents are a growing concern across the world, leading to injuries, fatalities, and property loss.  
            This app predicts the **severity level of a traffic accident** based on factors like  
            weather, road type, traffic density, and time of day.  

            ---
            👉 Click **“Prediction Page”** in the sidebar to input details  
            and view the predicted severity level instantly.
            """
        )

    # ---------------- PREDICTION PAGE ----------------
    elif page == "Prediction Page":
        st.markdown("<h2 style='text-align:center;'>ACCIDENT SEVERITY PREDICTION 🚦</h2>", unsafe_allow_html=True)

        # Load model, scaler, and encoders
        model = pickle.load(open("model_xgb.sav", "rb"))
        scaler = pickle.load(open("scaler_xgb.sav", "rb"))
        encoders = pickle.load(open("all_encoders.sav", "rb"))

        st.markdown("<p style='text-align:center;'>Fill in the details below to predict accident severity</p>", unsafe_allow_html=True)

        # --- Input Section ---
        with st.form("prediction_form"):
            st.markdown("### Environmental Conditions")
            col1, col2, col3 = st.columns(3)
            with col1:
                temp = st.number_input("Temperature (°C)", value=None, placeholder="Enter value")
                hum = st.number_input("Humidity (%)", value=None, placeholder="Enter value")
            with col2:
                vis = st.number_input("Visibility (m)", value=None, placeholder="Enter value")
                ws = st.number_input("Wind Speed (km/h)", value=None, placeholder="Enter value")
            with col3:
                weather = st.selectbox("Weather", options=[""] + list(encoders['Weather'].classes_), index=0, placeholder="Select Weather")
                time_of_day = st.selectbox("Time of Day", options=[""] + list(encoders['TimeOfDay'].classes_), index=0, placeholder="Select Time of Day")

            st.markdown("### Road & Traffic Conditions")
            col4, col5 = st.columns(2)
            with col4:
                road = st.selectbox("Road Type", options=[""] + list(encoders['RoadType'].classes_), index=0, placeholder="Select Road Type")
                traffic = st.selectbox("Traffic Density", options=[""] + list(encoders['TrafficDensity'].classes_), index=0, placeholder="Select Traffic Density")
            with col5:
                vc = st.number_input("Vehicle Count", value=None, placeholder="Enter value")

            predict_btn = st.form_submit_button("🔍 PREDICT")

        # --- When Predict Button Clicked ---
        if predict_btn:
            # Check for empty fields
            if (
                temp is None or hum is None or vis is None or ws is None or vc is None or
                weather == "" or time_of_day == "" or road == "" or traffic == ""
            ):
                st.warning("⚠️ Please fill in all the fields before predicting!")
            else:
                try:
                    # Encode categorical features
                    weather_encoded = encoders['Weather'].transform([weather])[0]
                    road_encoded = encoders['RoadType'].transform([road])[0]
                    traffic_encoded = encoders['TrafficDensity'].transform([traffic])[0]
                    time_encoded = encoders['TimeOfDay'].transform([time_of_day])[0]

                    # Combine into feature array
                    features = np.array([[temp, hum, vis, ws,
                                          weather_encoded, road_encoded,
                                          traffic_encoded, time_encoded, vc]])

                    # Scale and predict
                    scaled_features = scaler.transform(features)
                    result = model.predict(scaled_features)

                    # Decode severity
                    severity_label = encoders['Severity'].inverse_transform(result)[0].capitalize()

                    # Display styled result
                    st.markdown("---")
                    if severity_label == "Fatal":
                        st.markdown(f"<h2 style='color:red;'>⚠️ Predicted Severity: {severity_label}</h2>", unsafe_allow_html=True)
                        st.warning("Extremely dangerous! Take immediate precautions.")
                    elif severity_label == "Serious":
                        st.markdown(f"<h2 style='color:orange;'>🚨 Predicted Severity: {severity_label}</h2>", unsafe_allow_html=True)
                        st.info("Serious accident risk. Drive carefully!")
                    elif severity_label == "Moderate":
                        st.markdown(f"<h2 style='color:gold;'>⚠️ Predicted Severity: {severity_label}</h2>", unsafe_allow_html=True)
                        st.warning("Moderate accident risk. Stay alert.")
                    elif severity_label == "Minor":
                        st.markdown(f"<h2 style='color:lightgreen;'>✅ Predicted Severity: {severity_label}</h2>", unsafe_allow_html=True)
                        st.success("Minor accident risk. Safe conditions.")
                    else:
                        st.error("Unknown severity predicted!")
                except Exception as e:
                    st.error(f"⚠️ Error during prediction: {e}")


# Run the app
if __name__ == "__main__":
    main()
