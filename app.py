import streamlit as st
import pandas as pd
import numpy as np
import serial
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

st.set_page_config(layout="wide", page_title="FarmIQ - Smart Agriculture Rover")

try:
    model = load_model("plant_disease_model.h5")
except:
    st.error("Could not load the model file. Please make sure 'plant_disease_model.h5' exists.")
    model = None

class_labels = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

remedies = {
    "Pepper__bell___Bacterial_spot": "Use copper-based bactericides and remove infected leaves.",
    "Pepper__bell___healthy": "Plants look healthy. Maintain regular watering.",
    "Potato___Early_blight": "Apply fungicides containing chlorothalonil. Remove infected foliage.",
    "Potato___Late_blight": "Use fungicides like mancozeb. Avoid overhead watering.",
    "Potato___healthy": "Healthy plants. Keep regular care.",
    "Tomato_Bacterial_spot": "Spray copper-based sprays. Remove infected leaves.",
    "Tomato_Early_blight": "Use fungicides like chlorothalonil. Rotate crops.",
    "Tomato_Late_blight": "Apply fungicides early in season.",
    "Tomato_Leaf_Mold": "Improve airflow. Apply copper fungicides.",
    "Tomato_Septoria_leaf_spot": "Remove infected leaves. Use drip irrigation.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Spray neem oil or insecticidal soap.",
    "Tomato__Target_Spot": "Remove infected leaves. Avoid overhead irrigation.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Control whiteflies. Remove infected plants.",
    "Tomato__Tomato_mosaic_virus": "Use resistant varieties. Disinfect tools.",
    "Tomato_healthy": "Plants are healthy. Continue monitoring."
}

if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["Temperature", "Humidity", "Soil Moisture"])
if 'ser' not in st.session_state:
    st.session_state.ser = None
if 'reading_data' not in st.session_state:
    st.session_state.reading_data = False
if 'plant_health' not in st.session_state:
    st.session_state.plant_health = None
if 'plant_confidence' not in st.session_state:
    st.session_state.plant_confidence = 0
if 'port' not in st.session_state:
    st.session_state.port = "COM3"
if 'connection_status' not in st.session_state:
    st.session_state.connection_status = "disconnected"
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()
if 'connection_message' not in st.session_state:
    st.session_state.connection_message = None

st.title("🌱 FarmIQ - Smart Agriculture Rover Dashboard")

tab1, tab2, tab3 = st.tabs(["🏠 Dashboard", "📷 Plant Analysis", "🌾 Yield Prediction"])

with tab1:
    conn_col1, conn_col2, conn_col3 = st.columns([2, 1, 1])
    with conn_col1:
        st.session_state.port = st.text_input("Arduino COM Port:", st.session_state.port, 
                                             help="e.g., COM3 or /dev/ttyUSB0")
    
    with conn_col2:
        connect_button = st.button("🔌 Connect", use_container_width=True)
    with conn_col3:
        disconnect_button = st.button("❌ Disconnect", use_container_width=True)
    
    if st.session_state.connection_status == "connected":
        st.success(f"✅ {st.session_state.port} Connected - {len(st.session_state.data)} readings collected")
    else:
        st.warning("⚠️ Rover Not Connected")
    
    if connect_button:
        try:
            if st.session_state.ser is not None:
                try:
                    st.session_state.ser.close()
                except:
                    pass
            
            st.session_state.ser = serial.Serial(
                port=st.session_state.port, 
                baudrate=9600, 
                timeout=2, 
                write_timeout=2
            )
            
            st.session_state.ser.write(b'?')
            time.sleep(1)
            
            if st.session_state.ser.in_waiting > 0:
                response = st.session_state.ser.readline().decode().strip()
                st.session_state.connection_status = "connected"
                st.session_state.reading_data = True
                st.session_state.connection_message = f"Connected to {st.session_state.port} ✅"
            else:
                st.session_state.connection_status = "connected"
                st.session_state.reading_data = True
                st.session_state.connection_message = "Connected but no response from rover"
                
        except Exception as e:
            st.session_state.connection_status = "disconnected"
            st.session_state.reading_data = False
            st.session_state.connection_message = f"Connection failed: {e}"
            if st.session_state.ser is not None:
                try:
                    st.session_state.ser.close()
                except:
                    pass
            st.session_state.ser = None

    if disconnect_button:
        if st.session_state.ser is not None:
            try:
                st.session_state.ser.close()
            except:
                pass
        st.session_state.reading_data = False
        st.session_state.connection_status = "disconnected"
        st.session_state.connection_message = "Disconnected from rover"
        st.session_state.ser = None

    st.subheader("🕹 Rover Controls")
    if st.session_state.connection_status != "connected":
        st.warning("Connect to rover first")
    else:
        control_col1, control_col2, control_col3 = st.columns(3)
        
        with control_col1:
            if st.button("⬅️ Left", use_container_width=True, key="left_btn"):
                try:
                    st.session_state.ser.write(b'L')
                    st.session_state.ser.flush()
                    st.success("← Left")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        with control_col2:
            if st.button("⬆️ Forward", use_container_width=True, key="forward_btn"):
                try:
                    st.session_state.ser.write(b'f')
                    st.session_state.ser.flush()
                    st.success("↑ Forward")
                except Exception as e:
                    st.error(f"Error: {e}")
            
            if st.button("⬇️ Backward", use_container_width=True, key="backward_btn"):
                try:
                    st.session_state.ser.write(b'b')
                    st.session_state.ser.flush()
                    st.success("↓ Backward")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        with control_col3:
            if st.button("➡️ Right", use_container_width=True, key="right_btn"):
                try:
                    st.session_state.ser.write(b'r')
                    st.session_state.ser.flush()
                    st.success("→ Right")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if st.button("⏹️ Stop", use_container_width=True, key="stop_btn"):
            try:
                st.session_state.ser.write(b's')
                st.session_state.ser.flush()
                st.success("⏹ Stopped")
            except Exception as e:
                st.error(f"Error: {e}")

    st.subheader("📊 Live Sensor Data")

    if st.session_state.reading_data and st.session_state.connection_status == "connected":
        try:
            if st.session_state.ser.in_waiting > 0:
                line = st.session_state.ser.readline().decode("utf-8").strip()
                if line and "," in line:
                    try:
                        temp, hum, soil = map(float, line.split(","))
                        new_row = {"Temperature": temp, "Humidity": hum, "Soil Moisture": soil}
                        st.session_state.data = pd.concat(
                            [st.session_state.data, pd.DataFrame([new_row])],
                            ignore_index=True
                        )
                    except ValueError:
                        pass  
        except Exception as e:
            st.session_state.connection_status = "disconnected"
            st.session_state.reading_data = False
    
    if st.session_state.reading_data and st.session_state.connection_status == "connected":
        if time.time() - st.session_state.last_update > 2:
            st.session_state.last_update = time.time()
            st.rerun()

    if not st.session_state.data.empty:

        latest = st.session_state.data.iloc[-1]
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Temperature", f"{latest['Temperature']:.1f}°C", 
                     help="Optimal: 20-30°C")
        with metric_col2:
            st.metric("Humidity", f"{latest['Humidity']:.1f}%", 
                     help="Optimal: 40-80%")
        with metric_col3:
            st.metric("Soil Moisture", f"{latest['Soil Moisture']:.1f}", 
                     help="Higher values = more moisture")
        
        st.line_chart(st.session_state.data.tail(20), height=200)
        
        with st.expander("View Raw Data", expanded=False):
            st.dataframe(st.session_state.data.tail(10), use_container_width=True)
    else:
        st.info("No sensor data yet. Connect to the rover to start receiving data.")

with tab2:
    st.header("Plant Health Analysis")
    
    if model is None:
        st.error("Disease detection model not available")
    else:
        uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"],
                                        help="Upload an image of a plant leaf for analysis")

        if uploaded_file is not None:
            img = Image.open(uploaded_file).resize((224, 224))
            st.image(img, caption="Uploaded Leaf Image", use_container_width=True)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            with st.spinner("Analyzing image..."):
                prediction = model.predict(img_array)
                predicted_class = class_labels[np.argmax(prediction)]
                confidence = np.max(prediction) * 100
                st.session_state.plant_health = predicted_class
                st.session_state.plant_confidence = confidence

            if "healthy" in predicted_class:
                st.success(f"🌿 Healthy Plant ({confidence:.1f}% confidence)")
            else:
                st.error(f"⚠️ {predicted_class} ({confidence:.1f}% confidence)")
            
            st.write("**Treatment:**", remedies[predicted_class])
            
            with st.expander("View detailed confidence scores"):
                confidence_scores = {class_labels[i]: float(prediction[0][i]) * 100 for i in range(len(class_labels))}
                sorted_scores = dict(sorted(confidence_scores.items(), key=lambda item: item[1], reverse=True))
                
                for cls, score in sorted_scores.items():
                    st.write(f"{cls}: {score:.2f}%")

with tab3:
    st.header("Crop Yield Prediction")
    
    if not st.session_state.data.empty and st.session_state.plant_health:
        latest = st.session_state.data.iloc[-1]
        
        temp_factor = 1.0 - abs(25 - latest['Temperature']) / 25  
        humidity_factor = max(0, min(1, latest['Humidity'] / 80))  
        soil_factor = max(0, min(1, latest['Soil Moisture'] / 100))  
        
        if "healthy" in st.session_state.plant_health:
            health_factor = 1.0
        else:
            health_factor = 0.5  
            
        confidence_factor = st.session_state.plant_confidence / 100
        
        predicted_yield = (temp_factor * 0.3 + humidity_factor * 0.3 + 
                          soil_factor * 0.2 + health_factor * 0.2) * confidence_factor * 100
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Predicted Yield Efficiency", f"{predicted_yield:.1f}%")
        
        with col2:
            st.progress(predicted_yield/100, text="Yield Potential")
        
        with st.expander("Factors Affecting Yield", expanded=False):
            st.write(f"**Temperature:** {temp_factor:.2f} (Optimal: 25°C, Current: {latest['Temperature']:.1f}°C)")
            st.write(f"**Humidity:** {humidity_factor:.2f} (Optimal: 60-80%, Current: {latest['Humidity']:.1f}%)")
            st.write(f"**Soil Moisture:** {soil_factor:.2f}")
            st.write(f"**Plant Health:** {health_factor:.2f}")
            st.write(f"**Analysis Confidence:** {confidence_factor:.2f}")
            
        st.subheader("Recommendations")
        if temp_factor < 0.7:
            st.warning("🌡️ Temperature is suboptimal for maximum yield.")
        if humidity_factor < 0.7:
            st.warning("💧 Humidity is suboptimal for maximum yield.")
        if soil_factor < 0.7:
            st.warning("🌱 Soil moisture may need adjustment.")
        if health_factor < 1.0:
            st.warning("🪴 Plant health issues are reducing potential yield.")
            
        if temp_factor > 0.8 and humidity_factor > 0.8 and soil_factor > 0.8 and health_factor > 0.9:
            st.success("✅ Ideal conditions detected! Maintain current parameters.")
    else:
        st.info("Upload a plant image and connect sensors for yield prediction.")

st.markdown("""
<style>
    .stButton button {
        width: 100%;
        margin-bottom: 5px;
    }
    .stMetric {
        padding: 5px;
        border-radius: 5px;
    }
    div[data-testid="stExpander"] details summary p {
        font-weight: 500;
        font-size: 16px;
    }
    .compact-chart {
        height: 200px !important;
    }
</style>
""", unsafe_allow_html=True)
