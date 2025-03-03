from sklearn.linear_model import LogisticRegression
import streamlit as st
import os
import json
import pandas as pd
import numpy as np
from gtts import gTTS
from playsound import playsound
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import threading
import time

# File to store appointments
APPOINTMENTS_FILE = "appointments.json"

# Function to Load Appointments from File
def load_appointments():
    try:
        with open(APPOINTMENTS_FILE, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

# Function to Save Appointments to File
def save_appointments(appointments):
    with open(APPOINTMENTS_FILE, "w") as file:
        json.dump(appointments, file, indent=4)

# Initialize Appointments in Session
if "appointments" not in st.session_state:
    st.session_state.appointments = load_appointments()

# Initialize speech flags for each page
if "speech_flags" not in st.session_state:
    st.session_state.speech_flags = {
        "Dashboard": False,
        "Calendar": False,
        "Medication Reminder": False,
        "Health Monitor": False,
        "Diet Recommendations": False,
        "Daily Routines": False,
        "Emergency Assistance": False
    }

# Function for Text-to-Speech
def speak(text, force=False):
    filename = "speech.mp3"
    try:
        if os.path.exists(filename):
            os.remove(filename)  # Remove old file before saving a new one
        tts = gTTS(text=text, lang="en")
        tts.save(filename)
        playsound(filename)
    except Exception as e:
        st.error(f"Error in text-to-speech: {e}")

# Streamlit UI Setup
st.set_page_config(page_title="CareVox", layout="wide")
st.title("👵 CareVox 🎤")
st.subheader("A smart voice for golden years.")

# Sidebar Navigation
st.sidebar.title("Navigation")
feature = st.sidebar.radio("Choose a Feature:", [
    "Dashboard", "Medication Reminder", "Health Monitor", "Diet Recommendations",
    "Calendar", "Daily Routines", "Emergency Assistance"
])

# Reset speech flags when feature changes
if "previous_feature" not in st.session_state:
    st.session_state.previous_feature = feature
elif st.session_state.previous_feature != feature:
    st.session_state.speech_flags[feature] = False
    st.session_state.previous_feature = feature

# 🏠 Dashboard
if feature == "Dashboard":
    st.subheader("Welcome to CareVox")
    st.write("Welcome to CareVox! Let’s make your day smoother, one request at a time.")
    
    # Play welcome message only once for this page
    if not st.session_state.speech_flags["Dashboard"]:
        speak("Welcome to CareVox! Let’s make your day smoother, one request at a time.")
        st.session_state.speech_flags["Dashboard"] = True

# 📅 Calendar (Appointments)
elif feature == "Calendar":
    st.subheader("📅 Manage Appointments")
    
    # Play welcome message only once for this page
    if not st.session_state.speech_flags["Calendar"]:
        speak("You are in the Calendar section. Manage your appointments here.")
        st.session_state.speech_flags["Calendar"] = True

    # Add an Appointment
    st.write("### 📌 Add New Appointment")
    title = st.text_input("Appointment Title")
    date = st.date_input("Select Date")
    time = st.time_input("Select Time")
    details = st.text_area("Additional Details (Optional)")

    if st.button("Add Appointment"):
        if title and date and time:
            appointment = {
                "id": len(st.session_state.appointments) + 1,
                "title": title,
                "date": str(date),
                "time": str(time),
                "details": details
            }
            st.session_state.appointments.append(appointment)
            save_appointments(st.session_state.appointments)
            st.success("Appointment added successfully!")
            # This speech will always play when button is clicked
            speak(f"Your appointment for {title} has been added.")
        else:
            st.error("Please enter a title, date, and time.")

    # Remove an Appointment
    if st.session_state.appointments:
        st.write("### ❌ Remove an Appointment")
        appt_options = {f"{a['title']} - {a['date']} {a['time']}": a for a in st.session_state.appointments}
        selected_appt = st.selectbox("Select an appointment to remove:", list(appt_options.keys()))

        if st.button("Remove Appointment"):
            st.session_state.appointments.remove(appt_options[selected_appt])
            save_appointments(st.session_state.appointments)
            st.success("Appointment removed successfully!")
            # This speech will always play when button is clicked
            speak("Your appointment has been removed.")

    # Show All Appointments
    st.write("### 📆 Your Appointments:")
    if st.session_state.appointments:
        for appt in st.session_state.appointments:
            st.write(f"**{appt['title']}** - {appt['date']} at {appt['time']}")
            if appt["details"]:
                st.write(f"📝 {appt['details']}")
            st.write("---")
    else:
        st.write("No appointments scheduled.")

# 💊 Medication Reminder
elif feature == "Medication Reminder":
    st.subheader("Medication Reminder")
    
    # Play welcome message only once for this page
    if not st.session_state.speech_flags["Medication Reminder"]:
        speak("You are in the Medication Reminder section. Please provide the required details.")
        st.session_state.speech_flags["Medication Reminder"] = True
    
    # Initialize medication data
    medication_df = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "time_of_day": [8, 12, 18, 8, 14, 20, 9, 11, 19, 7],
        "notification_received": [1, 0, 1, 1, 1, 0, 0, 1, 1, 0],
        "past_adherence": [0.9, 0.8, 0.7, 0.85, 0.6, 0.5, 0.95, 0.75, 0.4, 0.3],
        "activity_level": [70, 50, 30, 90, 20, 10, 80, 55, 25, 15],
        "adherence": [1, 1, 0, 1, 0, 0, 1, 1, 0, 0]
    })
    
    medications = {
        "Diabetes": "Metformin or Insulin",
        "Hypertension": "Lisinopril or Amlodipine",
        "Asthma": "Albuterol or Montelukast",
        "Cholesterol": "Atorvastatin or Simvastatin",
        "Depression": "Sertraline or Fluoxetine",
        "Arthritis": "Ibuprofen or Naproxen",
        "No Chronic Condition": "No specific medication required"
    }
    
    # UI for user input
    time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
    notification_received = st.radio("Did you receive a reminder?", ["Yes", "No"])
    past_adherence = st.slider("Past Adherence (0-1)", 0.0, 1.0, 0.8)
    activity_level = st.slider("Activity Level (1-100)", 1, 100, 50)
    health_condition = st.selectbox("Select Health Condition", list(medications.keys()))
    
    if st.button("Predict Adherence"):
        adherence_prediction = "Likely to adhere" if past_adherence > 0.6 and activity_level > 30 else "Unlikely to adhere"
        medication_suggestion = medications[health_condition]
        st.success(f"Prediction: {adherence_prediction}")
        st.info(f"Recommended Medication: {medication_suggestion}")
        speak(f"Based on your input, you are {adherence_prediction}. Recommended medication: {medication_suggestion}")

# 📊 Health Monitor
elif feature == "Health Monitor":
    st.subheader("Health Monitor")
    if not st.session_state.speech_flags["Health Monitor"]:
        speak("You are in the Health Monitor section. Please enter your vital signs.")
        st.session_state.speech_flags["Health Monitor"] = True

    # Sample Health Data for Training (Modify as needed)
    health_df = pd.DataFrame({
        "heart_rate": [72, 85, 95, 60, 110, 120, 78, 88, 102, 115],
        "blood_pressure": [120, 135, 140, 110, 160, 170, 125, 138, 150, 165],
        "oxygen_level": [98, 95, 92, 99, 88, 85, 97, 94, 90, 87],
        "activity_level": [8000, 5000, 3000, 10000, 2000, 1500, 7500, 4800, 2500, 1800],
        "health_status": [0, 0, 1, 0, 1, 1, 0, 0, 1, 1]  # 0 - Normal, 1 - Concerning
    })

    # Train a Logistic Regression Model
    health_scaler = StandardScaler()
    X = health_scaler.fit_transform(health_df.drop(columns=["health_status"]))
    y = health_df["health_status"]

    health_model = LogisticRegression()
    health_model.fit(X, y)

    def predict_health_status(heart_rate, blood_pressure, oxygen_level, activity_level):
        """Predict health risk using Logistic Regression."""
        new_data = np.array([[heart_rate, blood_pressure, oxygen_level, activity_level]])
        new_data = health_scaler.transform(new_data)
        probability = health_model.predict_proba(new_data)[0][1]  # Probability of '1' (concerning health)

        result = f"Predicted health risk probability: {probability:.2f}"

        if probability > 0.5:
            speak("Warning! Your health status is concerning. Please seek medical attention immediately.")
        else:
            speak("Your health status is normal. Keep maintaining a healthy lifestyle.")

        return result

    heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=72)
    blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
    oxygen_level = st.number_input("Oxygen Level (%)", min_value=70, max_value=100, value=98)
    activity_level = st.number_input("Activity Level (steps)", min_value=0, max_value=20000, value=5000)

    if st.button("Check Health Status"):
        result = predict_health_status(heart_rate, blood_pressure, oxygen_level, activity_level)
        st.success(result)


# 🍎 Diet Recommendations
elif feature == "Diet Recommendations":
    st.subheader("Diet Recommendations")
    if not st.session_state.speech_flags["Diet Recommendations"]:
        speak("You are in the Diet Recommendations section. Please provide the required details.")
        st.session_state.speech_flags["Diet Recommendations"] = True
    
    diet_plans = {
        "Diabetes": "Eat whole grains, lean proteins, green leafy vegetables, and avoid processed sugar.",
        "Hypertension": "Reduce salt intake, eat potassium-rich foods like bananas, oranges, and leafy greens.",
        "Heart Disease": "Eat omega-3 rich foods like salmon, nuts, and fiber-rich fruits & vegetables.",
        "Kidney Disease": "Limit sodium, potassium, and phosphorus intake. Stay hydrated and avoid processed foods.",
        "Arthritis": "Consume anti-inflammatory foods like turmeric, ginger, berries, and fatty fish.",
        "Osteoporosis": "Increase calcium and vitamin D intake with dairy, leafy greens, and fortified foods."
    }
    condition = st.selectbox("Select Your Health Condition", list(diet_plans.keys()))
    
    if st.button("Get Diet Plan"):
        recommendation = diet_plans.get(condition, "Please consult a dietitian for a personalized meal plan.")
        st.success(f"Recommended Diet Plan for {condition}: {recommendation}")
        speak(f"For {condition}, {recommendation}")

# 📅 Daily Routines
elif feature == "Daily Routines":
    st.subheader("Daily Routines")
    
    if not st.session_state.speech_flags["Daily Routines"]:
        speak("You are in the Daily Routines section. Here are your scheduled activities.")
        st.session_state.speech_flags["Daily Routines"] = True
    
    daily_routines = {
        'Morning Routine': [
            '7:00 AM - Take morning medications',
            '7:30 AM - Check blood pressure',
            '8:00 AM - Drink water',
            '8:30 AM - Eat breakfast'
        ],
        'Afternoon Routine': [
            '12:00 PM - Light exercise',
            '12:30 PM - Have lunch',
            '2:00 PM - Take afternoon medications',
            '3:00 PM - Hydration check'
        ],
        'Evening Routine': [
            '6:00 PM - Prepare dinner',
            '7:00 PM - Evening walk',
            '8:00 PM - Take evening medications',
            '9:00 PM - Prepare for bed'
        ]
    }
    
    routine_choice = st.selectbox("Select Routine", list(daily_routines.keys()))
    st.write("Here is your ", routine_choice)
    for activity in daily_routines[routine_choice]:
        st.write(f"✅ {activity}")
    
    if st.button("Start Voice Reminders"):
        speak(f"Starting {routine_choice}. Here are your tasks.")
        for activity in daily_routines[routine_choice]:
            speak(activity)

    health_alerts = [
        ('8:00 AM', "It's time to drink water. Stay hydrated!"),
        ('11:00 AM', "Mid-morning hydration reminder. Have a glass of water."),
        ('2:00 PM', "Afternoon hydration break. Drink some water."),
        ('5:00 PM', "Evening hydration time. Keep yourself well-hydrated."),
        ('7:30 AM', "Good morning! Time for a nutritious breakfast."),
        ('12:30 PM', "Lunchtime! Enjoy a balanced meal."),
        ('6:30 PM', "Dinner time. Have a pleasant evening meal."),
        ('7:00 AM', "Morning medication time. Take your prescribed medicines."),
        ('2:00 PM', "Afternoon medication reminder."),
        ('8:00 PM', "Evening medications. Follow your health plan.")
    ]
    
    def start_daily_reminders():
        def get_current_time():
            return datetime.now().strftime('%I:%M %p')

        def reminder_loop():
            while True:
                current_time = get_current_time()
                for alert_time, alert_message in health_alerts:
                    if alert_time == current_time:
                        speak(alert_message)
                time.sleep(60)

        threading.Thread(target=reminder_loop, daemon=True).start()
        speak("Daily voice reminders activated. I'll help you stay on track with your health routine.")

    if st.button("Activate Health Reminders"):
        start_daily_reminders()

# 🚨 Emergency Assistance
elif feature == "Emergency Assistance":
    st.subheader("🚨 Emergency Assistance")
    
    if not st.session_state.speech_flags["Emergency Assistance"]:
        speak("You are in the Emergency Assistance section. Select an option.")
        st.session_state.speech_flags["Emergency Assistance"] = True
    
    emergency_status = st.radio("Select Emergency Status:", ["Feeling Fine", "Call Emergency Contact", "Call Emergency Services"])
    
    if emergency_status == "Call Emergency Contact":
        st.write("### 📞 Call an Emergency Contact")
        predefined_contacts = {"Doctor": "+1234567890", "Family Member": "+0987654321", "Neighbor": "+1122334455"}
        selected_contact = st.selectbox("Select a Contact:", list(predefined_contacts.keys()) + ["Custom Number"])
        
        if selected_contact == "Custom Number":
            custom_number = st.text_input("Enter Phone Number:", max_chars=15)
            if st.button("Call Custom Number") and custom_number:
                st.success(f"Calling {custom_number}...")
                speak(f"Calling {custom_number}. Please wait.")
        else:
            if st.button(f"Call {selected_contact}"):
                st.success(f"Calling {selected_contact} at {predefined_contacts[selected_contact]}...")
                speak(f"Calling {selected_contact}. Please wait.")
    
    elif emergency_status == "Call Emergency Services":
        st.error("Calling Emergency Services Now!")
        speak("Calling emergency services. Please stay calm.")