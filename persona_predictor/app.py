import os
import streamlit as st
import pandas as pd
import pickle

from tensorflow.keras.models import load_model

# Set up the app title and description
st.title("Persona Predictor: Introvert or Extrovert?")

st.write(
    "This app predicts whether a user is an introvert or extrovert based on their social behavior and preferences."
)

# Collect inputs from the user
time_spent_alone = st.number_input(
    "Time Spent Alone (hours):", min_value=0.0, max_value=24.0, value=8.0, step=0.5
)
stage_fear = st.selectbox("Stage Fear:", options=[0, 1], index=1)
social_event_attendance = st.number_input(
    "Social Event Attendance (times per month):",
    min_value=0.0,
    max_value=30.0,
    value=3.0,
    step=1.0,
)
going_outside = st.number_input(
    "Going Outside:", min_value=0, max_value=15, value=1, step=1
)
drained_after_socializing = st.selectbox(
    "Drained After Socializing:", options=[0, 1], index=1
)
friends_circle_size = st.number_input(
    "Friends Circle Size (number of close friends):",
    min_value=0.0,
    max_value=50.0,
    value=3.0,
    step=1.0,
)
post_frequency = st.number_input(
    "Post Frequency:",
    min_value=0.0,
    max_value=20.0,
    value=2.0,
    step=1.0,
)

# Create user_input dictionary
user_input = {
    "time_spent_alone": [time_spent_alone],
    "stage_fear": [stage_fear],
    "social_event_attendance": [social_event_attendance],
    "going_outside": [going_outside],
    "drained_after_socializing": [drained_after_socializing],
    "friends_circle_size": [friends_circle_size],
    "post_frequency": [post_frequency],
}

# Convert to DataFrame
X_test = pd.DataFrame(user_input)

# Display the user input DataFrame
st.write("User Input:")
st.dataframe(X_test, hide_index=True)


# Load the model and scaler with streamlit caching
@st.cache_resource
def load_model_cached():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "artifacts", "persona_predictor.keras")
    model = load_model(model_path, compile=False)
    return model


@st.cache_resource
def load_scaler_cached():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scaler_path = os.path.join(base_dir, "artifacts", "scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return scaler


status_placeholder = st.empty()
status_placeholder.markdown("🟡**Model Status:** Loading")
model = load_model_cached()
scaler = load_scaler_cached()
status_placeholder.markdown("🟢**Model Status:** Ready")

# Button to make predictions
if st.button("Predict"):
    prediction_status = st.empty()
    prediction_status.info("🔍 Making predictions...")

    # Scale the input data
    X_test_scaled = scaler.transform(X_test)

    # Make predictions
    predictions = model.predict(X_test_scaled)

    # Display the predictions
    confidence = predictions[0][0] * 100
    if confidence > 50:
        prediction_status.success(
            f"The model predicts that the user is likely to be an extrovert with a confidence of {confidence:.2f}%."
        )
    else:
        prediction_status.warning(
            f"The model predicts that the user is likely to be an introvert with a confidence of {100 - confidence:.2f}%."
        )
