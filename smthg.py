import streamlit as st
import numpy as np
import joblib
import requests
import pandas as pd  # Import pandas for Excel functionality

# Load the trained models
@st.cache_resource
def load_model1():
    return joblib.load("C:/Users/suhas/Downloads/wie code/random_forest_model.joblib")

@st.cache_resource
def load_model2():
    return joblib.load("C:/Users/suhas/Downloads/wie code/demand_prediction_model.joblib")  # Replace with actual path

model1 = load_model1()
model2 = load_model2()

# Feature lists for both pages
features_page1 = [
    'generation biomass', 'generation fossil brown coal/lignite',
    'generation fossil coal-derived gas', 'generation fossil gas',
    'generation fossil hard coal', 'generation fossil oil',
    'generation fossil oil shale', 'generation fossil peat',
    'generation geothermal', 'generation hydro pumped storage aggregated',
    'generation hydro pumped storage consumption', 'generation hydro run-of-river and poundage',
    'generation hydro water reservoir', 'generation marine',
    'generation nuclear', 'generation other', 'generation other renewable',
    'generation solar', 'generation waste', 'generation wind offshore',
    'generation wind onshore', 'forecast solar day ahead',
    'forecast wind offshore eday ahead', 'forecast wind onshore day ahead',
    'total load forecast', 'total load actual', 'price day ahead'
]

features_page2 = [
    'generation biomass', 'generation fossil brown coal/lignite',
    'generation fossil gas', 'generation hydro run-of-river and poundage',
    'generation solar', 'generation wind onshore', 'price day ahead',
    'year', 'month', 'day', 'hour'
]

# Tabs for navigation
tab1, tab2 = st.tabs(["Energy Generation Prediction", "Demand Prediction"])

# Function to call Gemini API for explanation
def explain_predictions(demand, generation):
    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    api_key = "AIzaSyAqmyZRoO1uJiyz8TDQycWrtt8ZAfssODs"  # Replace with your actual API key
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{
            "parts": [{"text": f"Explain how demand {demand} relates to generation {generation}"}]
        }]
    }
    try:
        response = requests.post(f"{api_url}?key={api_key}", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()  # Adjust based on how the API returns data
        else:
            return f"API call failed: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error calling API: {str(e)}"

def save_to_excel(data, filename):
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)

with tab1:
    st.title("Energy Generation Prediction")
    st.markdown("""This application predicts energy generation using various features...""")

    # Input form for Page 1
    user_input1 = {}
    with st.form(key="form1"):
        for feature in features_page1:
            user_input1[feature] = st.number_input(f"{feature}", value=0.0)
        submit1 = st.form_submit_button("Predict")

    if submit1:
        input_values1 = np.array([user_input1[feature] for feature in features_page1]).reshape(1, -1)
        try:
            prediction1 = model1.predict(input_values1)
            st.session_state.generation_value = prediction1[0] if prediction1.size > 0 else None
            st.success(f"Predicted Generation: {st.session_state.generation_value}")

            # Store the data for Excel
            if 'data_storage' not in st.session_state:
                st.session_state.data_storage = []
            st.session_state.data_storage.append({
                **user_input1,
                "Predicted Generation": st.session_state.generation_value,
                "Prediction Type": "Generation"
            })

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

with tab2:
    st.title("Demand Prediction and Generation")
    st.markdown("""This application predicts energy demand based on selected features...""")

    # Input form for Page 2
    user_input2 = {}
    with st.form(key="form2"):
        for feature in features_page2:
            user_input2[feature] = st.number_input(f"{feature}", value=0.0)
        submit2 = st.form_submit_button("Predict Demand")

    if submit2:
        input_values2 = np.array([user_input2[feature] for feature in features_page2]).reshape(1, -1)
        try:
            prediction2 = model2.predict(input_values2)
            demand_value = prediction2[0] if prediction2.size > 0 else None
            
            # Check if generation_value is available in session_state
            if 'generation_value' in st.session_state:
                generation_value = st.session_state.generation_value
                # Combine both predictions for explanation
                explanation = explain_predictions(demand_value, generation_value)

                st.success(f"Predicted Demand: {demand_value}")
                st.info(f"Explanation: {explanation}")

                # Store the data for Excel
                if 'data_storage' not in st.session_state:
                    st.session_state.data_storage = []
                st.session_state.data_storage.append({
                    **user_input2,
                    "Predicted Demand": demand_value,
                    "Predicted Generation": generation_value,
                    "Prediction Type": "Demand"
                })

            else:
                st.error("Generation value is not available. Please predict generation first.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Button to save data to Excel
if st.button("Save to Excel"):
    if 'data_storage' in st.session_state and st.session_state.data_storage:
        save_to_excel(st.session_state.data_storage, "C:/Users/suhas/Downloads/wie code/predictions.xlsx")
        st.success("Data saved to predictions.xlsx")
    else:
        st.warning("No data to save.")
