import streamlit as st
import requests

# Initialize session state
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None
if 'classes' not in st.session_state:
    st.session_state.classes = []

# Fetch available classes from the API
@st.cache_data(ttl=3600)  # Cache the result for 1 hour
def fetch_classes():
    try:
        response = requests.get('http://serving-api:8080/classes')
        return response.json()['classes']
    except Exception as e:
        st.error(f"Error fetching classes: {str(e)}")
        return ["unknown"]

# Get classes at startup
st.session_state.classes = fetch_classes()

st.write(f"""
# Chien ou chat
Bonjour ! Est-ce un *{st.session_state.classes[0]}* ou bien un *{st.session_state.classes[1]}* ?
""")

uploaded_file = st.file_uploader("Pick a file", accept_multiple_files=False)

def give_prediction():
    if uploaded_file is not None:
        try:
            url = 'http://serving-api:8080/predict'
            files = {'data': uploaded_file}
            r = requests.post(url, files=files)
            st.session_state.prediction_data = r.json() # exemple {'id': '1234', 'prediction': 'dog'} 
            print(f"Processed file: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

def give_feedback(result):
    if st.session_state.prediction_data is None:
        st.error("Please make a prediction first before giving feedback!")
        return
    
    try:
        print(f"Feedback received: {result} for prediction {st.session_state.prediction_data}")
        url = 'http://serving-api:8080/feedback'
        feedback = {
            'id_image': st.session_state.prediction_data['id'], 
            'data': result
        }
        r = requests.post(url, json=feedback)
        response = r.json()
        
        # Clear prediction data after sending feedback
        if r.status_code == 200:
            st.session_state.prediction_data = None
        else:
            st.error(f"Error from server: {response}")
    except Exception as e:
        st.error(f"Error sending feedback: {str(e)}")

st.button("Prédire image", on_click=give_prediction)

# Display prediction result
if st.session_state.prediction_data:
    st.subheader("Résultat de la prédiction:")
    prediction = st.session_state.prediction_data.get('prediction', 'unknown')
    st.write(f"Prédiction: {prediction}")
    
    # Display the image nicely with the prediction
    if uploaded_file is not None:
        st.image(uploaded_file, caption=f"Classified as: {prediction}")
    
    st.selectbox(
        "Quelle est la vraie classe de cette image ?",
        options=st.session_state.classes,
        key="selected_class"
    )
    st.button("Envoyer le feedback", on_click=give_feedback, args=(st.session_state.selected_class,))