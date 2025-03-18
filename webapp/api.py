import streamlit as st
import requests
 
st.write("""
# Chien ou chat
Bonjour ! Est-ce un *chien* ou bien un *chat* ?
""")

# Initialize session state if needed
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None

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

def give_feedback(result: bool):
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
        st.success(f"Feedback sent successfully: {response}")
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

st.button("Feedback positif", on_click=give_feedback, args=(True,))
st.button("Feedback négatif", on_click=give_feedback, args=(False,))