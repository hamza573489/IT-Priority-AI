import streamlit as st
import joblib

# Add this to the top of your app.py
with st.sidebar:
    st.header("⚙️ Admin Settings")
    st.write("Model: Logistic Regression")
    st.write("Version: 1.0.2")
    st.divider()
    st.info("This tool automatically flags high-risk tickets for the IT Infrastructure team.")


# --- 1. LOAD THE BRAIN ---
# We load the files you see in your Solution Explorer
model = joblib.load('priority_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# --- 2. THE FRONTEND DESIGN ---
st.set_page_config(page_title="IT Priority AI", page_icon="??")

st.title("?? IT Support Priority Portal")
st.write("Enter your issue below to let the AI determine the priority.")

# User types here
user_input = st.text_area("Ticket Description:", placeholder="e.g., The server is down and crashing!")

if st.button("Analyze Priority"):
    if user_input:
        # The Backend processing
        vec = tfidf.transform([user_input])
        prediction = model.predict(vec)[0]
        
        # The Admin-level data calculation
        probability = model.predict_proba(vec).max() 

        # The Visual output
        colors = {"Critical": "red", "High": "orange", "Medium": "blue", "Low": "green"}
        st.subheader(f"AI Prediction: :{colors.get(prediction, 'gray')}[{prediction}]")
        
        # Showing the confidence (The 'Why' behind the AI's choice)
        st.write(f"Confidence Score: {probability:.2%}")
    else:
        st.warning("Please type something first!")
