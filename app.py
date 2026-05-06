import streamlit as st
import joblib

# --- 1. ADMIN & SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Admin Settings")
    st.write("Model: Logistic Regression (Weighted)")
    st.write("Version: 1.0.5")
    st.divider()
    st.info("Hybrid AI: Rules handle known categories; the AI Brain handles everything else.")

# --- 2. LOAD THE BRAIN ---
model = joblib.load('priority_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# --- 3. FRONTEND DESIGN ---
st.set_page_config(page_title="IT Priority AI Portal", page_icon="🎫")
st.title("IT Support Priority Portal")
st.markdown("Enter your issue below. The system uses a mix of AI and business logic to decide priority.")

user_input = st.text_area("Ticket Description:", placeholder="Describe the issue here...")

if st.button("Analyze Priority"):
    if user_input:
        # STEP 1: THE AI BRAIN READS THE TEXT
        # This part handles 'anything' you type by looking at word patterns
        vec = tfidf.transform([user_input])
        prediction = model.predict(vec)[0]
        probability = model.predict_proba(vec).max()
        text_lower = user_input.lower()

        # STEP 2: DEFINE THEMES (Human Logic)
        danger_theme = ['ransomware', 'attack', 'hacked', 'breach', 'security', 'fire']
        hardware_theme = ['laptop', 'blue-screen', 'flickering', 'crashing', 'slow', 'freezing', 'screen']
        routine_theme = ['password', 'mouse', 'monitor', 'keyboard', 'login', 'reset', 'email']
        outage_theme = ['server', 'down', 'database', 'offline']

        # STEP 3: THE SMART LOGIC LAYERS
        
        # Layer 1: Force Critical for Security Disasters
        if any(word in text_lower for word in danger_theme):
            prediction = 'Critical'
            probability = 1.0
            
        # Layer 2: Force High for Hardware/Performance Issues
        elif any(word in text_lower for word in hardware_theme):
            prediction = 'High'
            # If the AI was already somewhat sure, we keep its score; otherwise, we set it to 85%
            probability = max(probability, 0.85)

        # Layer 3: Force Medium for Routine office tasks
        elif any(word in text_lower for word in routine_theme):
            prediction = 'Medium'
            probability = 1.0

        # Layer 4: Fallback (THE "READ ANYTHING" LAYER)
        # If no keywords are found, the 'prediction' and 'probability' 
        # from Step 1 are used exactly as the AI calculated them.
        else:
            # We don't change anything here. The AI's natural guess is shown.
            pass

        # STEP 4: VISUAL OUTPUT
        colors = {"Critical": "red", "High": "orange", "Medium": "blue", "Low": "green"}
        st.divider()
        st.subheader(f"AI Prediction: :{colors.get(prediction, 'gray')}[{prediction}]")
        st.write(f"**Confidence Score:** {probability:.2%}")        
        
        # Feedback messages
        if prediction == "Critical":
            st.error("🚨 Critical Alert: Immediate intervention required.")
        elif prediction == "High":
            st.warning("⚠️ High Priority: Technical support assigned.")
        elif prediction == "Medium":
            st.info("ℹ️ Standard Request: Will be processed in order.")
        else:
            st.success("✅ Low Priority: General inquiry or minor task.")

    else:
        st.warning("Please enter some text first.")
