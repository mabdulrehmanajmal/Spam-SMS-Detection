
import streamlit as st
import joblib

bundle = joblib.load('spam_sms_model.pkl')

model = bundle['model']
tfidf = bundle['tfidf']
model_name = bundle['model_name']

st.set_page_config(
    page_title="Spam SMS Detection",
    page_icon="📩",
    layout="centered"
)

st.title("📩 Spam SMS Detection System")

st.write("Enter an SMS message below to check whether it is Spam or Ham.")

st.info(f"Machine Learning Model: {model_name}")

message = st.text_area("Enter Your SMS Message")

if st.button("Predict"):
    if message.strip():
        transformed_message = tfidf.transform([message])
        prediction = model.predict(transformed_message)[0]

        if prediction == 'spam':
            st.error("Spam Message Detected!")
        else:
            st.success("This is a Ham (Legitimate) Message.")

    else:
        st.warning("Please enter an SMS message.")
