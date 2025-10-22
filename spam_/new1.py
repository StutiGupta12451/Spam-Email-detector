import streamlit as  st
from tensorflow.keras.models import load_model
import joblib
import numpy as np
model=load_model("spam/spam_email.keras")
try:
    vectorizer=joblib.load("spam_/vectorizer_spam.pkl")
except Exception as e2:
        print("Some Error occurred in loading the model")
def fun(email):
    try:
        transformed=vectorizer.transform([email]).toarray()
        pred=model.predict(transformed.reshape(1,-1))
        ans=(pred>0.5).astype("int32")
        return ans
    except Exception as e2:
        print("Some Error occurred in loading the model")
def feedback(email,ans):
    if ans==1:
        predans=0
    elif ans==0:
        predans=1
    if predans==0 or predans==1:
        transformed=vectorizer.transform([email]).toarray()
        pred=model.predict(transformed.reshape(1,-1))
        model.fit(transformed,np.array([predans]),epochs=1)
st.title("Spam Email detector")
st.markdown(
    """
    <style>
    .stTextInput>div>input {
        height: 50px; 
    }
    </style>
    """,
    unsafe_allow_html=True
)
email=st.text_input("Enter the Email")
ans=fun(email)
if "ans" not in st.session_state:
    st.session_state.ans = None

if st.button("Analyze"):
    if not email.strip():
        st.warning("Please enter an email before analyzing.")
    else:
        st.session_state.ans = fun(email)

if st.session_state.ans is not None:
    if st.session_state.ans == 1:
        st.warning("Spam detected")
    else:
        st.success("Ham (Not Spam)")

    st.text("Kindly help me improve the model by giving feedback:")
    option = st.selectbox(
        "Was the output correct?",
        ['Select an option...', 'Yes', 'No'],
        key="feedback_option"
    )
    if option == "No":
        feedback(email,ans)
    if option != 'Select an option...':
        st.info(f"Thanks for your feedback: {option}")

