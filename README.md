# Spam Email Detector using Streamlit and Keras

A simple yet powerful web application that detects whether an email is **Spam** or **Ham (Not Spam)** using a pre-trained deep learning model.

The project combines:
- **TensorFlow (Keras)** for spam classification
- **Scikit-learn (TF-IDF Vectorizer)** for text transformation
- **Streamlit** for a clean and interactive web interface

---

## Features

Classifies emails as **Spam** or **Ham** in real-time  
Uses a **trained deep learning model (`spam_email.keras`)**  
TF-IDF vectorization using **`vectorizer_spam.pkl`**  
**97% accuracy** on the test dataset  
Feedback mechanism to retrain the model on user corrections  
Simple and elegant UI using **Streamlit**

---

## How It Works

1. The input email text is converted into numerical features using the **TF-IDF vectorizer**.
2. These features are passed into the **Keras model (`spam_email.keras`)** for prediction.
3. The output is binary — `1` for spam and `0` for ham.
4. Users can provide feedback if the prediction is incorrect, allowing incremental retraining.

---

##  Installation

### 1️⃣ Clone this repository
```bash git clone https://github.com/<your-username>/spam-email-detector.git ```
```bash cd spam-email-detector```
###2️⃣ Install dependencies
```bash pip install -r requirements.txt ```
###3️⃣Add your model files

Place the following files in the project folder:

spam_email.keras (your trained Keras model)

vectorizer_spam.pkl (your saved TF-IDF vectorizer)

4️⃣ Run the Streamlit app
```bash streamlit run app.py```

## File Structure
spam-email-detector/
│
├── app.py                  # Streamlit app
├── spam_email.keras        # Trained Keras model
├── vectorizer_spam.pkl     # TF-IDF vectorizer
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
## Feedback

The app includes a feedback option that lets users report incorrect classifications.
This feedback can be used to incrementally retrain the model, improving accuracy over time.

