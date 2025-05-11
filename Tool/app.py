from flask import Flask, render_template, request
import pickle
import re
import pyttsx3
import threading

# Initialize Flask
app = Flask(__name__)

# Load model and vectorizer
with open('phishing_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Clean email text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Speak the result
def speak_output(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except RuntimeError:
        pass

@app.route('/')
def home():
    return render_template('index.html', prediction=None, label=None, email_text='')

@app.route('/predict', methods=['POST'])
def predict():
    email = request.form['email_text']
    cleaned = clean_text(email)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0][pred]

    label = 'SAFE' if pred == 0 else 'SPAM'
    message = f"This email is {label} with {prob * 100:.2f}% confidence."

    threading.Thread(target=speak_output, args=(message,), daemon=True).start()

    return render_template("index.html", prediction=message, label=label, email_text=email)

if __name__ == '__main__':
    import webbrowser
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True)