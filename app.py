from flask import Flask, render_template, request, session, redirect, url_for, flash
import pickle
import numpy as np
import pandas as pd
from collections import Counter

app = Flask(__name__)
app.secret_key = 'abcd123'

# Load trained model and scaler
model = pickle.load(open('model.pickle', 'rb'))
scaler = pickle.load(open('scaler.pickle', 'rb'))
users = {}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/user_registration', methods=['GET', 'POST'])
def user_registration():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('user_registration'))

        if username not in users:
            users[username] = {'password': password}
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('user_login'))
        else:
            flash('User already exists', 'error')
            return redirect(url_for('user_registration'))

    return render_template('user_registration.html')

@app.route('/user_login', methods=['GET', 'POST'])
def user_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username]['password'] == password:
            session['username'] = username
            flash('Successfully logged in', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
            return redirect(url_for('user_login'))

    return render_template('user_login.html')

@app.route('/index')
def index():
    if 'username' in session:
        return render_template('index.html')
    else:
        flash('You need to log in first', 'error')
        return redirect(url_for('user_login'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
            required_fields = [
                'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
                'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
                'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
                'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
                'spread1', 'spread2', 'D2', 'PPE'
            ]

            # Check for missing or empty fields
            for field in required_fields:
                if field not in request.form or request.form[field].strip() == "":
                    return f"Missing or empty field: {field}", 400

            # Parse inputs to float
            data = [float(request.form[field]) for field in required_fields]

            # Scale the input
            data_scaled = scaler.transform([data])
            prediction = model.predict(data_scaled)[0]

            parkinsons_labels = {
                0: "Patient is Healthy",
                1: "Patient Suffered with Parkinsons"
            }
            predicted_label = parkinsons_labels.get(prediction, "Unknown")
            return render_template('result.html', parkinsons=predicted_label)

        return redirect(url_for('index'))

    except ValueError as ve:
        return f"Value error (non-numeric input?): {str(ve)}", 400
    except Exception as e:
        return f"Server error: {str(e)}", 500

@app.route('/performance')
def performance():
    df = pd.read_csv('test.csv')
    disorder_counts = Counter(df['status'])
    labels = list(disorder_counts.keys())    
    values = list(disorder_counts.values())  
    return render_template('performance.html', labels=labels, values=values)

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)
