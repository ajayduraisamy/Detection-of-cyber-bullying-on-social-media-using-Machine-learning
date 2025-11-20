from flask import Flask, jsonify, render_template, request, redirect, url_for, session, flash, Response
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import sqlite3
from datetime import datetime
import pandas as pd
import pickle

app = Flask(__name__)
app.secret_key = 'wverihdfuvuwi2482'

DB_PATH = "users.db"

# ============================
# DATABASE CONNECTION FUNCTION
# ============================
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ============================
# CREATE TABLE IF NOT EXISTS
# ============================
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            number TEXT NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()  # initialize DB at app start

@app.context_processor
def inject_current_year():
    return {'current_year': datetime.now().year}




# ============================
# REGISTER
# ============================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        number = request.form['number']
        password = request.form['password']

        hashed_password = generate_password_hash(password)

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                'INSERT INTO users (name, email, number, password) VALUES (?, ?, ?, ?)',
                (name, email, number, hashed_password)
            )
            conn.commit()
            flash('Registration successful. Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email already exists.', 'danger')
        finally:
            conn.close()

    return render_template('register.html', title="Register")

# ============================
# LOGIN
# ============================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session['email'] = user["email"]
            session['name'] = user["name"]
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'danger')

    return render_template('login.html', title="Login")

# ============================
# DASHBOARD 
# ============================

import joblib



model = joblib.load("cyberbullying_model.joblib")
vectorizer = joblib.load("cyber_vectorizer.joblib")

@app.route("/", methods=["GET", "POST"])
def dashboard():
    if 'email' not in session:
        flash('Please login.', 'warning')
        return redirect(url_for('login'))

    prediction_result = None

    if request.method == "POST":
        user_text = request.form.get("user_text")

        if user_text and user_text.strip():
            transformed = vectorizer.transform([user_text])
            result = model.predict(transformed)[0]

            
            if result == 1:
                prediction_result = "🚨 ALERT: Potential Cyberbullying Detected - Action Recommended."
            else:
                prediction_result = "🟩 Status: No signs of cyberbullying detected."
        else:
            prediction_result = " Please enter text before submitting."

    return render_template("dashboard.html", result=prediction_result)



# ============================
# CONTACT PAGE
# ============================
@app.route('/contact')
def contact():
    if 'email' not in session:
        flash('Please login to view this page.', 'warning')
        return redirect(url_for('login'))
    return render_template('contact.html')

# ============================
# PROFILE PAGE
# ============================
@app.route('/profile')
def profile():
    if 'email' not in session:
        flash('Please login to view profile.', 'warning')
        return redirect(url_for('login'))

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (session['email'],))
    user = cursor.fetchone()
    conn.close()

    if not user:
        flash("User not found.", "danger")
        return redirect(url_for('login'))

    return render_template('profile.html', user=user)

# ============================
# LOGOUT
# ============================
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# ============================
# RUN SERVER
# ============================
if __name__ == '__main__':
    app.run(debug=True)
