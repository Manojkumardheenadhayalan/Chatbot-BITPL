import pyrebase
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
import numpy as np
import os
import datetime
import openai

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("secret")

openai.api_key = os.environ.get("openai_api_key")
# Configure Firebase
firebase_config = {
   "apiKey": os.environ.get("FIREBASE_API_KEY"),
   "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN"),
   "databaseURL": os.environ.get("FIREBASE_DATABASE_URL"),
   "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET"),
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
db = firebase.database()
storage = firebase.storage()
[]
@app.route('/')
def index():
    return render_template('login.html')

@app.route('/signup', methods=['POST'])
def signup():
    # Handle sign-up logic here
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')

    try:
        # Create a new user in Firebase Authentication
        user = auth.create_user_with_email_and_password(email, password)

        # Store the user's information in the Firebase Realtime Database
        user_data = {
            'name': name,
            'email': email
        }
        db.child('users').child(user['localId']).set(user_data)

        flash('User created successfully', 'success')
        return redirect('/')
    except Exception as e:
        # Handle exceptions
        flash(str(e), 'error')
        return redirect('/')

@app.route('/signin', methods=['POST'])
def signin():
    # Handle sign-in logic here
    email = request.form.get('email')
    password = request.form.get('password')

    try:
        # Sign in the user using Firebase Authentication
        user = auth.sign_in_with_email_and_password(email, password)
        print(str(user))
        # user is authenticated, store user's email in the session
        session['email'] = email
        # User is authenticated, redirect to the medical form
        return redirect('/chat')

    except Exception as e:
        # Handle exceptions
        flash('Invalid login credentials. Please check your email and password.', 'error')
        return redirect('/')

@app.route('/logout', methods=['POST'])
def logout():
    # Remove the email from the session
    session.pop('email', None)
    # Redirect to the home page
    return render_template("login.html")

@app.route('/chat')
def chat():
    # Add any logic needed for the chat page
    return render_template('chat.html')

@app.route('/assist', methods=['POST'])
def assist():
    user_text = request.json.get('userText')

    try:
        # Make a request to the OpenAI API using the GPT-3.5 Turbo model
        response = openai.ChatCompletion.create(
                                        model="gpt-3.5-turbo",
                                        messages=[
                                            {"role": "system", "content": "You are a helpful assistant."},
                                            {"role": "user", "content": user_text},
                                        ],
                                        temperature=0,
                                    )['choices'][0]['message']['content']

        return jsonify({'text': response})
    except Exception as e:
        return jsonify({'text': f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)