import pyrebase
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
import numpy as np
import os
from supabase import create_client, Client
from datetime import datetime
import openai
from ctransformers import AutoModelForCausalLM
import requests

from dotenv import load_dotenv
load_dotenv()

model = AutoModelForCausalLM.from_pretrained("D:\LLM_models\TheBloke\Mistral-7B-OpenOrca-GGUF\mistral-7b-openorca.Q5_0.gguf", model_type="gpt2", )



url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

app = Flask(__name__)
app.secret_key = os.environ.get("secret")

openai.api_key = os.environ.get("openai_api_key")

# Configure Firebase
# firebase_config = {
#    "apiKey": os.environ.get("FIREBASE_API_KEY"),
#    "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN"),
#    "databaseURL": os.environ.get("FIREBASE_DATABASE_URL"),
#    "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET"),
# }

# firebase = pyrebase.initialize_app(firebase_config)
# auth = firebase.auth()
# db = firebase.database()
# storage = firebase.storage()

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/signup', methods=['POST'])
def signup():
    # Handle sign-up logic here
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')
    phone = request.form.get('phone_no')

    try:
        # Create a new user in Firebase Authentication
        
        res = supabase.auth.sign_up({
        "email": email,
        "password": password,
        "options": {
            "data": {
            "first_name": name,
            "phone_no": phone,
            }
        }
        })

        # # Store the user's information in the Firebase Realtime Database
        # user_data = {
        #     'name': name,
        #     'email': email
        # }
        # db.child('users').child(user['localId']).set(user_data)

        flash('Registered successfully. Verify your Email. Mail has been sent', 'success')
        return redirect('/')
    except Exception as e:
        # Handle exceptions
        flash(str(e), 'error')
        return redirect('/')

@app.route('/signin', methods=['POST'])
def signin():
    
    try:
        if request.method == 'POST':
            # Handle form submission
            email = request.form.get('email')
            password = request.form.get('password')
            # Use the email and password to authenticate the user
            # Redirect to the home page or dashboard
        
            data = supabase.auth.sign_in_with_password({"email": email, "password": password})
            print(data)
            # user is authenticated, store user's email in the session
            session['email'] = email
            
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            table_name = f"{email}-{timestamp}"  # Create the table name
            
            # Create a new table with the format "mailID-timestamp"
            create_table_query = f"""
                CREATE TABLE {table_name} (
                    prompt TEXT,
                    answer TEXT
                );
            """

            try:
                # Execute the query to create the new table
                supabase.query(create_table_query)
            except Exception as e:
                return f"Login successful, but table creation failed: {str(e)}"
            # User is authenticated, redirect to the chat page
            return redirect(url_for('chat', table_name=table_name))
        else:
            provider = request.args.get('provider')
            if provider == 'google':
                # Handle Google sign-in
                return redirect(f"{url}/auth/v1/authorize?provider=google")
            elif provider == 'twitter':
                # Handle Twitter sign-in
                # Replace the URL with the actual Twitter sign-in URL
                return redirect("twitter_signin_url")
            elif provider == 'linkedin':
                # Handle LinkedIn sign-in
                # Replace the URL with the actual LinkedIn sign-in URL
                return redirect("linkedin_signin_url")

    except Exception as e:
        # Handle exceptions
        flash('Invalid login credentials. Please check your email and password.', 'error')
        return redirect('/')

@app.route('/logout', methods=['POST'])
def logout():
    # Remove the email from the session
    
    res = supabase.auth.sign_out()
    session.pop('email', None)
    # Redirect to the home page
    return render_template("login.html")

@app.route('/chat')
def chat():
    table_name = request.args.get('table_name')
    # Add any logic needed for the chat page
    return render_template('chat.html', table_name=table_name)

@app.route('/assist', methods=['POST'])
def assist():
    user_text = request.json.get('userText')
    table_name=request.json.get('table_name')

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
        


        # def query(payload):
        #     response = requests.post(API_URL, headers=headers, json=payload)
        #     return response.json()
            
        # response = query({
        #     "inputs": user_text,
        # })[0]['generated_text']
        
        data = {"prompt": user_text, "answer": response}
        response = response
        
        try:
            supabase.table(table_name).upsert([data])
        except:
            flash('Unable to insert', 'error')

        return jsonify({'text': response})
    except Exception as e:
        return jsonify({'text': f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)