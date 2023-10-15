#Version - 1

"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from transformers import BertForSequenceClassification, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import torch

from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util
import sqlite3
import openai
import threading

app = Flask(__name__)

# Load pretrained GPT-2 model and tokenizer
generation_model = GPT2LMHeadModel.from_pretrained('gpt2')
generation_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load pretrained Sentence-BERT model
retrieval_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load the saved BERT model and tokenizer
saved_model_path = "./content"  # Replace with the actual path
model = BertForSequenceClassification.from_pretrained(saved_model_path)
tokenizer = BertTokenizer.from_pretrained(saved_model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# OpenAI API Key
openai.api_key = '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'

# Define a thread-local storage for database connections and cursors
thread_local = threading.local()

# Database connection functions
def get_db_connection():
    if not hasattr(thread_local, "db_connection"):
        thread_local.db_connection = sqlite3.connect('responses.db')
    return thread_local.db_connection

def get_db_cursor():
    if not hasattr(thread_local, "db_cursor"):
        thread_local.db_cursor = get_db_connection().cursor()
    return thread_local.db_cursor

# Function for Sentence-BERT retrieval
def sbert_retrieval(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT id, prompt, response FROM responses")
    results = cursor.fetchall()
    
    # Implement sbert_retrieval logic here
    # ...

# Function for RAG response generation
def generate_rag_response(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT prompt, response FROM responses WHERE prompt = ?", (user_question,))
    result = cursor.fetchone()
    
    # Implement generate_rag_response logic here
    # ...

# Function to save response to the database
def save_response_to_database(prompt, response):
    cursor = get_db_cursor()
    
    # Create a table if it doesn't exist
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY,
            prompt TEXT,
            response TEXT
        )'''
    )

    # Insert response into the table
    cursor.execute('INSERT INTO responses (prompt, response) VALUES (?, ?)', (prompt, response))
    get_db_connection().commit()

# Sample data for sentiment analysis
data = [
    ("Positive", "I love this product! It's amazing."),
    ("Negative", "This is terrible. I'm very disappointed."),
    ("Positive", "The service was excellent and exceeded my expectations."),
    ("Negative", "I regret buying this. Waste of money."),
    # Add more data examples
]

# Separate data into intent labels and text
intents, texts = zip(*data)

# Convert intent labels into binary labels (1 for Positive, 0 for Negative)
labels = np.array([1 if intent == "Positive" else 0 for intent in intents])

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust max_features as needed

# Transform text data into TF-IDF features
tfidf_texts = tfidf_vectorizer.fit_transform(texts)

# Create a Support Vector Machine (SVM) classifier
svm_classifier = SVC(kernel='linear', random_state=42)

# Train the classifier
svm_classifier.fit(tfidf_texts, labels)

@app.route("/", methods=['POST', 'GET'])
def query_view():
    if request.method == 'POST':
        prompt = request.form['prompt']

        # Sentiment Analysis
        user_input = prompt
        tfidf_user_input = tfidf_vectorizer.transform([user_input])
        predicted_label = svm_classifier.predict(tfidf_user_input)
        sentiment_intent = "Positive" if predicted_label[0] == 1 else "Negative"

        response = ""
        if sentiment_intent == 'Positive':
            # Generate RAG Response
            rag_response = generate_rag_response(prompt)
            if rag_response is not None:
                # Determine Technical or Non-Technical
                if any(keyword in rag_response.lower() for keyword in ['error', 'bug', 'technical']):
                    intent = 'Technical'
                else:
                    intent = 'Non-Technical'
                response = f"{sentiment_intent}, {intent}: {rag_response}"
            else:
                # If RAG response is None, send the question to OpenAI API
                openai_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )['choices'][0]['message']['content']

                # Save the response to the database
                save_response_to_database(prompt, openai_response)

                response = f"{sentiment_intent}, {openai_response}"
            
        else:
            response = f"{sentiment_intent}, Non-Technical Sorry,Cannot answer the Question"

        return jsonify({'response': response})

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
"""

#Version - 2

"""
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import torch

from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import sqlite3
import openai
import threading

app = Flask(__name__)

# Load pretrained GPT-2 model and tokenizer
generation_model = GPT2LMHeadModel.from_pretrained('gpt2')
generation_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load pretrained Sentence-BERT model
retrieval_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load the saved BERT model and tokenizer
saved_model_path = "./content"  # Replace with the actual path
model = BertForSequenceClassification.from_pretrained(saved_model_path)
tokenizer = BertTokenizer.from_pretrained(saved_model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# OpenAI API Key
openai.api_key = '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'

# Define a thread-local storage for database connections and cursors
thread_local = threading.local()

# Database connection functions
def get_db_connection():
    if not hasattr(thread_local, "db_connection"):
        thread_local.db_connection = sqlite3.connect('responses.db')
    return thread_local.db_connection

def get_db_cursor():
    if not hasattr(thread_local, "db_cursor"):
        thread_local.db_cursor = get_db_connection().cursor()
    return thread_local.db_cursor

# Function for Sentence-BERT retrieval
def sbert_retrieval(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT id, prompt, response FROM responses")
    results = cursor.fetchall()
    
    # Implement sbert_retrieval logic here
    # ...

# Function for RAG response generation
def generate_rag_response(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT prompt, response FROM responses WHERE prompt = ?", (user_question,))
    result = cursor.fetchone()
    
    # Implement generate_rag_response logic here
    # ...

# Function to save response to the database
def save_response_to_database(prompt, response):
    cursor = get_db_cursor()
    
    # Create a table if it doesn't exist
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY,
            prompt TEXT,
            response TEXT
        )'''
    )

    # Insert response into the table
    cursor.execute('INSERT INTO responses (prompt, response) VALUES (?, ?)', (prompt, response))
    get_db_connection().commit()

@app.route("/", methods=['POST', 'GET'])
def query_view():
    if request.method == 'POST':
        prompt = request.form['prompt']

        # Sentiment Analysis using BERT
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).cpu().numpy()
        sentiment_intent = "Positive" if predicted_label[0] == 1 else "Negative"

        response = ""
        if sentiment_intent == 'Positive':
            # Sentence-BERT Retrieval
            sbert_response = sbert_retrieval(prompt)
            if sbert_response is not None:
                response = f"{sentiment_intent}, {sbert_response}"
            else:
                # Generate RAG Response
                rag_response = generate_rag_response(prompt)
                if rag_response is not None:
                    # Determine Technical or Non-Technical
                    if any(keyword in rag_response.lower() for keyword in ['error', 'bug', 'technical']):
                        intent = 'Technical'
                    else:
                        intent = 'Non-Technical'
                    response = f"{sentiment_intent}, {intent}: {rag_response}"
                else:
                    # If RAG response is None, send the question to OpenAI API
                    openai_response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0,
                    )['choices'][0]['message']['content']

                    # Save the response to the database
                    save_response_to_database(prompt, openai_response)

                    response = f"{sentiment_intent}, {openai_response}"
        else:
            response = f"{sentiment_intent}, Non-Technical: Sorry, cannot answer the question"

        return jsonify({'response': response})

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
"""
#Version - 3
"""
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import torch

from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import sqlite3
import openai
import threading

app = Flask(__name__)

# Load pretrained GPT-2 model and tokenizer
generation_model = GPT2LMHeadModel.from_pretrained('gpt2')
generation_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load pretrained Sentence-BERT model
retrieval_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load the saved BERT model and tokenizer
saved_model_path = "./content"  # Replace with the actual path
model = BertForSequenceClassification.from_pretrained(saved_model_path)
tokenizer = BertTokenizer.from_pretrained(saved_model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# OpenAI API Key
openai.api_key = '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'

# Define a thread-local storage for database connections and cursors
thread_local = threading.local()

# Database connection functions
def get_db_connection():
    if not hasattr(thread_local, "db_connection"):
        thread_local.db_connection = sqlite3.connect('responses.db')
    return thread_local.db_connection

def get_db_cursor():
    if not hasattr(thread_local, "db_cursor"):
        thread_local.db_cursor = get_db_connection().cursor()
    return thread_local.db_cursor

# Function for Sentence-BERT retrieval
def sbert_retrieval(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT id, prompt, response FROM responses")
    results = cursor.fetchall()
    
    # Implement sbert_retrieval logic here
    # ...

# Function for RAG response generation
def generate_rag_response(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT prompt, response FROM responses WHERE prompt = ?", (user_question,))
    result = cursor.fetchone()
    
    # Implement generate_rag_response logic here
    # ...

# Function to save response to the database
def save_response_to_database(prompt, response):
    cursor = get_db_cursor()
    
    # Create a table if it doesn't exist
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY,
            prompt TEXT,
            response TEXT
        )'''
    )

    # Insert response into the table
    cursor.execute('INSERT INTO responses (prompt, response) VALUES (?, ?)', (prompt, response))
    get_db_connection().commit()

# Threshold for sentiment classification confidence
SENTIMENT_THRESHOLD = 0.6

@app.route("/", methods=['POST', 'GET'])
def query_view():
    if request.method == 'POST':
        prompt = request.form['prompt']

        # Sentiment Analysis using BERT
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).cpu().numpy()
        sentiment_confidence = probabilities[0][predicted_label[0]].item()
        sentiment_intent = "Positive" if predicted_label[0] == 1 and sentiment_confidence > SENTIMENT_THRESHOLD else "Negative"

        response = ""
        if sentiment_intent == 'Positive':
            # Sentence-BERT Retrieval
            sbert_response = sbert_retrieval(prompt)
            if sbert_response is not None:
                response = f"{sentiment_intent}, {sbert_response}"
            else:
                # Generate RAG Response
                rag_response = generate_rag_response(prompt)
                if rag_response is not None:
                    # Determine Technical or Non-Technical
                    if any(keyword in rag_response.lower() for keyword in ['error', 'bug', 'technical']):
                        intent = 'Technical'
                    else:
                        intent = 'Non-Technical'
                    response = f"{sentiment_intent}, {intent}: {rag_response}"
                else:
                    # If RAG response is None, send the question to OpenAI API
                    openai_response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0,
                    )['choices'][0]['message']['content']

                    # Save the response to the database
                    save_response_to_database(prompt, openai_response)

                    response = f"{sentiment_intent}, {openai_response}"
        else:
            response = f"{sentiment_intent}, Non-Technical: Sorry, cannot answer the question"

        return jsonify({'response': response})

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
"""
#Version - 4
"""
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import torch

from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import sqlite3
import openai
import threading
import re

app = Flask(__name__)

# Load pretrained GPT-2 model and tokenizer
generation_model = GPT2LMHeadModel.from_pretrained('gpt2')
generation_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load pretrained Sentence-BERT model
retrieval_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load the saved BERT model and tokenizer
saved_model_path = "./content"  # Replace with the actual path
model = BertForSequenceClassification.from_pretrained(saved_model_path)
tokenizer = BertTokenizer.from_pretrained(saved_model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# OpenAI API Key
openai.api_key = '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'

# Define a thread-local storage for database connections and cursors
thread_local = threading.local()

# Define a list of negative words and patterns
negative_words = [
    r'\bnot\b', r'\bwouldn\'t\b', r'\bdon\'t\b', r'\bshouldn\'t\b', r'\bdoesn\'t\b', r'\bwould not\b',
    r'\bdo not\b', r'\bshould not\b', r'\baren\'t\b', r'\bare not\b', r'\bhasn\'t\b', r'\bhas not\b',
    r'\bhaven\'t\b', r'\bhave not\b', r'\bhadn\'t\b', r'\bhad not\b', r'\bnever\b', r'\bno\b',
    r'\bnegative\b', r'\bdisapprove\b', r'\bunhappy\b', r'\bfail\b', r'\bfailure\b', r'\bwrong\b',
    r'\bhate\b', r'\bregret\b', r'\bcan\'t\b', r'\bcannot\b', r'\bwon\'t\b', r'\bwouldn\'t\b',
    r'\bdoesn\'t\b', r'\baren\'t\b', r'\bain\'t\b', r'\bwasn\'t\b', r'\bwere\'n\'t\b', r'\bhasn\'t\b',
    r'\bhaven\'t\b', r'\bhadn\'t\b', r'\bwon\'t\b', r'\bwouldn\'t\b', r'\bshouldn\'t\b', r'\bmistake\b',
    r'\bsad\b', r'\bworst\b', r'\bterrible\b', r'\bhorrible\b', r'\bdreadful\b', r'\bunpleasant\b',
    r'\bdisgusting\b', r'\bdislike\b', r'\bdispleasure\b', r'\bunhappy\b', r'\bunfortunate\b', r'\bneglect\b',
    r'\bunfortunate\b', r'\bunfavorable\b', r'\bawful\b', r'\bnasty\b', r'\bpathetic\b', r'\btragic\b',
    r'\bfault\b', r'\bfiasco\b', r'\bregrettable\b', r'\bsorry\b', r'\bupset\b', r'\bdissatisfied\b',
    r'\bdismal\b', r'\bdepressing\b', r'\bworry\b', r'\bhurt\b', r'\bfrown\b', r'\bcry\b', r'\bcrisis\b',
    r'\bproblem\b', r'\bdanger\b', r'\bdefect\b', r'\bdisaster\b', r'\bfear\b', r'\bpain\b', r'\bshame\b',
    r'\bhopeless\b', r'\bdoubt\b', r'\binsult\b', r'\bnervous\b', r'\bthreat\b', r'\bguilt\b', r'\bweak\b',
    r'\bhorror\b', r'\bshocking\b', r'\bawful\b', r'\brepulsive\b', r'\batrocious\b', r'\bnegative\b',
    r'\bdiscomfort\b', r'\binconvenience\b', r'\bdismay\b', r'\bdespair\b', r'\bdisgust\b', r'\bdespise\b',
    r'\bdiscontent\b', r'\bdispleasure\b', r'\bworry\b', r'\bfear\b', r'\banxiety\b', r'\binsecurity\b',
    r'\bsuspicion\b', r'\bdoubt\b', r'\buncertain\b', r'\bguilt\b', r'\bremorse\b', r'\bshame\b',
    r'\bembarrassment\b', r'\bdismal\b', r'\bdejected\b', r'\bdowncast\b', r'\bsorrow\b', r'\bmelancholy\b',
    r'\bdesolate\b', r'\bdespair\b', r'\bgrief\b', r'\banguish\b', r'\bregretful\b', r'\bheartbroken\b',
    r'\bsadness\b', r'\bunhappy\b', r'\bglum\b', r'\bmiserable\b', r'\bwretched\b', r'\btroubled\b',
    r'\bdistressed\b', r'\bsuffering\b', r'\bpainful\b', r'\bsadly\b', r'\bsorry\b', r'\bsadly\b',
    r'\bunfortunately\b', r'\bhorrifying\b', r'\bgrim\b', r'\bsomber\b', r'\bdeplorable\b', r'\bagonizing\b',
    r'\bmournful\b', r'\bdreary\b', r'\bdisastrous\b', r'\bwoeful\b', r'\bwoe\b', r'\bsorrowful\b',
    r'\bdepress\b', r'\bstress\b', r'\btrouble\b', r'\bnervousness\b', r'\bconcern\b', r'\bapprehension\b'
]


# Combine negative words into a single regex pattern
negative_pattern = re.compile('|'.join(negative_words), re.IGNORECASE)

# Database connection functions
def get_db_connection():
    if not hasattr(thread_local, "db_connection"):
        thread_local.db_connection = sqlite3.connect('responses.db')
    return thread_local.db_connection

def get_db_cursor():
    if not hasattr(thread_local, "db_cursor"):
        thread_local.db_cursor = get_db_connection().cursor()
    return thread_local.db_cursor

# Function for Sentence-BERT retrieval
def sbert_retrieval(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT id, prompt, response FROM responses")
    results = cursor.fetchall()
    
    # Implement sbert_retrieval logic here
    # ...

# Function for RAG response generation
def generate_rag_response(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT prompt, response FROM responses WHERE prompt = ?", (user_question,))
    result = cursor.fetchone()
    
    # Implement generate_rag_response logic here
    # ...

# Function to save response to the database
def save_response_to_database(prompt, response):
    cursor = get_db_cursor()
    
    # Create a table if it doesn't exist
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY,
            prompt TEXT,
            response TEXT
        )'''
    )

    # Insert response into the table
    cursor.execute('INSERT INTO responses (prompt, response) VALUES (?, ?)', (prompt, response))
    get_db_connection().commit()

# Threshold for sentiment classification confidence
SENTIMENT_THRESHOLD = 0.6

@app.route("/", methods=['POST', 'GET'])
def query_view():
    if request.method == 'POST':
        prompt = request.form['prompt']

        # Sentiment Analysis using BERT
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).cpu().numpy()
        sentiment_confidence = probabilities[0][predicted_label[0]].item()

        # Check if the prompt contains negative words
        contains_negative = bool(negative_pattern.search(prompt))

        if predicted_label[0] == 1 and sentiment_confidence > SENTIMENT_THRESHOLD and not contains_negative:
            sentiment_intent = "Positive"
        else:
            sentiment_intent = "Negative"

        response = ""
        if sentiment_intent == 'Positive':
            # Sentence-BERT Retrieval
            sbert_response = sbert_retrieval(prompt)
            if sbert_response is not None:
                response = f"{sentiment_intent}, {sbert_response}"
            else:
                # Generate RAG Response
                rag_response = generate_rag_response(prompt)
                if rag_response is not None:
                    # Determine Technical or Non-Technical
                    if any(keyword in rag_response.lower() for keyword in ['error', 'bug', 'technical']):
                        intent = 'Technical'
                    else:
                        intent = 'Non-Technical'
                    response = f"{sentiment_intent}, {intent}: {rag_response}"
                else:
                    # If RAG response is None, send the question to OpenAI API
                    openai_response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0,
                    )['choices'][0]['message']['content']

                    # Save the response to the database
                    save_response_to_database(prompt, openai_response)

                    response = f"{sentiment_intent}, {openai_response}"
        else:
            response = f"{sentiment_intent}, Non-Technical: Sorry, cannot answer the question"

        return jsonify({'response': response})

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
"""

#Version - 5
"""
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import torch

from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import sqlite3
import openai
import threading
import re

app = Flask(__name__)

# Load pretrained GPT-2 model and tokenizer
generation_model = GPT2LMHeadModel.from_pretrained('gpt2')
generation_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load pretrained Sentence-BERT model
retrieval_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load the saved BERT model and tokenizer
saved_model_path = "./content"  # Replace with the actual path
model = BertForSequenceClassification.from_pretrained(saved_model_path)
tokenizer = BertTokenizer.from_pretrained(saved_model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# OpenAI API Key
openai.api_key = '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'

# Define a thread-local storage for database connections and cursors
thread_local = threading.local()

# Define a list of negative words and patterns
negative_words = [
    r'\bnot\b', r'\bwouldn\'t\b', r'\bdon\'t\b', r'\bshouldn\'t\b', r'\bdoesn\'t\b', r'\bwould not\b',
    r'\bdo not\b', r'\bshould not\b', r'\baren\'t\b', r'\bare not\b', r'\bhasn\'t\b', r'\bhas not\b',
    r'\bhaven\'t\b', r'\bhave not\b', r'\bhadn\'t\b', r'\bhad not\b', r'\bnever\b', r'\bno\b',
    r'\bnegative\b', r'\bdisapprove\b', r'\bunhappy\b', r'\bfail\b', r'\bfailure\b', r'\bwrong\b',
    r'\bhate\b', r'\bregret\b', r'\bcan\'t\b', r'\bcannot\b', r'\bwon\'t\b', r'\bwouldn\'t\b',
    r'\bdoesn\'t\b', r'\baren\'t\b', r'\bain\'t\b', r'\bwasn\'t\b', r'\bwere\'n\'t\b', r'\bhasn\'t\b',
    r'\bhaven\'t\b', r'\bhadn\'t\b', r'\bwon\'t\b', r'\bwouldn\'t\b', r'\bshouldn\'t\b', r'\bmistake\b',
    r'\bsad\b', r'\bworst\b', r'\bterrible\b', r'\bhorrible\b', r'\bdreadful\b', r'\bunpleasant\b',
    r'\bdisgusting\b', r'\bdislike\b', r'\bdispleasure\b', r'\bunhappy\b', r'\bunfortunate\b', r'\bneglect\b',
    r'\bunfortunate\b', r'\bunfavorable\b', r'\bawful\b', r'\bnasty\b', r'\bpathetic\b', r'\btragic\b',
    r'\bfault\b', r'\bfiasco\b', r'\bregrettable\b', r'\bsorry\b', r'\bupset\b', r'\bdissatisfied\b',
    r'\bdismal\b', r'\bdepressing\b', r'\bworry\b', r'\bhurt\b', r'\bfrown\b', r'\bcry\b', r'\bcrisis\b',
    r'\bproblem\b', r'\bdanger\b', r'\bdefect\b', r'\bdisaster\b', r'\bfear\b', r'\bpain\b', r'\bshame\b',
    r'\bhopeless\b', r'\bdoubt\b', r'\binsult\b', r'\bnervous\b', r'\bthreat\b', r'\bguilt\b', r'\bweak\b',
    r'\bhorror\b', r'\bshocking\b', r'\bawful\b', r'\brepulsive\b', r'\batrocious\b', r'\bnegative\b',
    r'\bdiscomfort\b', r'\binconvenience\b', r'\bdismay\b', r'\bdespair\b', r'\bdisgust\b', r'\bdespise\b',
    r'\bdiscontent\b', r'\bdispleasure\b', r'\bworry\b', r'\bfear\b', r'\banxiety\b', r'\binsecurity\b',
    r'\bsuspicion\b', r'\bdoubt\b', r'\buncertain\b', r'\bguilt\b', r'\bremorse\b', r'\bshame\b'
]

# Define the database connection functions
def get_db_connection():
    if not hasattr(thread_local, "db_connection"):
        thread_local.db_connection = sqlite3.connect('responses.db')
    return thread_local.db_connection

def get_db_cursor():
    if not hasattr(thread_local, "db_cursor"):
        thread_local.db_cursor = get_db_connection().cursor()
    return thread_local.db_cursor

# Function for Sentence-BERT retrieval
def sbert_retrieval(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT id, prompt, response FROM responses")
    results = cursor.fetchall()
    
    # Implement sbert_retrieval logic here
    # ...

# Function for RAG response generation
def generate_rag_response(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT prompt, response FROM responses WHERE prompt = ?", (user_question,))
    result = cursor.fetchone()
    
    # Implement generate_rag_response logic here
    # ...

# Function to save response to the database
def save_response_to_database(prompt, response):
    cursor = get_db_cursor()
    
    # Create a table if it doesn't exist
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY,
            prompt TEXT,
            response TEXT
        )'''
    )

    # Insert response into the table
    cursor.execute('INSERT INTO responses (prompt, response) VALUES (?, ?)', (prompt, response))
    get_db_connection().commit()

@app.route("/", methods=['POST', 'GET'])
def query_view():
    if request.method == 'POST':
        prompt = request.form['prompt']

        # Sentiment Analysis using BERT
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).cpu().numpy()
        sentiment_intent = "Positive" if predicted_label[0] == 1 else "Negative"

        # Check for negative words in the prompt
        for word in negative_words:
            if re.search(word, prompt, re.IGNORECASE):
                sentiment_intent = "Negative"
                break

        response = ""
        if sentiment_intent == 'Positive':
            # Sentence-BERT Retrieval
            sbert_response = sbert_retrieval(prompt)
            if sbert_response is not None:
                response = f"{sentiment_intent}, {sbert_response}"
            else:
                # Generate RAG Response
                rag_response = generate_rag_response(prompt)
                if rag_response is not None:
                    # Determine Technical or Non-Technical
                    if any(keyword in rag_response.lower() for keyword in ['error', 'bug', 'technical']):
                        intent = 'Technical'
                    else:
                        intent = 'Non-Technical'
                    response = f"{sentiment_intent}, {intent}: {rag_response}"
                else:
                    # If RAG response is None, send the question to OpenAI API
                    openai_response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0,
                    )['choices'][0]['message']['content']

                    # Save the response to the database
                    save_response_to_database(prompt, openai_response)

                    response = f"{sentiment_intent}, {openai_response}"
        else:
            response = f"{sentiment_intent}, Non-Technical: Sorry, cannot answer the question"

        return jsonify({'response': response})

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
"""

#Version - 6
"""
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import torch

from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import sqlite3
import openai
import threading
import re

app = Flask(__name__)

# Load pretrained GPT-2 model and tokenizer
generation_model = GPT2LMHeadModel.from_pretrained('gpt2')
generation_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load pretrained Sentence-BERT model
retrieval_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load the saved BERT model and tokenizer
saved_model_path = "./content"  # Replace with the actual path
model = BertForSequenceClassification.from_pretrained(saved_model_path)
tokenizer = BertTokenizer.from_pretrained(saved_model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# OpenAI API Key
openai.api_key = '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'

# Define a thread-local storage for database connections and cursors
thread_local = threading.local()

# Define a list of negative words and patterns
negative_words = [
    r'\bnot\b', r'\bwouldn\'t\b', r'\bdon\'t\b', r'\bshouldn\'t\b', r'\bdoesn\'t\b', r'\bwould not\b',
    r'\bdo not\b', r'\bshould not\b', r'\baren\'t\b', r'\bare not\b', r'\bhasn\'t\b', r'\bhas not\b',
    r'\bhaven\'t\b', r'\bhave not\b', r'\bhadn\'t\b', r'\bhad not\b', r'\bnever\b', r'\bno\b',
    r'\bnegative\b', r'\bdisapprove\b', r'\bunhappy\b', r'\bfail\b', r'\bwrong\b', r'\bhate\b', r'\bregret\b',
    r'\bcan\'t\b', r'\bcannot\b', r'\bwon\'t\b', r'\bwouldn\'t\b', r'\bdoesn\'t\b', r'\baren\'t\b',
    r'\bain\'t\b', r'\bwasn\'t\b', r'\bwere\'n\'t\b', r'\bhasn\'t\b', r'\bhaven\'t\b', r'\bhadn\'t\b',
    r'\bwon\'t\b', r'\bwouldn\'t\b', r'\bshouldn\'t\b', r'\bmistake\b', r'\bsad\b', r'\bworst\b',
    r'\bterrible\b', r'\bhorrible\b', r'\bdreadful\b', r'\bunpleasant\b', r'\bdisgusting\b', r'\bdislike\b',
    r'\bdispleasure\b', r'\bunhappy\b', r'\bunfortunate\b', r'\bneglect\b', r'\bunfortunate\b', r'\bunfavorable\b',
    r'\bawful\b', r'\bnasty\b', r'\bpathetic\b', r'\btragic\b', r'\bfault\b', r'\bfiasco\b', r'\bregrettable\b',
    r'\bsorry\b', r'\bupset\b', r'\bdissatisfied\b', r'\bdismal\b', r'\bdepressing\b', r'\bworry\b', r'\bhurt\b',
    r'\bfrown\b', r'\bcry\b', r'\bcrisis\b', r'\bproblem\b', r'\bdanger\b', r'\bdefect\b', r'\bdisaster\b',
    r'\bfear\b', r'\bpain\b', r'\bshame\b', r'\bhopeless\b', r'\bdoubt\b', r'\binsult\b', r'\bnervous\b',
    r'\bthreat\b', r'\bguilt\b', r'\bweak\b', r'\bhorror\b', r'\bshocking\b', r'\bawful\b', r'\brepulsive\b',
    r'\batrocious\b', r'\bnegative\b', r'\bdiscomfort\b', r'\binconvenience\b', r'\bdismay\b', r'\bdespair\b',
    r'\bdisgust\b', r'\bdespise\b', r'\bdiscontent\b', r'\bdispleasure\b', r'\bworry\b', r'\bfear\b',
    r'\banxiety\b', r'\binsecurity\b', r'\bsuspicion\b', r'\bdoubt\b', r'\buncertain\b', r'\bguilt\b',
    r'\bremorse\b', r'\bshame\b',
]

# Define the database connection functions
def get_db_connection():
    if not hasattr(thread_local, "db_connection"):
        thread_local.db_connection = sqlite3.connect('responses.db')
    return thread_local.db_connection

def get_db_cursor():
    if not hasattr(thread_local, "db_cursor"):
        thread_local.db_cursor = get_db_connection().cursor()
    return thread_local.db_cursor

# Function for Sentence-BERT retrieval
def sbert_retrieval(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT id, prompt, response FROM responses")
    results = cursor.fetchall()
    
    # Implement sbert_retrieval logic here
    # ...

# Function for RAG response generation
def generate_rag_response(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT prompt, response FROM responses WHERE prompt = ?", (user_question,))
    result = cursor.fetchone()
    
    # Implement generate_rag_response logic here
    # ...

# Function to save response to the database
def save_response_to_database(prompt, response):
    cursor = get_db_cursor()
    
    # Create a table if it doesn't exist
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY,
            prompt TEXT,
            response TEXT
        )'''
    )

    # Insert response into the table
    cursor.execute('INSERT INTO responses (prompt, response) VALUES (?, ?)', (prompt, response))
    get_db_connection().commit()

@app.route("/", methods=['POST', 'GET'])
def query_view():
    if request.method == 'POST':
        prompt = request.form['prompt']

        # Sentiment Analysis using BERT
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).cpu().numpy()

        # Check for negative words in the prompt
        is_negative = any(re.search(word, prompt, re.IGNORECASE) for word in negative_words)

        sentiment_intent = "Negative" if is_negative else "Positive"

        response = f"{sentiment_intent}, Non-Technical: Sorry, cannot answer the question"

        return jsonify({'response': response})

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
"""

#Version - 7
"""
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import torch

from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import sqlite3
import openai
import threading
import re

app = Flask(__name__)

# Load pretrained GPT-2 model and tokenizer
generation_model = GPT2LMHeadModel.from_pretrained('gpt2')
generation_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load pretrained Sentence-BERT model
retrieval_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load the saved BERT model and tokenizer
saved_model_path = "./content"  # Replace with the actual path
model = BertForSequenceClassification.from_pretrained(saved_model_path)
tokenizer = BertTokenizer.from_pretrained(saved_model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# OpenAI API Key
openai.api_key = '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'

# Define a thread-local storage for database connections and cursors
thread_local = threading.local()

# Define a list of negative words and patterns
negative_words = [
    r'\bnot\b', r'\bwouldn\'t\b', r'\bdon\'t\b', r'\bshouldn\'t\b', r'\bdoesn\'t\b', r'\bwould not\b',
    r'\bdo not\b', r'\bshould not\b', r'\baren\'t\b', r'\bare not\b', r'\bhasn\'t\b', r'\bhas not\b',
    r'\bhaven\'t\b', r'\bhave not\b', r'\bhadn\'t\b', r'\bhad not\b', r'\bnever\b', r'\bno\b',
    r'\bnegative\b', r'\bdisapprove\b', r'\bunhappy\b', r'\bfail\b', r'\bwrong\b', r'\bhate\b', r'\bregret\b',
    r'\bcan\'t\b', r'\bcannot\b', r'\bwon\'t\b', r'\bwouldn\'t\b', r'\bdoesn\'t\b', r'\baren\'t\b',
    r'\bain\'t\b', r'\bwasn\'t\b', r'\bwere\'n\'t\b', r'\bhasn\'t\b', r'\bhaven\'t\b', r'\bhadn\'t\b',
    r'\bwon\'t\b', r'\bwouldn\'t\b', r'\bshouldn\'t\b', r'\bmistake\b', r'\bsad\b', r'\bworst\b',
    r'\bterrible\b', r'\bhorrible\b', r'\bdreadful\b', r'\bunpleasant\b', r'\bdisgusting\b', r'\bdislike\b',
    r'\bdispleasure\b', r'\bunhappy\b', r'\bunfortunate\b', r'\bneglect\b', r'\bunfortunate\b', r'\bunfavorable\b',
    r'\bawful\b', r'\bnasty\b', r'\bpathetic\b', r'\btragic\b', r'\bfault\b', r'\bfiasco\b', r'\bregrettable\b',
    r'\bsorry\b', r'\bupset\b', r'\bdissatisfied\b', r'\bdismal\b', r'\bdepressing\b', r'\bworry\b', r'\bhurt\b',
    r'\bfrown\b', r'\bcry\b', r'\bcrisis\b', r'\bproblem\b', r'\bdanger\b', r'\bdefect\b', r'\bdisaster\b',
    r'\bfear\b', r'\bpain\b', r'\bshame\b', r'\bhopeless\b', r'\bdoubt\b', r'\binsult\b', r'\bnervous\b',
    r'\bthreat\b', r'\bguilt\b', r'\bweak\b', r'\bhorror\b', r'\bshocking\b', r'\bawful\b', r'\brepulsive\b',
    r'\batrocious\b', r'\bnegative\b', r'\bdiscomfort\b', r'\binconvenience\b', r'\bdismay\b', r'\bdespair\b',
    r'\bdisgust\b', r'\bdespise\b', r'\bdiscontent\b', r'\bdispleasure\b', r'\bworry\b', r'\bfear\b',
    r'\banxiety\b', r'\binsecurity\b', r'\bsuspicion\b', r'\bdoubt\b', r'\buncertain\b', r'\bguilt\b',
    r'\bremorse\b', r'\bshame\b',
]

# Define the database connection functions
def get_db_connection():
    if not hasattr(thread_local, "db_connection"):
        thread_local.db_connection = sqlite3.connect('responses.db')
    return thread_local.db_connection

def get_db_cursor():
    if not hasattr(thread_local, "db_cursor"):
        thread_local.db_cursor = get_db_connection().cursor()
    return thread_local.db_cursor

# Function for Sentence-BERT retrieval
def sbert_retrieval(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT id, prompt, response FROM responses")
    results = cursor.fetchall()
    
    # Implement sbert_retrieval logic here
    # ...

# Function for RAG response generation
def generate_rag_response(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT prompt, response FROM responses WHERE prompt = ?", (user_question,))
    result = cursor.fetchone()
    
    # Implement generate_rag_response logic here
    # ...

# Function to save response to the database
def save_response_to_database(prompt, response):
    cursor = get_db_cursor()
    
    # Create a table if it doesn't exist
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY,
            prompt TEXT,
            response TEXT
        )'''
    )

    # Insert response into the table
    cursor.execute('INSERT INTO responses (prompt, response) VALUES (?, ?)', (prompt, response))
    get_db_connection().commit()

@app.route("/", methods=['POST', 'GET'])
def query_view():
    if request.method == 'POST':
        prompt = request.form['prompt']

        # Sentiment Analysis using BERT
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).cpu().numpy()

        # Check for negative words in the prompt
        is_negative = any(re.search(word, prompt, re.IGNORECASE) for word in negative_words)

        sentiment_intent = "Negative" if is_negative else "Positive"

        response = ""

        if sentiment_intent == 'Positive':
            # Sentence-BERT Retrieval
            sbert_response = sbert_retrieval(prompt)
            if sbert_response is not None:
                response = f"{sentiment_intent}, {sbert_response}"
            else:
                # Generate RAG Response
                rag_response = generate_rag_response(prompt)
                if rag_response is not None:
                    response = f"{sentiment_intent}, Technical: {rag_response}"
                else:
                    # If RAG response is None, send the question to OpenAI API
                    openai_response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0,
                    )['choices'][0]['message']['content']

                    # Save the response to the database
                    save_response_to_database(prompt, openai_response)

                    response = f"{sentiment_intent}, Technical: {openai_response}"
        else:
            response = f"{sentiment_intent}, Non-Technical: Sorry, cannot answer the question"

        return jsonify({'response': response})

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
"""
#Version - 8
"""
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import torch

from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import sqlite3
import openai
import threading
import re

app = Flask(__name__)

# Load pretrained GPT-2 model and tokenizer
generation_model = GPT2LMHeadModel.from_pretrained('gpt2')
generation_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load pretrained Sentence-BERT model
retrieval_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load the saved BERT model and tokenizer
saved_model_path = "./content"  # Replace with the actual path
model = BertForSequenceClassification.from_pretrained(saved_model_path)
tokenizer = BertTokenizer.from_pretrained(saved_model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# OpenAI API Key
openai.api_key = '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'

# Define a thread-local storage for database connections and cursors
thread_local = threading.local()

# Define a list of negative words and patterns
negative_words = [
    r'\bnot\b', r'\bwouldn\'t\b', r'\bdon\'t\b', r'\bshouldn\'t\b', r'\bdoesn\'t\b', r'\bwould not\b',
    r'\bdo not\b', r'\bshould not\b', r'\baren\'t\b', r'\bare not\b', r'\bhasn\'t\b', r'\bhas not\b',
    r'\bhaven\'t\b', r'\bhave not\b', r'\bhadn\'t\b', r'\bhad not\b', r'\bnever\b', r'\bno\b',
    r'\bnegative\b', r'\bdisapprove\b', r'\bunhappy\b', r'\bfail\b', r'\bwrong\b', r'\bhate\b', r'\bregret\b',
    r'\bcan\'t\b', r'\bcannot\b', r'\bwon\'t\b', r'\bwouldn\'t\b', r'\bdoesn\'t\b', r'\baren\'t\b',
    r'\bain\'t\b', r'\bwasn\'t\b', r'\bwere\'n\'t\b', r'\bhasn\'t\b', r'\bhaven\'t\b', r'\bhadn\'t\b',
    r'\bwon\'t\b', r'\bwouldn\'t\b', r'\bshouldn\'t\b', r'\bmistake\b', r'\bsad\b', r'\bworst\b',
    r'\bterrible\b', r'\bhorrible\b', r'\bdreadful\b', r'\bunpleasant\b', r'\bdisgusting\b', r'\bdislike\b',
    r'\bdispleasure\b', r'\bunhappy\b', r'\bunfortunate\b', r'\bneglect\b', r'\bunfortunate\b', r'\bunfavorable\b',
    r'\bawful\b', r'\bnasty\b', r'\bpathetic\b', r'\btragic\b', r'\bfault\b', r'\bfiasco\b', r'\bregrettable\b',
    r'\bsorry\b', r'\bupset\b', r'\bdissatisfied\b', r'\bdismal\b', r'\bdepressing\b', r'\bworry\b', r'\bhurt\b',
    r'\bfrown\b', r'\bcry\b', r'\bcrisis\b', r'\bproblem\b', r'\bdanger\b', r'\bdefect\b', r'\bdisaster\b',
    r'\bfear\b', r'\bpain\b', r'\bshame\b', r'\bhopeless\b', r'\bdoubt\b', r'\binsult\b', r'\bnervous\b',
    r'\bthreat\b', r'\bguilt\b', r'\bweak\b', r'\bhorror\b', r'\bshocking\b', r'\bawful\b', r'\brepulsive\b',
    r'\batrocious\b', r'\bnegative\b', r'\bdiscomfort\b', r'\binconvenience\b', r'\bdismay\b', r'\bdespair\b',
    r'\bdisgust\b', r'\bdespise\b', r'\bdiscontent\b', r'\bdispleasure\b', r'\bworry\b', r'\bfear\b',
    r'\banxiety\b', r'\binsecurity\b', r'\bsuspicion\b', r'\bdoubt\b', r'\buncertain\b', r'\bguilt\b',
    r'\bremorse\b', r'\bshame\b',
]

# Define the database connection functions
def get_db_connection():
    if not hasattr(thread_local, "db_connection"):
        thread_local.db_connection = sqlite3.connect('responses.db')
    return thread_local.db_connection

def get_db_cursor():
    if not hasattr(thread_local, "db_cursor"):
        thread_local.db_cursor = get_db_connection().cursor()
    return thread_local.db_cursor

# Function for Sentence-BERT retrieval
def sbert_retrieval(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT id, prompt, response FROM responses")
    results = cursor.fetchall()
    
    # Implement sbert_retrieval logic here
    # ...

# Function for RAG response generation
def generate_rag_response(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT prompt, response FROM responses WHERE prompt = ?", (user_question,))
    result = cursor.fetchone()
    
    # Implement generate_rag_response logic here
    # ...

# Function to save response to the database
def save_response_to_database(prompt, response):
    cursor = get_db_cursor()
    
    # Create a table if it doesn't exist
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY,
            prompt TEXT,
            response TEXT
        )'''
    )

    # Insert response into the table
    cursor.execute('INSERT INTO responses (prompt, response) VALUES (?, ?)', (prompt, response))
    get_db_connection().commit()

@app.route("/", methods=['POST', 'GET'])
def query_view():
    if request.method == 'POST':
        prompt = request.form['prompt']

        # Sentiment Analysis using BERT
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).cpu().numpy()

        # Check for negative words in the prompt
        is_negative = any(re.search(word, prompt, re.IGNORECASE) for word in negative_words)

        sentiment_intent = "Negative" if is_negative else "Positive"
        
        response = ""
        if sentiment_intent == 'Positive':
            # Sentence-BERT Retrieval
            sbert_response = sbert_retrieval(prompt)
            if sbert_response is not None:
                response = f"{sentiment_intent}, {sbert_response}"
            else:
                # Generate RAG Response
                rag_response = generate_rag_response(prompt)
                if rag_response is not None:
                    # Determine if it is Technical or Non-Technical
                    if any(keyword in rag_response.lower() for keyword in ['error', 'bug', 'technical']):
                        intent = 'Technical'
                    else:
                        intent = 'Non-Technical'
                    response = f"{sentiment_intent}, {intent}: {rag_response}"
                else:
                    # If RAG response is None, send the question to OpenAI API
                    openai_response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0,
                    )['choices'][0]['message']['content']

                    # Save the response to the database
                    save_response_to_database(prompt, openai_response)

                    response = f"{sentiment_intent}, {openai_response}"
        else:
            response = f"{sentiment_intent}, Non-Technical: Sorry, cannot answer the question"

        return jsonify({'response': response})

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
"""

#Version - 9
"""
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import torch

from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import sqlite3
import openai
import threading
import re

app = Flask(__name__)

# Load pretrained GPT-2 model and tokenizer
generation_model = GPT2LMHeadModel.from_pretrained('gpt2')
generation_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load pretrained Sentence-BERT model
retrieval_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load the saved BERT model and tokenizer
saved_model_path = "./content"  # Replace with the actual path
model = BertForSequenceClassification.from_pretrained(saved_model_path)
tokenizer = BertTokenizer.from_pretrained(saved_model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# OpenAI API Key
openai.api_key = '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'

# Define a thread-local storage for database connections and cursors
thread_local = threading.local()

# Define a list of negative words and patterns
negative_words = [
    r'\bnot\b', r'\bwouldn\'t\b', r'\bdon\'t\b', r'\bshouldn\'t\b', r'\bdoesn\'t\b', r'\bwould not\b',
    r'\bdo not\b', r'\bshould not\b', r'\baren\'t\b', r'\bare not\b', r'\bhasn\'t\b', r'\bhas not\b',
    r'\bhaven\'t\b', r'\bhave not\b', r'\bhadn\'t\b', r'\bhad not\b', r'\bnever\b', r'\bno\b',
    r'\bnegative\b', r'\bdisapprove\b', r'\bunhappy\b', r'\bfail\b', r'\bwrong\b', r'\bhate\b', r'\bregret\b',
    r'\bcan\'t\b', r'\bcannot\b', r'\bwon\'t\b', r'\bwouldn\'t\b', r'\bdoesn\'t\b', r'\baren\'t\b',
    r'\bain\'t\b', r'\bwasn\'t\b', r'\bwere\'n\'t\b', r'\bhasn\'t\b', r'\bhaven\'t\b', r'\bhadn\'t\b',
    r'\bwon\'t\b', r'\bwouldn\'t\b', r'\bshouldn\'t\b', r'\bmistake\b', r'\bsad\b', r'\bworst\b',
    r'\bterrible\b', r'\bhorrible\b', r'\bdreadful\b', r'\bunpleasant\b', r'\bdisgusting\b', r'\bdislike\b',
    r'\bdispleasure\b', r'\bunhappy\b', r'\bunfortunate\b', r'\bneglect\b', r'\bunfortunate\b', r'\bunfavorable\b',
    r'\bawful\b', r'\bnasty\b', r'\bpathetic\b', r'\btragic\b', r'\bfault\b', r'\bfiasco\b', r'\bregrettable\b',
    r'\bsorry\b', r'\bupset\b', r'\bdissatisfied\b', r'\bdismal\b', r'\bdepressing\b', r'\bworry\b', r'\bhurt\b',
    r'\bfrown\b', r'\bcry\b', r'\bcrisis\b', r'\bproblem\b', r'\bdanger\b', r'\bdefect\b', r'\bdisaster\b',
    r'\bfear\b', r'\bpain\b', r'\bshame\b', r'\bhopeless\b', r'\bdoubt\b', r'\binsult\b', r'\bnervous\b',
    r'\bthreat\b', r'\bguilt\b', r'\bweak\b', r'\bhorror\b', r'\bshocking\b', r'\bawful\b', r'\brepulsive\b',
    r'\batrocious\b', r'\bnegative\b', r'\bdiscomfort\b', r'\binconvenience\b', r'\bdismay\b', r'\bdespair\b',
    r'\bdisgust\b', r'\bdespise\b', r'\bdiscontent\b', r'\bdispleasure\b', r'\bworry\b', r'\bfear\b',
    r'\banxiety\b', r'\binsecurity\b', r'\bsuspicion\b', r'\bdoubt\b', r'\buncertain\b', r'\bguilt\b',
    r'\bremorse\b', r'\bshame\b',
]

# Define the database connection functions
def get_db_connection():
    if not hasattr(thread_local, "db_connection"):
        thread_local.db_connection = sqlite3.connect('responses.db')
    return thread_local.db_connection

def get_db_cursor():
    if not hasattr(thread_local, "db_cursor"):
        thread_local.db_cursor = get_db_connection().cursor()
    return thread_local.db_cursor

# Function for Sentence-BERT retrieval
def sbert_retrieval(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT id, prompt, response FROM responses")
    results = cursor.fetchall()
    
    # Implement sbert_retrieval logic here
    # ...

# Function for RAG response generation
def generate_rag_response(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT prompt, response FROM responses WHERE prompt = ?", (user_question,))
    result = cursor.fetchone()
    
    # Implement generate_rag_response logic here
    # ...

# Function to save response to the database
def save_response_to_database(prompt, response):
    cursor = get_db_cursor()
    
    # Create a table if it doesn't exist
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY,
            prompt TEXT,
            response TEXT
        )'''
    )

    # Insert response into the table
    cursor.execute('INSERT INTO responses (prompt, response) VALUES (?, ?)', (prompt, response))
    get_db_connection().commit()

@app.route("/", methods=['POST', 'GET'])
def query_view():
    if request.method == 'POST':
        prompt = request.form['prompt']

        # Sentiment Analysis using BERT
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).cpu().numpy()

        # Check for negative words in the prompt
        is_negative = any(re.search(word, prompt, re.IGNORECASE) for word in negative_words)

        sentiment_intent = "Negative" if is_negative else "Positive"
        response = ""
        intent = None  # Initialize intent as None

        # Perform retrieval or generation based on sentiment and intent
        if sentiment_intent == 'Positive':
            # Sentence-BERT Retrieval
            sbert_response = sbert_retrieval(prompt)
            if sbert_response is not None:
                # Check intent (Technical or Non-Technical)
                if any(keyword in sbert_response.lower() for keyword in ['error', 'bug', 'technical']):
                    intent = 'Technical'
                else:
                    intent = 'Non-Technical'
                
                # Generate response based on intent
                if intent == 'Technical':
                    response = f"{sentiment_intent}, {intent}: {sbert_response}"
                else:
                    response = "Sorry, cannot answer the question"
            else:
                # RAG Response Generation (intent does not apply here)
                rag_response = generate_rag_response(prompt)
                if rag_response is not None:
                    response = f"{sentiment_intent}, {rag_response}"
                else:
                    # If RAG response is None, send the question to OpenAI API
                    openai_response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0,
                    )['choices'][0]['message']['content']

                    # Save the response to the database
                    save_response_to_database(prompt, openai_response)

                    response = f"{sentiment_intent}, {openai_response}"
        else:  # Negative sentiment
            # Sentence-BERT Retrieval
            sbert_response = sbert_retrieval(prompt)
            if sbert_response is not None:
                # Check intent (Technical or Non-Technical)
                if any(keyword in sbert_response.lower() for keyword in ['error', 'bug', 'technical']):
                    intent = 'Technical'
                else:
                    intent = 'Non-Technical'
                
                # Generate response based on intent
                if intent == 'Technical':
                    response = f"{sentiment_intent}, {intent}: {sbert_response}"
                else:
                    response = "Sorry, cannot answer the question"
            else:
                # RAG Response Generation (intent does not apply here)
                rag_response = generate_rag_response(prompt)
                if rag_response is not None:
                    response = f"{sentiment_intent}, {rag_response}"
                else:
                    response = "Sorry, cannot answer the question"

        return jsonify({'response': response})

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
"""

#Version - 10
"""
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import torch
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import sqlite3
import openai
import threading
import re

app = Flask(__name__)

# Load pretrained GPT-2 model and tokenizer
generation_model = GPT2LMHeadModel.from_pretrained('gpt2')
generation_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load pretrained Sentence-BERT model
retrieval_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load the saved BERT model and tokenizer
saved_model_path = "./content"  # Replace with the actual path
model = BertForSequenceClassification.from_pretrained(saved_model_path)
tokenizer = BertTokenizer.from_pretrained(saved_model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# OpenAI API Key
openai.api_key = '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'

# Define a thread-local storage for database connections and cursors
thread_local = threading.local()

# Define a list of negative words and patterns
negative_words = [
    r'\bnot\b', r'\bwouldn\'t\b', r'\bdon\'t\b', r'\bshouldn\'t\b', r'\bdoesn\'t\b', r'\bwould not\b',
    r'\bdo not\b', r'\bshould not\b', r'\baren\'t\b', r'\bare not\b', r'\bhasn\'t\b', r'\bhas not\b',
    r'\bhaven\'t\b', r'\bhave not\b', r'\bhadn\'t\b', r'\bhad not\b', r'\bnever\b', r'\bno\b',
    r'\bnegative\b', r'\bdisapprove\b', r'\bunhappy\b', r'\bfail\b', r'\bwrong\b', r'\bhate\b', r'\bregret\b',
    r'\bcan\'t\b', r'\bcannot\b', r'\bwon\'t\b', r'\bwouldn\'t\b', r'\bdoesn\'t\b', r'\baren\'t\b',
    r'\bain\'t\b', r'\bwasn\'t\b', r'\bwere\'n\'t\b', r'\bhasn\'t\b', r'\bhaven\'t\b', r'\bhadn\'t\b',
    r'\bwon\'t\b', r'\bwouldn\'t\b', r'\bshouldn\'t\b', r'\bmistake\b', r'\bsad\b', r'\bworst\b',
    r'\bterrible\b', r'\bhorrible\b', r'\bdreadful\b', r'\bunpleasant\b', r'\bdisgusting\b', r'\bdislike\b',
    r'\bdispleasure\b', r'\bunhappy\b', r'\bunfortunate\b', r'\bneglect\b', r'\bunfortunate\b', r'\bunfavorable\b',
    r'\bawful\b', r'\bnasty\b', r'\bpathetic\b', r'\btragic\b', r'\bfault\b', r'\bfiasco\b', r'\bregrettable\b',
    r'\bsorry\b', r'\bupset\b', r'\bdissatisfied\b', r'\bdismal\b', r'\bdepressing\b', r'\bworry\b', r'\bhurt\b',
    r'\bfrown\b', r'\bcry\b', r'\bcrisis\b', r'\bproblem\b', r'\bdanger\b', r'\bdefect\b', r'\bdisaster\b',
    r'\bfear\b', r'\bpain\b', r'\bshame\b', r'\bhopeless\b', r'\bdoubt\b', r'\binsult\b', r'\bnervous\b',
    r'\bthreat\b', r'\bguilt\b', r'\bweak\b', r'\bhorror\b', r'\bshocking\b', r'\bawful\b', r'\brepulsive\b',
    r'\batrocious\b', r'\bnegative\b', r'\bdiscomfort\b', r'\binconvenience\b', r'\bdismay\b', r'\bdespair\b',
    r'\bdisgust\b', r'\bdespise\b', r'\bdiscontent\b', r'\bdispleasure\b', r'\bworry\b', r'\bfear\b',
    r'\banxiety\b', r'\binsecurity\b', r'\bsuspicion\b', r'\bdoubt\b', r'\buncertain\b', r'\bguilt\b',
    r'\bremorse\b', r'\bshame\b',
]

# Define the database connection functions
def get_db_connection():
    if not hasattr(thread_local, "db_connection"):
        thread_local.db_connection = sqlite3.connect('responses.db')
    return thread_local.db_connection

def get_db_cursor():
    if not hasattr(thread_local, "db_cursor"):
        thread_local.db_cursor = get_db_connection().cursor()
    return thread_local.db_cursor

# Function for Sentence-BERT retrieval
def sbert_retrieval(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT id, prompt, response FROM responses")
    results = cursor.fetchall()
    
    # Implement sbert_retrieval logic here
    # ...

# Function for RAG response generation
def generate_rag_response(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT prompt, response FROM responses WHERE prompt = ?", (user_question,))
    result = cursor.fetchone()
    
    # Implement generate_rag_response logic here
    # ...

# Function to save response to the database
def save_response_to_database(prompt, response):
    cursor = get_db_cursor()
    
    # Create a table if it doesn't exist
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY,
            prompt TEXT,
            response TEXT
        )'''
    )

    # Insert response into the table
    cursor.execute('INSERT INTO responses (prompt, response) VALUES (?, ?)', (prompt, response))
    get_db_connection().commit()

# Function to retrieve response from the database
def retrieve_response_from_database(prompt):
    cursor = get_db_cursor()
    cursor.execute("SELECT response FROM responses WHERE prompt = ?", (prompt,))
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        return None

@app.route("/", methods=['POST', 'GET'])
def query_view():
    if request.method == 'POST':
        prompt = request.form['prompt']

        # Sentiment Analysis using BERT
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).cpu().numpy()

        # Check for negative words in the prompt
        is_negative = any(re.search(word, prompt, re.IGNORECASE) for word in negative_words)

        sentiment_intent = "Negative" if is_negative else "Positive"
        response = ""
        intent = None  # Initialize intent as None

        # Implement the conditions you specified
        if sentiment_intent == 'Positive':
            # Check if intent is technical
            if any(keyword in prompt.lower() for keyword in ['error', 'bug', 'technical']):
                intent = 'Technical'
            
            if intent == 'Technical':
                # Generate a response for positive sentiment and technical intent from the database
                db_response = retrieve_response_from_database(prompt)
                if db_response:
                    response = db_response
                else:
                    response = "Sorry, cannot answer the question"
            else:
                response = "Sorry, cannot answer the question"
        else:  # Negative sentiment
            # Check if intent is technical
            if any(keyword in prompt.lower() for keyword in ['error', 'bug', 'technical']):
                intent = 'Technical'
            
            if intent == 'Technical':
                # Generate a response for negative sentiment and technical intent from the database
                db_response = retrieve_response_from_database(prompt)
                if db_response:
                    response = db_response
                else:
                    response = "Sorry, cannot answer the question"
            else:
                response = "Sorry, cannot answer the question"

        return jsonify({'response': response})

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
"""
#Version 11
"""
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import torch
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import sqlite3
import openai
import threading
import re

app = Flask(__name__)

# Load pretrained GPT-2 model and tokenizer
generation_model = GPT2LMHeadModel.from_pretrained('gpt2')
generation_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load pretrained Sentence-BERT model
retrieval_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load the saved BERT model and tokenizer
saved_model_path = "./content"  # Replace with the actual path
model = BertForSequenceClassification.from_pretrained(saved_model_path)
tokenizer = BertTokenizer.from_pretrained(saved_model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# OpenAI API Key
openai.api_key = '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'

# Define a thread-local storage for database connections and cursors
thread_local = threading.local()

# Define a list of negative words and patterns
negative_words = [
    r'\bnot\b', r'\bwouldn\'t\b', r'\bdon\'t\b', r'\bshouldn\'t\b', r'\bdoesn\'t\b', r'\bwould not\b',
    r'\bdo not\b', r'\bshould not\b', r'\baren\'t\b', r'\bare not\b', r'\bhasn\'t\b', r'\bhas not\b',
    r'\bhaven\'t\b', r'\bhave not\b', r'\bhadn\'t\b', r'\bhad not\b', r'\bnever\b', r'\bno\b',
    # (list of negative words and patterns continues...)
]

# Define the database connection functions
def get_db_connection():
    if not hasattr(thread_local, "db_connection"):
        thread_local.db_connection = sqlite3.connect('responses.db')
    return thread_local.db_connection

def get_db_cursor():
    if not hasattr(thread_local, "db_cursor"):
        thread_local.db_cursor = get_db_connection().cursor()
    return thread_local.db_cursor

# Function for Sentence-BERT retrieval
def sbert_retrieval(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT id, prompt, response FROM responses")
    results = cursor.fetchall()
    
    # Implement sbert_retrieval logic here
    # ...

# Function for RAG response generation
def generate_rag_response(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT prompt, response FROM responses WHERE prompt = ?", (user_question,))
    result = cursor.fetchone()
    
    # Implement generate_rag_response logic here
    # ...

# Function to save response to the database
def save_response_to_database(prompt, response):
    cursor = get_db_cursor()
    
    # Create a table if it doesn't exist
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY,
            prompt TEXT,
            response TEXT
        )'''
    )

    # Insert response into the table
    cursor.execute('INSERT INTO responses (prompt, response) VALUES (?, ?)', (prompt, response))
    get_db_connection().commit()

# Function to retrieve response from the database
def retrieve_response_from_database(prompt):
    cursor = get_db_cursor()
    cursor.execute("SELECT response FROM responses WHERE prompt = ?", (prompt,))
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        return None

@app.route("/", methods=['POST', 'GET'])
def query_view():
    if request.method == 'POST':
        prompt = request.form['prompt']

        # Sentiment Analysis using BERT
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).cpu().numpy()

        # Check for negative words in the prompt
        is_negative = any(re.search(word, prompt, re.IGNORECASE) for word in negative_words)

        sentiment_intent = "Negative" if is_negative else "Positive"
        response = ""
        intent = None  # Initialize intent as None

        # Check if intent is technical
        if any(keyword in prompt.lower() for keyword in ['error', 'bug', 'technical']):
            intent = 'Technical'

        # Implement the conditions you specified
        if intent == 'Technical':
            # Try to retrieve a response from the database
            db_response = retrieve_response_from_database(prompt)

            if db_response:
                # Response found in the database
                response = db_response
            else:
                # Response not in the database, get it from OpenAI
                openai_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )['choices'][0]['message']['content']

                # Save the response to the database
                save_response_to_database(prompt, openai_response)

                response = openai_response
        else:
            response = "Sorry, cannot answer the question"

        return jsonify({'response': response})

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
"""
#Version 12
"""
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset, DataLoader
from flask import Flask, render_template, request, jsonify
import sqlite3
import openai
import re
import pandas as pd
import threading

app = Flask(__name__)

# Load the saved BERT model and tokenizer
saved_model_path = "./content"  # Replace with the actual path
model = BertForSequenceClassification.from_pretrained(saved_model_path)
tokenizer = BertTokenizer.from_pretrained(saved_model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# OpenAI API Key
openai.api_key = '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'

# Define a thread-local storage for database connections and cursors
thread_local = threading.local()

# Database connection functions
def get_db_connection():
    if not hasattr(thread_local, "db_connection"):
        thread_local.db_connection = sqlite3.connect('responses.db')
    return thread_local.db_connection

def get_db_cursor():
    if not hasattr(thread_local, "db_cursor"):
        thread_local.db_cursor = get_db_connection().cursor()
    return thread_local.db_cursor

# Function for RAG response generation
def generate_rag_response(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT prompt, response FROM responses WHERE prompt = ?", (user_question,))
    result = cursor.fetchone()

    # Implement generate_rag_response logic here
    # ...

# Function to save response to the database
def save_response_to_database(prompt, response):
    cursor = get_db_cursor()

    # Create a table if it doesn't exist
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY,
            prompt TEXT,
            response TEXT
        )'''
    )

    # Insert response into the table
    cursor.execute('INSERT INTO responses (prompt, response) VALUES (?, ?)', (prompt, response))
    get_db_connection().commit()

# List of negative words
negative_words = [
    r'\bnot\b', r'\bwouldn\'t\b', r'\bdon\'t\b', r'\bshouldn\'t\b', r'\bdoesn\'t\b', r'\bwould not\b',
    r'\bdo not\b', r'\bshould not\b', r'\baren\'t\b', r'\bare not\b', r'\bhasn\'t\b', r'\bhas not\b',
    r'\bhaven\'t\b', r'\bhave not\b', r'\bhadn\'t\b', r'\bhad not\b', r'\bnever\b', r'\bno\b',
    r'\bnegative\b', r'\bdisapprove\b', r'\bunhappy\b', r'\bfail\b', r'\bwrong\b', r'\bhate\b', r'\bregret\b',
    r'\bcan\'t\b', r'\bcannot\b', r'\bwon\'t\b', r'\bwouldn\'t\b', r'\bdoesn\'t\b', r'\baren\'t\b',
    r'\bain\'t\b', r'\bwasn\'t\b', r'\bwere\'n\'t\b', r'\bhasn\'t\b', r'\bhaven\'t\b', r'\bhadn\'t\b',
    r'\bwon\'t\b', r'\bwouldn\'t\b', r'\bshouldn\'t\b', r'\bmistake\b', r'\bsad\b', r'\bworst\b',
    r'\bterrible\b', r'\bhorrible\b', r'\bdreadful\b', r'\bunpleasant\b', r'\bdisgusting\b', r'\bdislike\b',
    r'\bdispleasure\b', r'\bunhappy\b', r'\bunfortunate\b', r'\bneglect\b', r'\bunfortunate\b', r'\bunfavorable\b',
    r'\bawful\b', r'\bnasty\b', r'\bpathetic\b', r'\btragic\b', r'\bfault\b', r'\bfiasco\b', r'\bregrettable\b',
    r'\bsorry\b', r'\bupset\b', r'\bdissatisfied\b', r'\bdismal\b', r'\bdepressing\b', r'\bworry\b', r'\bhurt\b',
    r'\bfrown\b', r'\bcry\b', r'\bcrisis\b', r'\bproblem\b', r'\bdanger\b', r'\bdefect\b', r'\bdisaster\b',
    r'\bfear\b', r'\bpain\b', r'\bshame\b', r'\bhopeless\b', r'\bdoubt\b', r'\binsult\b', r'\bnervous\b',
    r'\bthreat\b', r'\bguilt\b', r'\bweak\b', r'\bhorror\b', r'\bshocking\b', r'\bawful\b', r'\brepulsive\b',
    r'\batrocious\b', r'\bnegative\b', r'\bdiscomfort\b', r'\binconvenience\b', r'\bdismay\b', r'\bdespair\b',
    r'\bdisgust\b', r'\bdespise\b', r'\bdiscontent\b', r'\bdispleasure\b', r'\bworry\b', r'\bfear\b',
    r'\banxiety\b', r'\binsecurity\b', r'\bsuspicion\b', r'\bdoubt\b', r'\buncertain\b', r'\bguilt\b',
    r'\bremorse\b', r'\bshame\b',
]

# Create a regular expression pattern for negative words
negative_pattern = re.compile('|'.join(negative_words), re.IGNORECASE)

# Sample data for intent classification
data = [
    ("Technical", "How do I fix a 404 error on my website?"),
    ("Technical", "My code is giving a segmentation fault. What should I do?"),
    ("Non-Technical", "What are the best books to read this summer?"),
    ("Non-Technical", "Tell me about the latest fashion trends."),
    # Add more data examples
]

# Separate data into intent labels and text
intents, texts = zip(*data)

# Create a DataFrame from the data
intent_df = pd.DataFrame({'text': texts, 'intent': intents})

# Load a pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(intent_df['intent'].unique()))

# Tokenize the input text and encode labels
encoding = tokenizer(list(intent_df['text']), truncation=True, padding=True)
intent_labels = intent_df['intent'].astype('category').cat.codes.values.astype(np.int64)  # Modify label encoding

# Define batch size
batch_size = 4

# Create a Dataset
class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # Convert labels to long data type
        return item

    def __len__(self):
        return len(self.labels)

intent_dataset = IntentDataset(encoding, intent_labels)

# Split dataset into training and evaluation sets
from sklearn.model_selection import train_test_split

train_texts, eval_texts, train_labels, eval_labels = train_test_split(intent_df['text'], intent_labels, test_size=0.2, random_state=42)

# Tokenize and encode the training and evaluation datasets
train_encoding = tokenizer(list(train_texts), truncation=True, padding=True)
eval_encoding = tokenizer(list(eval_texts), truncation=True, padding=True)

# Create training and evaluation datasets
train_dataset = IntentDataset(train_encoding, train_labels)
eval_dataset = IntentDataset(eval_encoding, eval_labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./intent_model',
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    evaluation_strategy='epoch',
    logging_dir='./logs',
)

# Create a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Use the training dataset
    eval_dataset=eval_dataset,    # Set the evaluation dataset
)

# Fine-tune the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()

# Define a function to classify intents using the fine-tuned BERT model
def classify_intent(user_input):
    encoding = tokenizer(user_input, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**encoding).logits
    intent_id = torch.argmax(logits, dim=1).item()
    intent_labels = list(intent_df['intent'].unique())
    return intent_labels[intent_id]

@app.route("/", methods=['POST', 'GET'])
def query_view():
    if request.method == 'POST':
        prompt = request.form['prompt']

        # Sentiment Analysis using regular expressions
        sentiment_intent = "Negative" if negative_pattern.search(prompt) else "Positive"

        # Intent Classification using BERT
        intent = classify_intent(prompt)

        response = ""
        if sentiment_intent == 'Positive':
            # Generate RAG Response
            rag_response = generate_rag_response(prompt)
            if rag_response is not None:
                # Determine Technical or Non-Technical
                if any(keyword in rag_response.lower() for keyword in ['error', 'bug', 'technical']):
                    intent = 'Technical'
                else:
                    intent = 'Non-Technical'
                response = f"{sentiment_intent}, {intent}: {rag_response}"
            else:
                # If RAG response is None, send the question to OpenAI API
                openai_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )['choices'][0]['message']['content']

                # Save the response to the database
                save_response_to_database(prompt, openai_response)

                response = f"{sentiment_intent}, {openai_response}"

        else:
            response = f"{sentiment_intent}, Non-Technical Sorry, Cannot answer the Question"

        return jsonify({'response': response})

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
"""

#Version - 13
"""
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset, DataLoader
from flask import Flask, render_template, request, jsonify
import sqlite3
import openai
import re
import pandas as pd
import threading

app = Flask(__name__)

# Load the saved BERT model and tokenizer
saved_model_path = "./content"  # Replace with the actual path
model = BertForSequenceClassification.from_pretrained(saved_model_path)
tokenizer = BertTokenizer.from_pretrained(saved_model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# OpenAI API Key (Replace with your actual API key)
openai.api_key = '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'

# Define a thread-local storage for database connections and cursors
thread_local = threading.local()

# Database connection functions
def get_db_connection():
    if not hasattr(thread_local, "db_connection"):
        thread_local.db_connection = sqlite3.connect('responses.db')
    return thread_local.db_connection

def get_db_cursor():
    if not hasattr(thread_local, "db_cursor"):
        thread_local.db_cursor = get_db_connection().cursor()
    return thread_local.db_cursor

# Function for RAG response generation
def generate_rag_response(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT prompt, response FROM responses WHERE prompt = ?", (user_question,))
    result = cursor.fetchone()

    if result:
        return result[1]
    else:
        return None

# Function to save response to the database
def save_response_to_database(prompt, response):
    cursor = get_db_cursor()

    # Create a table if it doesn't exist
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY,
            prompt TEXT,
            response TEXT
        )'''
    )

    # Insert response into the table
    cursor.execute('INSERT INTO responses (prompt, response) VALUES (?, ?)', (prompt, response))
    get_db_connection().commit()

# List of negative words
negative_words = [
    r'\bnot\b', r'\bwouldn\'t\b', r'\bdon\'t\b', r'\bshouldn\'t\b', r'\bdoesn\'t\b', r'\bwould not\b',
    r'\bdo not\b', r'\bshould not\b', r'\baren\'t\b', r'\bare not\b', r'\bhasn\'t\b', r'\bhas not\b',
    r'\bhaven\'t\b', r'\bhave not\b', r'\bhadn\'t\b', r'\bhad not\b', r'\bnever\b', r'\bno\b',
    r'\bnegative\b', r'\bdisapprove\b', r'\bunhappy\b', r'\bfail\b', r'\bwrong\b', r'\bhate\b', r'\bregret\b',
    r'\bcan\'t\b', r'\bcannot\b', r'\bwon\'t\b', r'\bwouldn\'t\b', r'\bdoesn\'t\b', r'\baren\'t\b',
    r'\bain\'t\b', r'\bwasn\'t\b', r'\bwere\'n\'t\b', r'\bhasn\'t\b', r'\bhaven\'t\b', r'\bhadn\'t\b',
    r'\bwon\'t\b', r'\bwouldn\'t\b', r'\bshouldn\'t\b', r'\bmistake\b', r'\bsad\b', r'\bworst\b',
    r'\bterrible\b', r'\bhorrible\b', r'\bdreadful\b', r'\bunpleasant\b', r'\bdisgusting\b', r'\bdislike\b',
    r'\bdispleasure\b', r'\bunhappy\b', r'\bunfortunate\b', r'\bneglect\b', r'\bunfortunate\b', r'\bunfavorable\b',
    r'\bawful\b', r'\bnasty\b', r'\bpathetic\b', r'\btragic\b', r'\bfault\b', r'\bfiasco\b', r'\bregrettable\b',
    r'\bsorry\b', r'\bupset\b', r'\bdissatisfied\b', r'\bdismal\b', r'\bdepressing\b', r'\bworry\b', r'\bhurt\b',
    r'\bfrown\b', r'\bcry\b', r'\bcrisis\b', r'\bproblem\b', r'\bdanger\b', r'\bdefect\b', r'\bdisaster\b',
    r'\bfear\b', r'\bpain\b', r'\bshame\b', r'\bhopeless\b', r'\bdoubt\b', r'\binsult\b', r'\bnervous\b',
    r'\bthreat\b', r'\bguilt\b', r'\bweak\b', r'\bhorror\b', r'\bshocking\b', r'\bawful\b', r'\brepulsive\b',
    r'\batrocious\b', r'\bnegative\b', r'\bdiscomfort\b', r'\binconvenience\b', r'\bdismay\b', r'\bdespair\b',
    r'\bdisgust\b', r'\bdespise\b', r'\bdiscontent\b', r'\bdispleasure\b', r'\bworry\b', r'\bfear\b',
    r'\banxiety\b', r'\binsecurity\b', r'\bsuspicion\b', r'\bdoubt\b', r'\buncertain\b', r'\bguilt\b',
    r'\bremorse\b', r'\bshame\b',
]

# Create a regular expression pattern for negative words
negative_pattern = re.compile('|'.join(negative_words), re.IGNORECASE)

# Sample data for intent classification
data = [
    ("Technical", "How do I fix a 404 error on my website?"),
    ("Technical", "My code is giving a segmentation fault. What should I do?"),
    ("Non-Technical", "What are the best books to read this summer?"),
    ("Non-Technical", "Tell me about the latest fashion trends."),
    # Add more data examples
]

# Separate data into intent labels and text
intents, texts = zip(*data)

# Create a DataFrame from the data
intent_df = pd.DataFrame({'text': texts, 'intent': intents})

# Load a pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(intent_df['intent'].unique()))

# Tokenize the input text and encode labels
encoding = tokenizer(list(intent_df['text']), truncation=True, padding=True)
intent_labels = intent_df['intent'].astype('category').cat.codes.values.astype(np.int64)  # Modify label encoding

# Define batch size
batch_size = 4

# Create a Dataset
class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # Convert labels to long data type
        return item

    def __len__(self):
        return len(self.labels)

intent_dataset = IntentDataset(encoding, intent_labels)

# Split dataset into training and evaluation sets
from sklearn.model_selection import train_test_split

train_texts, eval_texts, train_labels, eval_labels = train_test_split(intent_df['text'], intent_labels, test_size=0.2, random_state=42)

# Tokenize and encode the training and evaluation datasets
train_encoding = tokenizer(list(train_texts), truncation=True, padding=True)
eval_encoding = tokenizer(list(eval_texts), truncation=True, padding=True)

# Create training and evaluation datasets
train_dataset = IntentDataset(train_encoding, train_labels)
eval_dataset = IntentDataset(eval_encoding, eval_labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./intent_model',
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    evaluation_strategy='epoch',
    logging_dir='./logs',
)

# Create a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Use the training dataset
    eval_dataset=eval_dataset,    # Set the evaluation dataset
)

# Fine-tune the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()

# Define a function to classify intents using the fine-tuned BERT model
def classify_intent(user_input):
    encoding = tokenizer(user_input, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**encoding).logits
    intent_id = torch.argmax(logits, dim=1).item()
    intent_labels = list(intent_df['intent'].unique())
    return intent_labels[intent_id]

@app.route("/", methods=['POST', 'GET'])
def query_view():
    if request.method == 'POST':
        prompt = request.form['prompt']

        # Sentiment Analysis using regular expressions
        sentiment_intent = "Negative" if negative_pattern.search(prompt) else "Positive"

        # Intent Classification using BERT
        intent = classify_intent(prompt)

        response = ""
        if sentiment_intent == 'Positive':
            # Generate RAG Response
            rag_response = generate_rag_response(prompt)
            if rag_response is not None:
                # Determine Technical or Non-Technical
                if any(keyword in rag_response.lower() for keyword in ['error', 'bug', 'technical']):
                    intent = 'Technical'
                    response = f"{sentiment_intent}, {intent}: {rag_response}"
                else:
                    intent = 'Non-Technical'
                response = f"{sentiment_intent}, {intent}: {rag_response}"
            else:
                # If RAG response is None, send the question to OpenAI API
                openai_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )['choices'][0]['message']['content']

                # Save the response to the database
                save_response_to_database(prompt, openai_response)

                response = f"{sentiment_intent}, {openai_response}"

        else:
            response = f"{sentiment_intent}, Non-Technical: Sorry, cannot answer the question"

        return jsonify({'response': response})

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
"""

#Version - 14
"""
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset, DataLoader
from flask import Flask, render_template, request, jsonify
import sqlite3
import openai
import re
import pandas as pd
import threading

app = Flask(__name__)

# Load the saved BERT model and tokenizer
saved_model_path = "./content"  # Replace with the actual path
model = BertForSequenceClassification.from_pretrained(saved_model_path)
tokenizer = BertTokenizer.from_pretrained(saved_model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# OpenAI API Key (Replace with your actual API key)
openai.api_key = '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'

# Define a thread-local storage for database connections and cursors
thread_local = threading.local()

# Database connection functions
def get_db_connection():
    if not hasattr(thread_local, "db_connection"):
        thread_local.db_connection = sqlite3.connect('responses.db')
    return thread_local.db_connection

def get_db_cursor():
    if not hasattr(thread_local, "db_cursor"):
        thread_local.db_cursor = get_db_connection().cursor()
    return thread_local.db_cursor

# Function for RAG response generation
def generate_rag_response(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT prompt, response FROM responses WHERE prompt = ?", (user_question,))
    result = cursor.fetchone()

    if result:
        return result[1]
    else:
        return None

# Function to save response to the database
def save_response_to_database(prompt, response):
    cursor = get_db_cursor()

    # Create a table if it doesn't exist
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY,
            prompt TEXT,
            response TEXT
        )'''
    )

    # Insert response into the table
    cursor.execute('INSERT INTO responses (prompt, response) VALUES (?, ?)', (prompt, response))
    get_db_connection().commit()

# List of negative words
negative_words = [
    r'\bnot\b', r'\bwouldn\'t\b', r'\bdon\'t\b', r'\bshouldn\'t\b', r'\bdoesn\'t\b', r'\bwould not\b',
    r'\bdo not\b', r'\bshould not\b', r'\baren\'t\b', r'\bare not\b', r'\bhasn\'t\b', r'\bhas not\b',
    r'\bhaven\'t\b', r'\bhave not\b', r'\bhadn\'t\b', r'\bhad not\b', r'\bnever\b', r'\bno\b',
    r'\bnegative\b', r'\bdisapprove\b', r'\bunhappy\b', r'\bfail\b', r'\bwrong\b', r'\bhate\b', r'\bregret\b',
    r'\bcan\'t\b', r'\bcannot\b', r'\bwon\'t\b', r'\bwouldn\'t\b', r'\bdoesn\'t\b', r'\baren\'t\b',
    r'\bain\'t\b', r'\bwasn\'t\b', r'\bwere\'n\'t\b', r'\bhasn\'t\b', r'\bhaven\'t\b', r'\bhadn\'t\b',
    r'\bwon\'t\b', r'\bwouldn\'t\b', r'\bshouldn\'t\b', r'\bmistake\b', r'\bsad\b', r'\bworst\b',
    r'\bterrible\b', r'\bhorrible\b', r'\bdreadful\b', r'\bunpleasant\b', r'\bdisgusting\b', r'\bdislike\b',
    r'\bdispleasure\b', r'\bunhappy\b', r'\bunfortunate\b', r'\bneglect\b', r'\bunfortunate\b', r'\bunfavorable\b',
    r'\bawful\b', r'\bnasty\b', r'\bpathetic\b', r'\btragic\b', r'\bfault\b', r'\bfiasco\b', r'\bregrettable\b',
    r'\bsorry\b', r'\bupset\b', r'\bdissatisfied\b', r'\bdismal\b', r'\bdepressing\b', r'\bworry\b', r'\bhurt\b',
    r'\bfrown\b', r'\bcry\b', r'\bcrisis\b', r'\bproblem\b', r'\bdanger\b', r'\bdefect\b', r'\bdisaster\b',
    r'\bfear\b', r'\bpain\b', r'\bshame\b', r'\bhopeless\b', r'\bdoubt\b', r'\binsult\b', r'\bnervous\b',
    r'\bthreat\b', r'\bguilt\b', r'\bweak\b', r'\bhorror\b', r'\bshocking\b', r'\bawful\b', r'\brepulsive\b',
    r'\batrocious\b', r'\bnegative\b', r'\bdiscomfort\b', r'\binconvenience\b', r'\bdismay\b', r'\bdespair\b',
    r'\bdisgust\b', r'\bdespise\b', r'\bdiscontent\b', r'\bdispleasure\b', r'\bworry\b', r'\bfear\b',
    r'\banxiety\b', r'\binsecurity\b', r'\bsuspicion\b', r'\bdoubt\b', r'\buncertain\b', r'\bguilt\b',
    r'\bremorse\b', r'\bshame\b',
]

# Create a regular expression pattern for negative words
negative_pattern = re.compile('|'.join(negative_words), re.IGNORECASE)

# Sample data for intent classification
data = [
    ("Technical", "How do I fix a 404 error on my website?"),
    ("Technical", "My code is giving a segmentation fault. What should I do?"),
    ("Non-Technical", "What are the best books to read this summer?"),
    ("Non-Technical", "Tell me about the latest fashion trends."),
    # Add more data examples
]

# Separate data into intent labels and text
intents, texts = zip(*data)

# Create a DataFrame from the data
intent_df = pd.DataFrame({'text': texts, 'intent': intents})

# Load a pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(intent_df['intent'].unique()))

# Tokenize the input text and encode labels
encoding = tokenizer(list(intent_df['text']), truncation=True, padding=True)
intent_labels = intent_df['intent'].astype('category').cat.codes.values.astype(np.int64)  # Modify label encoding

# Define batch size
batch_size = 4

# Create a Dataset
class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # Convert labels to long data type
        return item

    def __len__(self):
        return len(self.labels)

intent_dataset = IntentDataset(encoding, intent_labels)

# Split dataset into training and evaluation sets
from sklearn.model_selection import train_test_split

train_texts, eval_texts, train_labels, eval_labels = train_test_split(intent_df['text'], intent_labels, test_size=0.2, random_state=42)

# Tokenize and encode the training and evaluation datasets
train_encoding = tokenizer(list(train_texts), truncation=True, padding=True)
eval_encoding = tokenizer(list(eval_texts), truncation=True, padding=True)

# Create training and evaluation datasets
train_dataset = IntentDataset(train_encoding, train_labels)
eval_dataset = IntentDataset(eval_encoding, eval_labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./intent_model',
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    evaluation_strategy='epoch',
    logging_dir='./logs',
)

# Create a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Use the training dataset
    eval_dataset=eval_dataset,    # Set the evaluation dataset
)

# Fine-tune the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()

# Define a function to classify sentiment using regular expressions
def classify_sentiment(prompt):
    return "Negative" if negative_pattern.search(prompt) else "Positive"

# Define a function to classify intent using the fine-tuned BERT model
def classify_intent(prompt):
    encoding = tokenizer(prompt, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**encoding).logits
    intent_id = torch.argmax(logits, dim=1).item()
    intent_labels = list(intent_df['intent'].unique())
    return intent_labels[intent_id]

# Define a function to generate responses based on sentiment and intent
def generate_response(prompt):
    sentiment = classify_sentiment(prompt)
    intent = classify_intent(prompt)

    if sentiment == "Positive":
        if intent == "Technical":
            # Generate response from the database
            rag_response = generate_rag_response(prompt)
            if rag_response is not None:
                return f"Positive, Technical: {rag_response}"
        elif intent == "Non-Technical":
            return "Sorry, cannot answer the question"

        # Generate response from OpenAI API
        openai_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )['choices'][0]['message']['content']

        # Save the response to the database
        save_response_to_database(prompt, openai_response)

        return f"Positive, {openai_response}"
    elif sentiment == "Negative":
        if intent == "Technical":
            # Generate response from the database
            rag_response = generate_rag_response(prompt)
            if rag_response is not None:
                return f"Negative, Technical: {rag_response}"
        elif intent == "Non-Technical":
            return "Sorry, cannot answer the question"

        # Generate response from OpenAI API
        openai_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )['choices'][0]['message']['content']

        # Save the response to the database
        save_response_to_database(prompt, openai_response)

        return f"Negative, {openai_response}"

@app.route("/", methods=['POST', 'GET'])
def query_view():
    if request.method == 'POST':
        prompt = request.form['prompt']

        response = generate_response(prompt)

        return jsonify({'response': response})

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
"""
#Version - 15 - Myself
"""
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset, DataLoader
from flask import Flask, render_template, request, jsonify
import sqlite3
import openai
import re
import pandas as pd
import threading

app = Flask(__name__)

# Load the saved BERT model and tokenizer
saved_model_path = "./content"  # Replace with the actual path
model = BertForSequenceClassification.from_pretrained(saved_model_path)
tokenizer = BertTokenizer.from_pretrained(saved_model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# OpenAI API Key (Replace with your actual API key)
openai.api_key = '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'

# Define a thread-local storage for database connections and cursors
thread_local = threading.local()

# Database connection functions
def get_db_connection():
    if not hasattr(thread_local, "db_connection"):
        thread_local.db_connection = sqlite3.connect('responses.db')
    return thread_local.db_connection

def get_db_cursor():
    if not hasattr(thread_local, "db_cursor"):
        thread_local.db_cursor = get_db_connection().cursor()
    return thread_local.db_cursor

# Function for RAG response generation
def generate_rag_response(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT prompt, response FROM responses WHERE prompt = ?", (user_question,))
    result = cursor.fetchone()

    if result:
        return result[1]
    else:
        return None

# Function to save response to the database
def save_response_to_database(prompt, response):
    cursor = get_db_cursor()

    # Create a table if it doesn't exist
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY,
            prompt TEXT,
            response TEXT
        )'''
    )

    # Insert response into the table
    cursor.execute('INSERT INTO responses (prompt, response) VALUES (?, ?)', (prompt, response))
    get_db_connection().commit()

# List of negative words
negative_words = [
    r'\bnot\b', r'\bwouldn\'t\b', r'\bdon\'t\b', r'\bshouldn\'t\b', r'\bdoesn\'t\b', r'\bwould not\b',
    r'\bdo not\b', r'\bshould not\b', r'\baren\'t\b', r'\bare not\b', r'\bhasn\'t\b', r'\bhas not\b',
    r'\bhaven\'t\b', r'\bhave not\b', r'\bhadn\'t\b', r'\bhad not\b', r'\bnever\b', r'\bno\b',
    r'\bnegative\b', r'\bdisapprove\b', r'\bunhappy\b', r'\bfail\b', r'\bwrong\b', r'\bhate\b', r'\bregret\b',
    r'\bcan\'t\b', r'\bcannot\b', r'\bwon\'t\b', r'\bwouldn\'t\b', r'\bdoesn\'t\b', r'\baren\'t\b',
    r'\bain\'t\b', r'\bwasn\'t\b', r'\bwere\'n\'t\b', r'\bhasn\'t\b', r'\bhaven\'t\b', r'\bhadn\'t\b',
    r'\bwon\'t\b', r'\bwouldn\'t\b', r'\bshouldn\'t\b', r'\bmistake\b', r'\bsad\b', r'\bworst\b',
    r'\bterrible\b', r'\bhorrible\b', r'\bdreadful\b', r'\bunpleasant\b', r'\bdisgusting\b', r'\bdislike\b',
    r'\bdispleasure\b', r'\bunhappy\b', r'\bunfortunate\b', r'\bneglect\b', r'\bunfortunate\b', r'\bunfavorable\b',
    r'\bawful\b', r'\bnasty\b', r'\bpathetic\b', r'\btragic\b', r'\bfault\b', r'\bfiasco\b', r'\bregrettable\b',
    r'\bsorry\b', r'\bupset\b', r'\bdissatisfied\b', r'\bdismal\b', r'\bdepressing\b', r'\bworry\b', r'\bhurt\b',
    r'\bfrown\b', r'\bcry\b', r'\bcrisis\b', r'\bproblem\b', r'\bdanger\b', r'\bdefect\b', r'\bdisaster\b',
    r'\bfear\b', r'\bpain\b', r'\bshame\b', r'\bhopeless\b', r'\bdoubt\b', r'\binsult\b', r'\bnervous\b',
    r'\bthreat\b', r'\bguilt\b', r'\bweak\b', r'\bhorror\b', r'\bshocking\b', r'\bawful\b', r'\brepulsive\b',
    r'\batrocious\b', r'\bnegative\b', r'\bdiscomfort\b', r'\binconvenience\b', r'\bdismay\b', r'\bdespair\b',
    r'\bdisgust\b', r'\bdespise\b', r'\bdiscontent\b', r'\bdispleasure\b', r'\bworry\b', r'\bfear\b',
    r'\banxiety\b', r'\binsecurity\b', r'\bsuspicion\b', r'\bdoubt\b', r'\buncertain\b', r'\bguilt\b',
    r'\bremorse\b', r'\bshame\b',
]

# Create a regular expression pattern for negative words
negative_pattern = re.compile('|'.join(negative_words), re.IGNORECASE)

# List of technical words
technical_words = [
    "algorithm","programming","database","server","software","hardware","networking","cybersecurity",
    "machine learning","artificial intelligence""cloud computing","data analysis","web development",
    "framework","API","debugging","encryption","compiler","operating system","microcontroller","data science",
    "neural network","blockchain", "virtual reality", "augmented reality", "IoT (Internet of Things)", "DevOps",
    "agile methodology", "version control","front-end","back-end","full-stack","SQL","NoSQL","responsive design",
    "algorithm complexity","UI/UX design","API integration","data mining","natural language processing",
    "Linux","Windows","macOS","containerization","serverless","CI/CD","firewall","router","VPN","Hadoop",
    "TensorFlow","PyTorch","Java","Python","JavaScript","C++","Ruby","Go","HTML","CSS(Cascading Style Sheets)",
    "DNS","HTTPS","TCP/IP","RESTful API","IoT sensors","ethical hacking","data visualization","cloud storage","data center", 
    "cyber threat","authentication","authorization","microservices","API gateway","Big Data","deep learning",
    "responsive web design","UX/UI design principles","distributed computing","load balancing","SQL injection",
    "cross-site scripting (XSS)","agile project management", "web security","server-side scripting","front-end framework",
    "back-end framework","container orchestration","continuous integration","continuous deployment", "communiction networks"
    "firewall configuration","network architecture","network protocol","data warehouse","data modeling", "cloud computing"
    "cloud architecture","data privacy","cloud security","server virtualization","edge computing","DevSecOps",
    "CI/CD pipeline","cloud-native","blockchain technology","quantum computing","cybersecurity threats",
    "data encryption","API design","software development lifecycle","database management","backend development",
    "frontend development","mobile app development","cloud services","data analytics","system architecture",
    "network security","cyber threats","ethical hacking","information security","computer vision","data engineering",
    "robotics","artificial neural networks","cloud migration","software testing","user interface design","user experience design",
    "parallel computing","algorithm optimization","biometric authentication","quantum cryptography",
    "software architecture","virtualization technology","automation engineering","bioinformatics","genetic algorithms",
    "quantitative analysis","embedded systems","internet protocols","web application security",
]

# Function to classify intent based on technical words
def classify_intent(prompt):
    prompt_lower = prompt.lower()
    
    # Check if any technical word is present in the user's input
    for word in technical_words:
        if word in prompt_lower:
            return "Technical"
    
    # If no technical word is found, classify as "Non-Technical"
    return "Non-Technical"

@app.route("/", methods=['POST', 'GET'])
def query_view():
    if request.method == 'POST':
        prompt = request.form['prompt']

        # Sentiment Analysis using regular expressions
        sentiment_intent = "Negative" if negative_pattern.search(prompt) else "Positive"

        # Intent Classification using technical words
        intent = classify_intent(prompt)

        response = ""
        if sentiment_intent == 'Positive':
            # Generate RAG Response
            rag_response = generate_rag_response(prompt)
            if rag_response is not None:
                # Determine Technical or Non-Technical
                if intent == 'Technical':
                    response = f"Positive, Technical: {rag_response}"
                else:
                    response = f"Positive, Non-Technical: Sorry, cannot answer the question"
            else:
                # If RAG response is None, send the question to OpenAI API
                openai_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )['choices'][0]['message']['content']

                # Save the response to the database
                save_response_to_database(prompt, openai_response)

                if intent == 'Technical':
                    response = f"Positive, Technical: {openai_response}"
                else:
                    response = f"{sentiment_intent}, Non-Technical: Sorry, cannot answer the question"

        else:
            response = f"{sentiment_intent}, Non-Technical: Sorry, cannot answer the question"

        return jsonify({'response': response})

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
"""

#Version - 16 - Yuvan
"""
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset, DataLoader
from flask import Flask, render_template, request, jsonify
import sqlite3
import openai
import re
import pandas as pd
import threading

app = Flask(__name__)

# Load the saved BERT model and tokenizer
saved_model_path = "./content"  # Replace with the actual path
model = BertForSequenceClassification.from_pretrained(saved_model_path)
tokenizer = BertTokenizer.from_pretrained(saved_model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# OpenAI API Key (Replace with your actual API key)
openai.api_key = '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'

# Define a thread-local storage for database connections and cursors
thread_local = threading.local()

# Database connection functions
def get_db_connection():
    if not hasattr(thread_local, "db_connection"):
        thread_local.db_connection = sqlite3.connect('responses.db')
    return thread_local.db_connection

def get_db_cursor():
    if not hasattr(thread_local, "db_cursor"):
        thread_local.db_cursor = get_db_connection().cursor()
    return thread_local.db_cursor

# Function for RAG response generation
def generate_rag_response(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT prompt, response FROM responses WHERE prompt = ?", (user_question,))
    result = cursor.fetchone()

    if result:
        return result[1]
    else:
        return None

# Function to save response to the database
def save_response_to_database(prompt, response):
    cursor = get_db_cursor()

    # Create a table if it doesn't exist
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY,
            prompt TEXT,
            response TEXT
        )'''
    )

    # Insert response into the table
    cursor.execute('INSERT INTO responses (prompt, response) VALUES (?, ?)', (prompt, response))
    get_db_connection().commit()

# List of negative words
negative_words = [
    r'\bnot\b', r'\bwouldn\'t\b', r'\bdon\'t\b', r'\bshouldn\'t\b', r'\bdoesn\'t\b', r'\bwould not\b',
    r'\bdo not\b', r'\bshould not\b', r'\baren\'t\b', r'\bare not\b', r'\bhasn\'t\b', r'\bhas not\b',
    r'\bhaven\'t\b', r'\bhave not\b', r'\bhadn\'t\b', r'\bhad not\b', r'\bnever\b', r'\bno\b',
    r'\bnegative\b', r'\bdisapprove\b', r'\bunhappy\b', r'\bfail\b', r'\bwrong\b', r'\bhate\b', r'\bregret\b',
    r'\bcan\'t\b', r'\bcannot\b', r'\bwon\'t\b', r'\bwouldn\'t\b', r'\bdoesn\'t\b', r'\baren\'t\b',
    r'\bain\'t\b', r'\bwasn\'t\b', r'\bwere\'n\'t\b', r'\bhasn\'t\b', r'\bhaven\'t\b', r'\bhadn\'t\b',
    r'\bwon\'t\b', r'\bwouldn\'t\b', r'\bshouldn\'t\b', r'\bmistake\b', r'\bsad\b', r'\bworst\b',
    r'\bterrible\b', r'\bhorrible\b', r'\bdreadful\b', r'\bunpleasant\b', r'\bdisgusting\b', r'\bdislike\b',
    r'\bdispleasure\b', r'\bunhappy\b', r'\bunfortunate\b', r'\bneglect\b', r'\bunfortunate\b', r'\bunfavorable\b',
    r'\bawful\b', r'\bnasty\b', r'\bpathetic\b', r'\btragic\b', r'\bfault\b', r'\bfiasco\b', r'\bregrettable\b',
    r'\bsorry\b', r'\bupset\b', r'\bdissatisfied\b', r'\bdismal\b', r'\bdepressing\b', r'\bworry\b', r'\bhurt\b',
    r'\bfrown\b', r'\bcry\b', r'\bcrisis\b', r'\bproblem\b', r'\bdanger\b', r'\bdefect\b', r'\bdisaster\b',
    r'\bfear\b', r'\bpain\b', r'\bshame\b', r'\bhopeless\b', r'\bdoubt\b', r'\binsult\b', r'\bnervous\b',
    r'\bthreat\b', r'\bguilt\b', r'\bweak\b', r'\bhorror\b', r'\bshocking\b', r'\bawful\b', r'\brepulsive\b',
    r'\batrocious\b', r'\bnegative\b', r'\bdiscomfort\b', r'\binconvenience\b', r'\bdismay\b', r'\bdespair\b',
    r'\bdisgust\b', r'\bdespise\b', r'\bdiscontent\b', r'\bdispleasure\b', r'\bworry\b', r'\bfear\b',
    r'\banxiety\b', r'\binsecurity\b', r'\bsuspicion\b', r'\bdoubt\b', r'\buncertain\b', r'\bguilt\b',
    r'\bremorse\b', r'\bshame\b',
]

# Create a regular expression pattern for negative words
negative_pattern = re.compile('|'.join(negative_words), re.IGNORECASE)

# List of technical words
technical_words = [
    "algorithm","programming","database","server","software","hardware","networking","cybersecurity",
    "machine learning","artificial intelligence""cloud computing","data analysis","web development",
    "framework","API","debugging","encryption","compiler","operating system","microcontroller","data science",
    "neural network","blockchain", "virtual reality", "augmented reality", "IoT (Internet of Things)", "DevOps",
    "agile methodology", "version control","front-end","back-end","full-stack","SQL","NoSQL","responsive design",
    "algorithm complexity","UI/UX design","API integration","data mining","natural language processing",
    "Linux","Windows","macOS","containerization","serverless","CI/CD","firewall","router","VPN","Hadoop",
    "TensorFlow","PyTorch","Java","Python","JavaScript","C++","Ruby","Go","HTML","CSS(Cascading Style Sheets)",
    "DNS","HTTPS","TCP/IP","RESTful API","IoT sensors","ethical hacking","data visualization","cloud storage","data center", 
    "cyber threat","authentication","authorization","microservices","API gateway","Big Data","deep learning",
    "responsive web design","UX/UI design principles","distributed computing","load balancing","SQL injection",
    "cross-site scripting (XSS)","agile project management", "web security","server-side scripting","front-end framework",
    "back-end framework","container orchestration","continuous integration","continuous deployment", "communiction networks"
    "firewall configuration","network architecture","network protocol","data warehouse","data modeling", "cloud computing"
    "cloud architecture","data privacy","cloud security","server virtualization","edge computing","DevSecOps",
    "CI/CD pipeline","cloud-native","blockchain technology","quantum computing","cybersecurity threats",
    "data encryption","API design","software development lifecycle","database management","backend development",
    "frontend development","mobile app development","cloud services","data analytics","system architecture",
    "network security","cyber threats","ethical hacking","information security","computer vision","data engineering",
    "robotics","artificial neural networks","cloud migration","software testing","user interface design","user experience design",
    "parallel computing","algorithm optimization","biometric authentication","quantum cryptography",
    "software architecture","virtualization technology","automation engineering","bioinformatics","genetic algorithms",
    "quantitative analysis","embedded systems","internet protocols","web application security","java","iot","javascript","html","css",
    "c++","c","sql",
]

# Function to classify intent based on technical words
def classify_intent(prompt):
    prompt_lower = prompt.lower()
    
    # Check if any technical word is present in the user's input
    for word in technical_words:
        if word in prompt_lower:
            return "Technical"
    
    # If no technical word is found, classify as "Non-Technical"
    return "Non-Technical"

@app.route("/", methods=['POST', 'GET'])
def query_view():
    if request.method == 'POST':
        prompt = request.form['prompt']

        # Sentiment Analysis using regular expressions
        sentiment_intent = "Negative" if negative_pattern.search(prompt) else "Positive"

        # Intent Classification using technical words
        intent = classify_intent(prompt)

        response = ""
        if sentiment_intent == 'Positive':
            # Generate RAG Response
            rag_response = generate_rag_response(prompt)
            if rag_response is not None:
                # Determine Technical or Non-Technical
                if intent == 'Technical':
                    response = f"Positive, Technical: {rag_response}"
                else:
                    response = f"Positive, Non-Technical: Sorry, cannot answer the question"
            else:
                # If RAG response is None, send the question to OpenAI API
                openai_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )['choices'][0]['message']['content']

                # Save the response to the database
                save_response_to_database(prompt, openai_response)

                if intent == 'Technical':
                    response = f"Positive, Technical: {openai_response}"
                else:
                    response = f"{sentiment_intent}, Non-Technical: Sorry, cannot answer the question"

        else:
            response = f"{sentiment_intent}, Non-Technical: Sorry, cannot answer the question"

        return jsonify({'response': response})

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
"""

#Version - 17 - Sabari
"""
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset, DataLoader
from flask import Flask, render_template, request, jsonify
import sqlite3
import openai
import re
import pandas as pd
import threading

app = Flask(__name__)

# Load the saved BERT model and tokenizer
saved_model_path = "./content"  # Replace with the actual path
model = BertForSequenceClassification.from_pretrained(saved_model_path)
tokenizer = BertTokenizer.from_pretrained(saved_model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# OpenAI API Key (Replace with your actual API key)
openai.api_key = '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'

# Define a thread-local storage for database connections and cursors
thread_local = threading.local()

# Database connection functions
def get_db_connection():
    if not hasattr(thread_local, "db_connection"):
        thread_local.db_connection = sqlite3.connect('responses.db')
    return thread_local.db_connection

def get_db_cursor():
    if not hasattr(thread_local, "db_cursor"):
        thread_local.db_cursor = get_db_connection().cursor()
    return thread_local.db_cursor

# Function for RAG response generation
def generate_rag_response(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT prompt, response FROM responses WHERE prompt = ?", (user_question,))
    result = cursor.fetchone()

    if result:
        return result[1]
    else:
        return None

# Function to save response to the database
def save_response_to_database(prompt, response):
    cursor = get_db_cursor()

    # Create a table if it doesn't exist
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY,
            prompt TEXT,
            response TEXT
        )'''
    )

    # Insert response into the table
    cursor.execute('INSERT INTO responses (prompt, response) VALUES (?, ?)', (prompt, response))
    get_db_connection().commit()

# List of negative words
negative_words = [
    r'\bnot\b', r'\bwouldn\'t\b', r'\bdon\'t\b', r'\bshouldn\'t\b', r'\bdoesn\'t\b', r'\bwould not\b',
    r'\bdo not\b', r'\bshould not\b', r'\baren\'t\b', r'\bare not\b', r'\bhasn\'t\b', r'\bhas not\b',
    r'\bhaven\'t\b', r'\bhave not\b', r'\bhadn\'t\b', r'\bhad not\b', r'\bnever\b', r'\bno\b',
    r'\bnegative\b', r'\bdisapprove\b', r'\bunhappy\b', r'\bfail\b', r'\bwrong\b', r'\bhate\b', r'\bregret\b',
    r'\bcan\'t\b', r'\bcannot\b', r'\bwon\'t\b', r'\bwouldn\'t\b', r'\bdoesn\'t\b', r'\baren\'t\b',
    r'\bain\'t\b', r'\bwasn\'t\b', r'\bwere\'n\'t\b', r'\bhasn\'t\b', r'\bhaven\'t\b', r'\bhadn\'t\b',
    r'\bwon\'t\b', r'\bwouldn\'t\b', r'\bshouldn\'t\b', r'\bmistake\b', r'\bsad\b', r'\bworst\b',
    r'\bterrible\b', r'\bhorrible\b', r'\bdreadful\b', r'\bunpleasant\b', r'\bdisgusting\b', r'\bdislike\b',
    r'\bdispleasure\b', r'\bunhappy\b', r'\bunfortunate\b', r'\bneglect\b', r'\bunfortunate\b', r'\bunfavorable\b',
    r'\bawful\b', r'\bnasty\b', r'\bpathetic\b', r'\btragic\b', r'\bfault\b', r'\bfiasco\b', r'\bregrettable\b',
    r'\bsorry\b', r'\bupset\b', r'\bdissatisfied\b', r'\bdismal\b', r'\bdepressing\b', r'\bworry\b', r'\bhurt\b',
    r'\bfrown\b', r'\bcry\b', r'\bcrisis\b', r'\bproblem\b', r'\bdanger\b', r'\bdefect\b', r'\bdisaster\b',
    r'\bfear\b', r'\bpain\b', r'\bshame\b', r'\bhopeless\b', r'\bdoubt\b', r'\binsult\b', r'\bnervous\b',
    r'\bthreat\b', r'\bguilt\b', r'\bweak\b', r'\bhorror\b', r'\bshocking\b', r'\bawful\b', r'\brepulsive\b',
    r'\batrocious\b', r'\bnegative\b', r'\bdiscomfort\b', r'\binconvenience\b', r'\bdismay\b', r'\bdespair\b',
    r'\bdisgust\b', r'\bdespise\b', r'\bdiscontent\b', r'\bdispleasure\b', r'\bworry\b', r'\bfear\b',
    r'\banxiety\b', r'\binsecurity\b', r'\bsuspicion\b', r'\bdoubt\b', r'\buncertain\b', r'\bguilt\b',
    r'\bremorse\b', r'\bshame\b',
]

# Create a regular expression pattern for negative words
negative_pattern = re.compile('|'.join(negative_words), re.IGNORECASE)

# List of technical words
technical_words = [
    "algorithm","programming","database","server","software","hardware","networking","cybersecurity",
    "machine learning","artificial intelligence""cloud computing","data analysis","web development",
    "framework","api","debugging","encryption","compiler","operating system","microcontroller","data science",
    "neural network","blockchain", "virtual reality", "augmented reality", "IoT (Internet of Things)", "DevOps",
    "agile methodology", "version control","front-end","back-end","full-stack","SQL","NoSQL","responsive design",
    "algorithm complexity","ui/ux design","api integration","data mining","natural language processing",
    "Linux","Windows","macOS","containerization","serverless","CI/CD","firewall","router","VPN","Hadoop",
    "tensorflow","pytorch","Java","Python","JavaScript","C++","Ruby","Go","HTML","CSS(Cascading Style Sheets)",
    "DNS","HTTPS","TCP/IP","RESTful API","IoT sensors","ethical hacking","data visualization","cloud storage","data center", 
    "cyber threat","authentication","authorization","microservices","API gateway","Big Data","deep learning",
    "responsive web design","UX/UI design principles","distributed computing","load balancing","SQL injection",
    "cross-site scripting (XSS)","agile project management", "web security","server-side scripting","front-end framework",
    "back-end framework","container orchestration","continuous integration","continuous deployment", "communiction networks"
    "firewall configuration","network architecture","network protocol","data warehouse","data modeling", "cloud computing"
    "cloud architecture","data privacy","cloud security","server virtualization","edge computing","DevSecOps",
    "CI/CD pipeline","cloud-native","blockchain technology","quantum computing","cybersecurity threats",
    "data encryption","API design","software development lifecycle","database management","backend development",
    "frontend development","mobile app development","cloud services","data analytics","system architecture",
    "network security","cyber threats","ethical hacking","information security","computer vision","data engineering",
    "robotics","artificial neural networks","cloud migration","software testing","user interface design","user experience design",
    "parallel computing","algorithm optimization","biometric authentication","quantum cryptography",
    "software architecture","virtualization technology","automation engineering","bioinformatics","genetic algorithms",
    "quantitative analysis","embedded systems","internet protocols","web application security","java","iot","javascript","html","css",
    "c++","c","sql","api","os","devops","nosql","ui/ux design","api integration","linux","windows","macos","ci/cd","vpn","hadoop","tensorflow",
    "pytorch","ruby","go","html","css","dns","https","tcp/ip","restful api","iot sensors","api gateway","big data","ux/ui design principles",
    "sql injection","ci/cd architecture","udp","c","c++","go"
]

# Function to classify intent based on technical words
def classify_intent(prompt):
    prompt_lower = prompt.lower()
    
    # Check if any technical word is present in the user's input
    for word in technical_words:
        if word in prompt_lower:
            return "Technical"
    
    # If no technical word is found, classify as "Non-Technical"
    return "Non-Technical"

@app.route("/", methods=['POST', 'GET'])
def query_view():
    if request.method == 'POST':
        prompt = request.form['prompt']

        # Sentiment Analysis using regular expressions
        sentiment_intent = "Negative" if negative_pattern.search(prompt) else "Positive"

        # Intent Classification using technical words
        intent = classify_intent(prompt)

        response = ""
        if sentiment_intent == 'Positive':
            # Generate RAG Response
            rag_response = generate_rag_response(prompt)
            if rag_response is not None:
                # Determine Technical or Non-Technical
                if intent == 'Technical':
                    response = f"Positive, Technical: {rag_response}"
                else:
                    response = f"Positive, Non-Technical: Sorry, cannot answer the question"
            else:
                # If RAG response is None, send the question to OpenAI API
                openai_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )['choices'][0]['message']['content']

                # Save the response to the database
                save_response_to_database(prompt, openai_response)

                if intent == 'Technical':
                    response = f"Positive, Technical: {openai_response}"
                else:
                    response = f"{sentiment_intent}, Non-Technical: Sorry, cannot answer the question"

        else:
            response = f"{sentiment_intent}, Non-Technical: Sorry, cannot answer the question"

        return jsonify({'response': response})

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

"""
#Version - 18 - Using SpaCy Model
"""
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset, DataLoader
from flask import Flask, render_template, request, jsonify
import sqlite3
import openai
import re
import threading
import spacy

app = Flask(__name__)

# Load the saved BERT model and tokenizer
saved_model_path = "./content"  # Replace with the actual path
model = BertForSequenceClassification.from_pretrained(saved_model_path)
tokenizer = BertTokenizer.from_pretrained(saved_model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# OpenAI API Key (Replace with your actual API key)
openai.api_key = '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'

# Define a thread-local storage for database connections and cursors
thread_local = threading.local()

# Load the spaCy language model for part-of-speech tagging
nlp = spacy.load("en_core_web_sm")

# ... (Other functions and lists from your original code)

# Function to classify technical words using spaCy
def classify_technical_words(prompt):
    doc = nlp(prompt)
    technical_words = []

    # Iterate through tokens and identify technical nouns
    for token in doc:
        if token.pos_ == "NOUN":
            technical_words.append(token.text)

    return technical_words

@app.route("/", methods=['POST', 'GET'])
def query_view():
    if request.method == 'POST':
        prompt = request.form['prompt']

        # Sentiment Analysis using regular expressions
        sentiment_intent = "Negative" if negative_pattern.search(prompt) else "Positive"

        # Identify Technical Words using spaCy
        technical_words = classify_technical_words(prompt)

        response = ""
        if sentiment_intent == 'Positive':
            # Generate RAG Response
            rag_response = generate_rag_response(prompt)
            if rag_response is not None:
                # Determine Technical or Non-Technical
                if technical_words:
                    response = f"Positive, Technical Words: {', '.join(technical_words)} - {rag_response}"
                else:
                    response = f"Positive, Non-Technical: {rag_response}"
            else:
                # If RAG response is None, send the question to OpenAI API
                openai_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )['choices'][0]['message']['content']

                # Save the response to the database
                save_response_to_database(prompt, openai_response)

                if technical_words:
                    response = f"Positive, Technical Words: {', '.join(technical_words)} - {openai_response}"
                else:
                    response = f"{sentiment_intent}, Non-Technical: {openai_response}"

        else:
            response = f"{sentiment_intent}, Non-Technical: Sorry, cannot answer the question"

        return jsonify({'response': response})

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
"""

#Version - 19 - With more Technical Words


import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset, DataLoader
from flask import Flask, render_template, request, jsonify
import sqlite3
import openai
import re
import pandas as pd
import threading

app = Flask(__name__)

# Load the saved BERT model and tokenizer
saved_model_path = "./content"  # Replace with the actual path
model = BertForSequenceClassification.from_pretrained(saved_model_path)
tokenizer = BertTokenizer.from_pretrained(saved_model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# OpenAI API Key (Replace with your actual API key)
openai.api_key = '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'

# Define a thread-local storage for database connections and cursors
thread_local = threading.local()

# Database connection functions
def get_db_connection():
    if not hasattr(thread_local, "db_connection"):
        thread_local.db_connection = sqlite3.connect('responses.db')
    return thread_local.db_connection

def get_db_cursor():
    if not hasattr(thread_local, "db_cursor"):
        thread_local.db_cursor = get_db_connection().cursor()
    return thread_local.db_cursor

# Function for RAG response generation
def generate_rag_response(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT prompt, response FROM responses WHERE prompt = ?", (user_question,))
    result = cursor.fetchone()

    if result:
        return result[1]
    else:
        return None

# Function to save response to the database
def save_response_to_database(prompt, response):
    cursor = get_db_cursor()

    # Create a table if it doesn't exist
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY,
            prompt TEXT,
            response TEXT
        )'''
    )

    # Insert response into the table
    cursor.execute('INSERT INTO responses (prompt, response) VALUES (?, ?)', (prompt, response))
    get_db_connection().commit()

# List of negative words
negative_words = [
    r'\bnot\b', r'\bwouldn\'t\b', r'\bdon\'t\b', r'\bshouldn\'t\b', r'\bdoesn\'t\b', r'\bwould not\b',
    r'\bdo not\b', r'\bshould not\b', r'\baren\'t\b', r'\bare not\b', r'\bhasn\'t\b', r'\bhas not\b',
    r'\bhaven\'t\b', r'\bhave not\b', r'\bhadn\'t\b', r'\bhad not\b', r'\bnever\b', r'\bno\b',
    r'\bnegative\b', r'\bdisapprove\b', r'\bunhappy\b', r'\bfail\b', r'\bwrong\b', r'\bhate\b', r'\bregret\b',
    r'\bcan\'t\b', r'\bcannot\b', r'\bwon\'t\b', r'\bwouldn\'t\b', r'\bdoesn\'t\b', r'\baren\'t\b',
    r'\bain\'t\b', r'\bwasn\'t\b', r'\bwere\'n\'t\b', r'\bhasn\'t\b', r'\bhaven\'t\b', r'\bhadn\'t\b',
    r'\bwon\'t\b', r'\bwouldn\'t\b', r'\bshouldn\'t\b', r'\bmistake\b', r'\bsad\b', r'\bworst\b',
    r'\bterrible\b', r'\bhorrible\b', r'\bdreadful\b', r'\bunpleasant\b', r'\bdisgusting\b', r'\bdislike\b',
    r'\bdispleasure\b', r'\bunhappy\b', r'\bunfortunate\b', r'\bneglect\b', r'\bunfortunate\b', r'\bunfavorable\b',
    r'\bawful\b', r'\bnasty\b', r'\bpathetic\b', r'\btragic\b', r'\bfault\b', r'\bfiasco\b', r'\bregrettable\b',
    r'\bsorry\b', r'\bupset\b', r'\bdissatisfied\b', r'\bdismal\b', r'\bdepressing\b', r'\bworry\b', r'\bhurt\b',
    r'\bfrown\b', r'\bcry\b', r'\bcrisis\b', r'\bproblem\b', r'\bdanger\b', r'\bdefect\b', r'\bdisaster\b',
    r'\bfear\b', r'\bpain\b', r'\bshame\b', r'\bhopeless\b', r'\bdoubt\b', r'\binsult\b', r'\bnervous\b',
    r'\bthreat\b', r'\bguilt\b', r'\bweak\b', r'\bhorror\b', r'\bshocking\b', r'\bawful\b', r'\brepulsive\b',
    r'\batrocious\b', r'\bnegative\b', r'\bdiscomfort\b', r'\binconvenience\b', r'\bdismay\b', r'\bdespair\b',
    r'\bdisgust\b', r'\bdespise\b', r'\bdiscontent\b', r'\bdispleasure\b', r'\bworry\b', r'\bfear\b',
    r'\banxiety\b', r'\binsecurity\b', r'\bsuspicion\b', r'\bdoubt\b', r'\buncertain\b', r'\bguilt\b',
    r'\bremorse\b', r'\bshame\b',
]

# Create a regular expression pattern for negative words
negative_pattern = re.compile('|'.join(negative_words), re.IGNORECASE)

# List of technical words
technical_words = [
    "algorithm","programming","database","server","software","hardware","networking","cybersecurity",
    "machine learning","artificial intelligence""cloud computing","data analysis","web development",
    "framework","API","debugging","encryption","compiler","operating system","microcontroller","data science",
    "neural network","blockchain", "virtual reality", "augmented reality", "IoT (Internet of Things)", "DevOps",
    "agile methodology", "version control","front-end","back-end","full-stack","SQL","NoSQL","responsive design",
    "algorithm complexity","UI/UX design","API integration","data mining","natural language processing",
    "Linux","Windows","macOS","containerization","serverless","CI/CD","firewall","router","VPN","Hadoop",
    "TensorFlow","PyTorch","Java","Python","JavaScript","C++","Ruby","Go","HTML","CSS(Cascading Style Sheets)",
    "DNS","HTTPS","TCP/IP","RESTful API","IoT sensors","ethical hacking","data visualization","cloud storage","data center", 
    "cyber threat","authentication","authorization","microservices","API gateway","Big Data","deep learning",
    "responsive web design","UX/UI design principles","distributed computing","load balancing","SQL injection",
    "cross-site scripting (XSS)","agile project management", "web security","server-side scripting","front-end framework",
    "back-end framework","container orchestration","continuous integration","continuous deployment", "communiction networks"
    "firewall configuration","network architecture","network protocol","data warehouse","data modeling", "cloud computing"
    "cloud architecture","data privacy","cloud security","server virtualization","edge computing","DevSecOps",
    "CI/CD pipeline","cloud-native","blockchain technology","quantum computing","cybersecurity threats",
    "data encryption","API design","software development lifecycle","database management","backend development",
    "frontend development","mobile app development","cloud services","data analytics","system architecture",
    "network security","cyber threats","ethical hacking","information security","computer vision","data engineering",
    "robotics","artificial neural networks","cloud migration","software testing","user interface design","user experience design",
    "parallel computing","algorithm optimization","biometric authentication","quantum cryptography",
    "software architecture","virtualization technology","automation engineering","bioinformatics","genetic algorithms",
    "quantitative analysis","embedded systems","internet protocols","web application security","java","iot","javascript","html","css",
    "c++","c","sql","api","os","devops","nosql","ui/ux design","api integration","linux","windows","macos","ci/cd","vpn","hadoop","tensorflow",
    "pytorch","ruby","go","html","css","dns","https","tcp/ip","restful api","iot sensors","api gateway","big data","ux/ui design principles",
    "sql injection","ci/cd architecture","python",
#AI keywords
    "machine learning", "deep learning", "neural networks", "natural language processing", "computer vision", "reinforcement learning", "supervised learning", 
    "unsupervised learning", "semi-supervised learning", "transfer learning", 
    "generative adversarial networks", "convolutional neural networks", "recurrent neural networks", "long short-term memory", "support vector machines", 
    "decision trees", "random forest", "clustering", "dimensionality reduction", "feature engineering", 
    "data preprocessing", "hyperparameter tuning", "model evaluation metrics", "overfitting", "underfitting",
    "bias-variance tradeoff", "gradient descent", "backpropagation", "activation functions", "loss functions", 
    "model deployment", "edge ai", "model interpretability", "explainable ai", "automl", "reinforcement learning frameworks",
    "ai ethics", "data privacy", "bias and fairness in ai", "robotic process automation", "cognitive computing", "expert systems", 
    "chatbots", "natural language understanding", "speech recognition", "image recognition", "anomaly detection", "recommender systems", 
    "time series forecasting", "quantum machine learning", "artificial intelligence", "data science", "big data", "deep reinforcement learning", 
    "unsupervised learning algorithms", "classification algorithms", "regression analysis", "ensemble learning", "nearest neighbor", "principal component analysis (pca)",
    "k-means clustering", "logistic regression", "cross-validation", "dropout", "learning rate", "transfer function", "neuroevolution", "bayesian networks", "markov decision process (mdp)",
    "q-learning", "recurrent q-network (rqn)", "word embeddings", "word2vec", "doc2vec", "bert", "gpt (generative pre-trained transformer)", "self-attention mechanism", "rnn cell", "lstm cell", 
    "perceptron", "gini index", "adaboost", "bagging", "random search", "bayesian optimization", "federated learning", "edge computing", "ai chips", "data labeling", "ai-powered automation", "ai chatbots",
    "ai in healthcare", "ai in finance", "ai in robotics", "ai in autonomous vehicles", "ai in natural language generation", "ai in recommendation systems", "ai in computer games", "ai in cybersecurity", "ai in image generation", 
    "ai in drug discovery", "ai in fraud detection", "ai in supply chain management", "ai in e-commerce", "ai in social media analytics", "ai in customer service", "ai in content moderation", "ai in speech synthesis", "ai in sentiment analysis",
    "ai in reinforcement learning environments", "ai in virtual reality", "ai in augmented reality", "hyperparameter optimization", "kernel methods", "one-shot learning", "attention mechanism", "transformer architecture", "exponential smoothing",
    "word embeddings", "word2vec", "fasttext", "t-sne", "autoencoders", "bayesian networks", "probabilistic graphical models", "monte carlo methods", "reinforcement learning algorithms", "actor-critic", "deep deterministic policy gradients (ddpg)",
    "proximal policy optimization (ppo)", "trust region policy optimization (trpo)", "asynchronous advantage actor-critic (a3c)", "imitation learning", "inverse reinforcement learning (irl)", "self-supervised learning", "few-shot learning", "multi-instance learning", 
    "semi-supervised gans", "adversarial training", "gan architectures", "conditional gans", "cyclegan", "pix2pix", "wasserstein gan (wgan)", "super-resolution gan (srgan)", "neuroevolution of augmenting topologies (neat)", "transfer learning in nlp", "word sense disambiguation", 
    "text summarization", "named entity recognition (ner)", "machine translation", "transformer-based language models", "bert variants (roberta, xlnet, etc.)", "attention is all you need", "convolutional sequence-to-sequence models", "image segmentation", "semantic segmentation", 
    "object detection", "yolo ", "pose estimation", "image captioning", "image style transfer", "anomaly detection techniques", "isolation forests", "autoencoders for anomaly detection", "matrix factorization", "collaborative filtering", "item-based recommendation",
    "user-based recommendation", "time series forecasting models", "arima", "prophet", "long short-term memory (lstm) for time series", "seasonal decomposition", "quantum machine learning algorithms", "quantum annealing", "quantum neural networks", "quantum support vector machines (qsvm)", 
    "quantum generative models", "quantum natural language processing", "contrastive learning", "self-supervised representation learning", "image generation models", "variational autoencoders (vaes)", "normalizing flows", "generative models for text", "topic modeling", "latent dirichlet allocation (lda)", 
    "self-organizing maps (som)", "quantum computing", "quantum annealer", "quantum circuits", "quantum entanglement", "quantum superposition", "quantum qubits", "quantum logic gates", "quantum algorithm", "quantum cryptography", "quantum simulators", "quantum error correction", "quantum advantage", 
    "quantum cloud services", "quantum hybrid systems", "quantum key distribution (qkd)", "quantum teleportation", "quantum cryptanalysis", "quantum cryptocurrency", "quantum machine learning frameworks", "quantum neural networks (qnn)", "quantum variational algorithms", "quantum walks", "quantum fourier transform",
    "quantum phase estimation", "quantum approximate optimization algorithm (qaoa)", "quantum support vector machine (qsvm)", "quantum circuit learning", "quantum-enhanced reinforcement learning", "quantum natural language processing", "quantum neural network libraries", "quantum error mitigation", "quantum sensing", 
    "quantum imaging", "quantum telecommunication", "quantum materials", "quantum entropy", "quantum robotics", "quantum biology", "quantum chemistry", "quantum finance", "quantum transportation", "quantum internet", "quantum cloud computing", "quantum cryptography protocols", "quantum data compression", "quantum game theory", 
    "quantum random number generation", "quantum metrology", "quantum machine learning algorithms", "quantum reinforcement learning", "quantum generative adversarial networks (qgans)", "quantum reinforcement learning frameworks", "quantum-enhanced sensors", "quantum-enhanced imaging", "quantum-enhanced communication", 
    "quantum-enhanced computing", "quantum-enhanced security", "quantum-enhanced optimization", "quantum-enhanced simulation", "quantum-enhanced machine learning", "quantum-enhanced ai", "quantum-enhanced decision making", "quantum cryptography post-quantum security", "quantum-resistant cryptography", "quantum-safe encryption",
    "quantum-safe authentication", "quantum-safe hash functions", "quantum-safe digital signatures", "quantum-safe key exchange", "quantum-safe","knn","svm","lstm","gan","cnn","rnn","dnn","rl","ar","vr","dl","nlp","nn","opencv",
#Microsoft keywords
    "windows", "office", "azure", "microsoft 365", "sql server", "power bi",
    "visual studio", "c#", ".net", "windows server", "active directory", "exchange server",
    "sharepoint", "powerapps", "dynamics 365", "microsoft teams", "azure devops", "power automate",
    "microsoft edge", "onenote", "windows 10", "powerpoint", "excel", "outlook",
    "word", "access", "onenote", "teams", "windows 11", "azure sql database",
    "azure functions", "azure app service", "azure blob storage", "microsoft graph", "power query",
    "power pivot", "power virtual agents", "power platform", "microsoft flow", "power bi desktop",
    "visual studio code", "typescript", "asp.net", "entity framework", "visual basic", "hololens",
    "xamarin", "azure virtual machines", "azure kubernetes service", "microsoft sql server", "microsoft access", "microsoft excel", "microsoft word", "microsoft outlook", "microsoft powerpoint", "microsoft onedrive", "microsoft sharepoint", "microsoft exchange online", "microsoft dynamics 365", "microsoft powerapps",
    "microsoft power automate", "microsoft power bi", "microsoft flow", "microsoft azure", "azure virtual network", "azure functions", "azure app service", "azure cosmos db", "azure machine learning", "azure active directory", "azure storage", "microsoft 365 admin center", "office 365", "azure devops services", 
    "azure resource manager", "azure logic apps", "azure data factory", "azure virtual machines", "windows server 2019", "windows server 2022", "active directory domain services", "microsoft iis", "microsoft hyper-v", "microsoft exchange server", "microsoft sharepoint server", "microsoft system center", "microsoft dynamics 365 sales", 
    "microsoft dynamics 365 customer service", "microsoft dynamics 365 finance", "microsoft dynamics 365 business central", "microsoft teams meetings", "microsoft teams chat", "microsoft teams channels", "microsoft edge browser", "microsoft onenote", "microsoft powerpoint online", "microsoft excel online", "microsoft word online", "microsoft access online", 
    "microsoft sharepoint online", "microsoft exchange online", "microsoft dynamics 365 admin", "microsoft powerapps admin", "microsoft power automate admin", "microsoft power bi admin", "microsoft flow admin", "microsoft azure portal", "azure virtual machines pricing", "azure functions pricing", "azure app service pricing", "azure cosmos db pricing",
    "azure machine learning pricing", "azure active directory pricing", "azure storage pricing", "microsoft 365 pricing", "azure devops pricing", "azure resource manager templates", "azure logic apps templates", "azure data factory templates", "microsoft certifications", "microsoft learn", "microsoft documentation", "microsoft community", "microsoft support",
]

# Function to classify intent based on technical words
def classify_intent(prompt):
    prompt_lower = prompt.lower()
    
    # Check if any technical word is present in the user's input
    for word in technical_words:
        if word in prompt_lower:
            return "Technical"
    
    # If no technical word is found, classify as "Non-Technical"
    return "Non-Technical"

@app.route("/", methods=['POST', 'GET'])
def query_view():
    if request.method == 'POST':
        prompt = request.form['prompt']

        # Sentiment Analysis using regular expressions
        sentiment_intent = "Negative" if negative_pattern.search(prompt) else "Positive"

        # Intent Classification using technical words
        intent = classify_intent(prompt)

        response = ""
        if sentiment_intent == 'Positive':
            # Generate RAG Response
            rag_response = generate_rag_response(prompt)
            if rag_response is not None:
                # Determine Technical or Non-Technical
                if intent == 'Technical':
                    response = f"Positive, Technical: {rag_response}"
                else:
                    response = f"Positive, Non-Technical: Sorry, cannot answer the question"
            else:
                # If RAG response is None, send the question to OpenAI API
                openai_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )['choices'][0]['message']['content']

                # Save the response to the database
                save_response_to_database(prompt, openai_response)

                if intent == 'Technical':
                    response = f"Positive, Technical: {openai_response}"
                else:
                    response = f"{sentiment_intent}, Non-Technical: Sorry, cannot answer the question"

        else:
            response = f"{sentiment_intent}, Non-Technical: Sorry, cannot answer the question"

        return jsonify({'response': response})

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

