from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
import sqlite3
import openai
from flask import Flask, render_template, request
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import threading

app = Flask(__name__)

# Load pretrained GPT-2 model and tokenizer
generation_model = GPT2LMHeadModel.from_pretrained('gpt2')
generation_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')



# Load the saved model and tokenizer
saved_model_path = "./content"  # Replace with the actual path
model = BertForSequenceClassification.from_pretrained(saved_model_path)
tokenizer = BertTokenizer.from_pretrained(saved_model_path)

# Set the device to GPU if available, else use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load pretrained Sentence-BERT model
retrieval_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# OpenAI API Key
#openai.api_key = 'YOUR_OPENAI_API_KEY'
openai.api_key = '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'

# Define a thread-local storage for database connections and cursors
thread_local = threading.local()

def get_db_connection():
    # Create a new connection if not already in the thread-local storage
    if not hasattr(thread_local, "db_connection"):
        thread_local.db_connection = sqlite3.connect('responses.db')
    return thread_local.db_connection

def get_db_cursor():
    # Create a new cursor if not already in the thread-local storage
    if not hasattr(thread_local, "db_cursor"):
        thread_local.db_cursor = get_db_connection().cursor()
    return thread_local.db_cursor

def sbert_retrieval(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT id, prompt, response FROM responses")
    results = cursor.fetchall()

    prompts = [row[1] for row in results]

    # Encode prompts using Sentence-BERT
    prompt_embeddings = retrieval_model.encode(prompts, convert_to_tensor=True)
    user_question_embedding = retrieval_model.encode([user_question], convert_to_tensor=True)

    # Calculate cosine similarity scores
    similarities = util.pytorch_cos_sim(user_question_embedding, prompt_embeddings)[0]
    ranked_indices = similarities.argsort(descending=True)

    return [results[i] for i in ranked_indices]

def generate_rag_response(user_question):
    cursor = get_db_cursor()
    cursor.execute("SELECT prompt, response FROM responses WHERE prompt = ?", (user_question,))
    result = cursor.fetchone()

    if result:
        prompt, answer = result
        return f"Question: {prompt}\nAnswer: {answer}"
    else:
        cursor.execute("SELECT prompt, response FROM responses")
        all_results = cursor.fetchall()
        for prompt, answer in all_results:
            similarity = util.pytorch_cos_sim(
                retrieval_model.encode([user_question], convert_to_tensor=True),
                retrieval_model.encode([prompt], convert_to_tensor=True)
            )[0][0]
            if similarity > 0.6:
                return f"Question: {prompt}\nAnswer: {answer}"

    return None

@app.route("/", methods=['POST', 'GET'])
def query_view():
    if request.method == 'POST':
        prompt = request.form['prompt']
        
        # Tokenize user input
        input_tokens = tokenizer(prompt, padding=True, truncation=True, return_tensors='pt', max_length=128)

        # Perform prediction
        with torch.no_grad():
            model.eval()
            input_ids = input_tokens['input_ids'].to(device)
            attention_mask = input_tokens['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            predicted_class = torch.argmax(outputs.logits).item()
        class_labels = ['Non-Technical', 'Technical']
        predicted_label = class_labels[predicted_class]
        
        if predicted_label == 'Technical':
            # Perform the desired action for technical input (generate RAG response)
            rag_response = generate_rag_response(prompt)
            if rag_response is not None:
                return jsonify({'response': rag_response})
            else:
                # If RAG response is None, send the question to OpenAI API
                openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your OpenAI API key
                openai_response = openai.Completion.create(
                    engine="davinci-codex",
                    prompt=prompt,
                    max_tokens=50,
                )
                response = openai_response.choices[0].text.strip()
                
                # Save the response to the database
                save_response_to_database(prompt, response)
                
                return jsonify({'response': response})
            
        else:
            return jsonify({'response': "Non-Technical"})

    return render_template('index.html')

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

if __name__ == "__main__":
    app.run(debug=True)