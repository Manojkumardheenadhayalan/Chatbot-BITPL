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

