from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import string

app = Flask(__name__)

# Define your model and tokenizer here
chatbot_model = 'chatbot_model'
tokenizer = ''
max_sequence_length = ''
le = ''
responses = ''

@app.route('/')
def home():
    return "hey"

@app.route('/chat', methods=['POST'])
def chat():
    texts_p = []
    prediction_input = request.form['message']

    # Remove punctuation and convert to lowercase
    prediction_input = [letter.lower() for letter in prediction_input if letter not in string.punctuation]
    prediction_input = ''.join(prediction_input)
    texts_p.append(prediction_input)

    # Tokenize and pad the input
    prediction_input = tokenizer.texts_to_sequences(texts_p)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input], max_sequence_length)

    # Get the model's output prediction
    output = chatbot_model.predict(prediction_input)
    output = output.argmax()

    # Find the corresponding response based on the predicted tag
    response_tag = le.inverse_transform([output])[0]
    response = random.choice(responses[response_tag])

    return {'response': response}

if __name__ == '__main__':
    app.run()
