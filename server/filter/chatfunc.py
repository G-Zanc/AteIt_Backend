import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from filter.model import NeuralNet
#from model import NeuralNet


import json
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data_file = open('./filter/intents.json').read()
#data_file = open('server/filter/intents.json').read()
intents = json.loads(data_file)

FILE = "../data.pth"
#FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
words = data['words']
classes = data['classes']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

#INPUT SENTENCE HERE eg. what kind of back excercises should I do?
def runFilter(sentence):
    #sentence = "What kind of foods should I eat to lose weight?"
    sentence = nltk.word_tokenize(sentence)
    pattern_words = [lemmatizer.lemmatize(x.lower()) for x in sentence]
    X = []
    for w in words:
        X.append(1) if w in pattern_words else X.append(0)
    X = np.array(X)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(dtype=torch.float32).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    c = classes[predicted.item()]

    probs = torch.softmax(output, dim=1)
    p = probs[0][predicted.item()]

    #THIS PRINTS PREDICTED CLASS
    print(c)

    #THIS PRINTS PROBABILITY OF THAT CLASS
    print(p.item())

    #print(p.item().type())
    for i in intents["intents"]:
        if i["tag"] == c:
            response = random.choice(i["responses"])

    if p.item() > 0.8:
        return response
    
    else:
        return "Sorry, I haven't been built to answer that."
    
runFilter("Make me a workout routing for back?")