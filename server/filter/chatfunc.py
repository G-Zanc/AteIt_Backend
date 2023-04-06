import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from model import NeuralNet

import json
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
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

#INPUT SENTENCE HERE
#What kind of back excercises should I do?
sentence = "What kind of foods should I eat to lose weight?"
sentence = nltk.word_tokenize(sentence)
pattern_words = [lemmatizer.lemmatize(x.lower()) for x in sentence]
X = []
for w in words:
    X.append(1) if w in pattern_words else X.append(0)
X = np.array(X)
X = X.reshape(1, X.shape[0])
X = torch.from_numpy(X).to(dtype=torch.float32).to(device)

print(X.dtype)
output = model(X)
_, predicted = torch.max(output, dim=1)
c = classes[predicted.item()]
probs = torch.softmax(output, dim=1)
prob = probs[0][predicted.item()]

#THIS PRINTS PREDICTED CLASS
print(c)

#THIS PRINTS PROBABILITY OF THAT CLASS
print(prob.item())