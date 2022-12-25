import nltk
from nltk.stem.lancaster import LancasterStemmer
import json
import os
import numpy as np
from dataclasses import dataclass, field
import pickle

@dataclass
class Data:
    words: list[str]
    labels: list[str]
    keys: list[list[str]] = field(default_factory=list)
    values: list[list[str]] = field(default_factory=list)

def parse_json() -> dict:
    f = open("intents.json")
    data = json.load(f)
    f.close()
    return data


def stem_words(words: list[str]):
    stm = LancasterStemmer()
    return { stm.stem(w.lower()) for w in words if w != '?'}


def process_data():
    
    data = parse_json() 
    words = []
    labels = []
    keys = []
    values = []

    for intent in data['intents']:
        tag = intent['tag']
        for pattern in intent['patterns']:
            root_words = nltk.word_tokenize(pattern)
            wrds = stem_words(root_words)
            words.extend(wrds)
            keys.append(wrds)
            values.append(tag)

        if tag not in labels:
            labels.append(tag)

    return Data(words, labels, keys, values)


def one_hot_encoded(data: Data):
    
    training = []
    output = []
    zeros = [0] * len(data.labels)

    for i, doc in enumerate(data.keys):
        collection = []
        wrds = stem_words(doc)

        for w in data.words:
            collection.append(int(w in wrds)) # 1 if exists 0 otherwise

        output_row = zeros[:]
        output_row[data.labels.index(data.values[i])] = 1
        training.append(collection)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    return training, output



def read_data():
    with open("./data/data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
 
    return Data(words, labels), training, output


def save_data(data, training, output):
    filename = "data/data.pickle"
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    with open(filename, "wb") as f:
        pickle.dump((data.words, data.labels, training, output), f)