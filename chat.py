import nltk
import numpy as np
import tflearn as tfl
import os
import random
from preprocessing import stem_words, parse_json


def word_collection(s, words):
    collection = [0] * len(words)
    s_words = nltk.word_tokenize(s)
    s_words = stem_words(s_words)

    for s_word in s_words:
        for i, w in enumerate(words):
            if w == s_word: # word is in the sentence
                collection[i] = 1

    return np.array(collection)


def chat(model: tfl.DNN, words: list[str], labels: list[str]):
    os.system('cls')
    min_confidence = 0.82

    while True:
        prompt = input("You: ").lower()

        if prompt in ['quit', 'exit']:
            break

        results = model.predict([word_collection(prompt, words)])[0]
        max_idx = np.argmax(results)
        highest = results[max_idx]
        tag = labels[max_idx] # tag with the highest probability

        if highest > min_confidence:
            data = parse_json()
            for label in data['intents']:
                if label['tag'] == tag:
                    responses = label['responses']
                    break
            
            print(f'IRIS: {random.choice(responses)}')
            
            if tag == 'goodbye':
                break
        else:
            print(f'IRIS: Sorry, I didn\'t understand.')
        print(f'Confidence: {highest}')