import nltk
import numpy as np
import tflearn as tfl
import random
from processing import stem_words, parse_json


def word_collection(prompt: str, words: list[str]):
    collection = [0] * len(words)
    prompt_words = stem_words(nltk.word_tokenize(prompt))

    for pw in prompt_words:
        for i, w in enumerate(words):
            if pw == w: # word is in the sentence
                collection[i] = 1

    return np.array(collection)


def send_random_response(data:dict, tag:str, confidence:float):
    for label in data['intents']:
        if label['tag'] == tag:
            response = random.choice(label['responses'])
            confidence = format(confidence * 100, '.2f')
            print(f'IRIS: {response} (Confidence: {confidence} %)')
            break 
            

def chat(model: tfl.DNN, words: list[str], labels: list[str]):
    data = parse_json()
    min_confidence = 0.7

    while True:
        prompt = input("You: ").lower()

        if prompt in ['quit', 'exit']:
            break

        results = model.predict([word_collection(prompt, words)])[0]
        max_idx = np.argmax(results)
        highest = results[max_idx]
        tag = labels[max_idx] # tag with the highest probability

        if highest > min_confidence:
            send_random_response(data, tag, highest)
            if tag == 'goodbye':
                break
        else:
            send_random_response(data, 'noanswer', highest)

    