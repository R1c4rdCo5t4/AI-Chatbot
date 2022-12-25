import os
import numpy as np
import tensorflow as tf
import tflearn as tfl
import random
import glob

from preprocessing import *
from model import *


if os.path.isfile('./data/data.pickle'):
    print("Reading JSON file...")
    data, training, output = read_data()
else:
    print("Processing JSON file...")
    data = process_data()
    training, output = one_hot_encoded(data)
    save_data(data, training, output)


tf.compat.v1.reset_default_graph()
model = generate_neural_network(len(training[0]), len(output[0]))

if glob.glob('./data/model.tflearn*'):
    print("Reading model.tflearn...")
    model.load('./data/model.tflearn')
else:
    print("Training Model")
    train(model, training, output)
    model.save('./data/model.tflearn')


