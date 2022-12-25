import tflearn as tfl 


def generate_neural_network(input_layer_size: int, output_layer_size: int) -> tfl.DNN:

    # input layer
    net = tfl.input_data(shape=[None, input_layer_size])

    # hidden layers with 8 neurons
    net = tfl.fully_connected(net, 8) 
    net = tfl.fully_connected(net, 8)

    # output layer
    net = tfl.fully_connected(net, output_layer_size, activation='softmax') # softmax gives probabilities for each output

    net = tfl.regression(net)
    model = tfl.DNN(net) # deep neural network

    return model


def train(model, training, output, n_epoch=1000):
    model.fit(training, output, n_epoch=n_epoch, batch_size=8, show_metric=True)



