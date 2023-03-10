import tflearn as tfl 


def generate_neural_network(input_layer_size: int, output_layer_size: int) -> tfl.DNN:

    # input layer
    net = tfl.input_data(shape=[None, input_layer_size])

    # hidden layers with 10 neurons each
    net = tfl.fully_connected(net, 10) 
    net = tfl.fully_connected(net, 10)

    # output layer
    net = tfl.fully_connected(net, output_layer_size, activation='softmax') # softmax gives probabilities for each output

    net = tfl.regression(net)
    model = tfl.DNN(net) # deep neural network

    return model


def train(model: tfl.DNN, training, output, n_epoch=1000) -> None:
    model.fit(training, output, n_epoch=n_epoch, batch_size=8, show_metric=True)



