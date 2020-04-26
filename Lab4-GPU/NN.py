import numpy as np

class MLP():
    def __init__(self, nn_architecture, matmul=None, seed=99):
        self.nn_architecture = nn_architecture
        self.params_values = self.create_init_model(seed)
        if matmul:
            self.matmul = matmul
        else:
            self.matmul = np.dot

    def create_init_model(self, seed):
        # random seed initiation
        np.random.seed(seed)
        # parameters storage initiation
        params_values = {}

        # iteration over network layers
        for idx, layer in enumerate(self.nn_architecture):
            # we number network layers from 1
            layer_idx = idx + 1

            # extracting the number of units in layers
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]

            # initiating the values of the W matrix and vector b for subsequent layers with random values
            params_values['W' + str(layer_idx)] = np.random.randn(layer_output_size, layer_input_size) * 0.1
            params_values['b' + str(layer_idx)] = np.random.randn(layer_output_size, 1) * 0.1
        return params_values

    def sigmoid(self, Z):
        return (1.0 / (1.0 + np.exp(-Z)))

    def relu(self, Z):
        z = np.maximum(0, Z)
        return z

    def sigmoid_backward(self, dA, Z):
        sig = self.sigmoid(Z)
        return (dA * sig * (1.0 - sig))

    def relu_backward(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def single_layer_forward_propagation(self, A, W, b, activation="relu"):
        # TODO: calculate Z, the input value for the activation function of the layer
        if activation == "relu":
            activation_func = self.relu
        elif activation == "sigmoid":
            activation_func = self.sigmoid
        else:
            raise Exception('Non-supported activation function')

        Z = self.matmul(W, A) + b
        
        return activation_func(Z), Z

    def full_forward_propagation(self, X):
        # creating a temporary memory to store the information needed for a backward step
        memory = {}
        # X vector is the activation for layer 0
        A_curr = X

        # TODO: iterate over network layers and do the calculations of the activation values for each layer
        #  based on the input of the previous layer and the weights of the current layer.
        #  Remember to use the function single_layer_forward_propagation
        #  Note: you have to save the calculated values in the memory dictionary because they are being used in the backpropagation
        #  memory contain activation values of all the layers starting form the input layer memory['A0'] = X, memory['A1']=activation(Z1) : Z1=W1*X+b1, etc...
        #  and the inputs of the activation function Z, where memory['Z1'] = W1*X+b1, memory['Z2'] = W2*A1+b2, etc...

        #  [Your code goes here]
        memory["A0"] = A_curr
        for prev_layer_i, layers in enumerate(self.nn_architecture):
            curr_layer_i = prev_layer_i + 1
            W = self.params_values["W"+str(curr_layer_i)] 
            b = self.params_values["b"+str(curr_layer_i)]
            func = layers["activation"]
            A_curr, Z_curr = self.single_layer_forward_propagation(A_curr, W, b, func)
            memory["A"+str(curr_layer_i)] = A_curr
            memory["Z"+str(curr_layer_i)] = Z_curr
            
        y = A_curr
        # return of prediction vector and a dictionary containing intermediate values
        return y, memory

    def get_cost_value(self, Y_hat, Y):
        # number of examples
        m = Y_hat.shape[1]
        Y = Y.reshape(Y_hat.shape)
        # TODO: calculate of the cost according to the binary cross-entropy formula
        cost = (-1/m)*(self.matmul(Y, np.log(Y_hat.T)) + self.matmul((1-Y), np.log(1-Y_hat.T)))
        
        return np.squeeze(cost)

    # an auxiliary function that converts probability into class label
    def convert_prob_into_class(self, probs):
        probs_ = np.copy(probs)
        probs_[probs_ > 0.5] = 1
        probs_[probs_ <= 0.5] = 0
        return probs_

    def get_accuracy_value(self, Y_hat, Y):
        Y_hat_ = self.convert_prob_into_class(Y_hat)
        return (Y_hat_ == Y).all(axis=0).mean()

    def single_layer_backward_propagation(self, dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
        # number of examples
        m = A_prev.shape[1]

        # selection of activation function
        if activation == "relu":
            backward_activation_func = self.relu_backward
        elif activation == "sigmoid":
            backward_activation_func = self.sigmoid_backward
        else:
            raise Exception('Non-supported activation function')

        # calculation of the activation function derivative
        dZ_curr = backward_activation_func(dA_curr, Z_curr)

        # derivative of the matrix W
        dW_curr = self.matmul(dZ_curr, A_prev.T) / m
        # derivative of the vector b
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        # derivative of the matrix A_prev
        dA_prev = self.matmul(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr

    def full_backward_propagation(self, Y_hat, Y, memory):
        grads_values = {}

        # number of examples
        m = Y.shape[1]
        # a hack ensuring the same shape of the prediction vector and labels vector
        Y = Y.reshape(Y_hat.shape)

        # initiation of gradient descent algorithm
        dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

        for layer_idx_prev, layer in reversed(list(enumerate(self.nn_architecture))):
            # we number network layers from 1
            layer_idx_curr = layer_idx_prev + 1
            # extraction of the activation function for the current layer
            activ_function_curr = layer["activation"]

            dA_curr = dA_prev

            A_prev = memory["A" + str(layer_idx_prev)]
            Z_curr = memory["Z" + str(layer_idx_curr)]

            W_curr = self.params_values["W" + str(layer_idx_curr)]
            b_curr = self.params_values["b" + str(layer_idx_curr)]

            dA_prev, dW_curr, db_curr = self.single_layer_backward_propagation(
                dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

            grads_values["dW" + str(layer_idx_curr)] = dW_curr
            grads_values["db" + str(layer_idx_curr)] = db_curr

        return grads_values

    def update(self, grads_values, learning_rate):

        # TODO: iterate over network layers and update the values of W and b stored in params_values based on the gradients stored in grads_values.
        #  This dictionary contains the gradients values for each layer in the form grads_values['dW1], grads_values['db1], etc...
        #  which represent dL/dw1, dL/db1 respectivly, and the same for each layer.
        for i, layer in enumerate(self.nn_architecture):
            i += 1
            self.params_values["W"+str(i)] -= grads_values["dW"+str(i)]*learning_rate
            self.params_values["b"+str(i)] -= grads_values["db"+str(i)]*learning_rate


    def train(self, X, Y, epochs, learning_rate, verbose=False):

        # initiation of lists storing the history of metrics calculated during the learning process
        cost_history = []
        accuracy_history = []

        # TODO: Use the previously developed methods (full_forward_propagation, full_backward_propagation, update) to train the neural network.
        #  Remember to calculate the cost and accuracy after each forward pass to store the values in the cost_history, and accuracy_history for monitoring
        for i in range(epochs):

            # [Your code goes here]
            y_hat, memory = self.full_forward_propagation(X)
            cost = self.get_cost_value(y_hat, Y)
            accuracy = self.get_accuracy_value(y_hat, Y)
            cost_history.append(cost)
            accuracy_history.append(accuracy)
            grads = self.full_backward_propagation(y_hat,Y, memory)
            self.update(grads, learning_rate)
            if (i % 100 == 0):
                if (verbose):
                    print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(i, cost, accuracy))

        return {'loss': cost_history, 'acc': accuracy_history}

    def predict(self, X):
        y, _ = self.full_forward_propagation(X)
        return y
