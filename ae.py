import tensorflow as tf
import numpy as np
import pprint

def configure_config(config):

    if type(config['Activations']) is not list:
        config['Activations'] = (len(config['HiddenLayers'])+1)*[config['Activations']]
    
    if type(config['InitWeights']) is not list:
        config['InitWeights'] = (len(config['HiddenLayers'])+1)*[config['InitWeights']]

    return config

class AutoEncoder(object):

    def __init__(self, n_features, config): #neurons, activations):

        self.n_features = n_features
        self.config = config#.self.configure_config(config)

        self.neurons = self.config['HiddenLayers']
        self.latent_ix = int(np.floor(len(self.neurons)/2))

        self.init_weights()

        self.optimizer = tf.optimizers.Adam()
        self.total_n_epochs = 0


    def print_config(self):
        pprint.pprint(self.config)

    def init_weights(self):
        self.names = []
        self.trainable_weights = []

        initializer = self.config['InitWeights'][0]
        w = tf.Variable(initializer(shape=[self.neurons[0], self.n_features]))
        self.names.append("layer1")
        self.trainable_weights.append(w)

        for ix in range(1,len(self.neurons)):
            initializer = self.config['InitWeights'][ix]
            w = tf.Variable(initializer(shape=[self.neurons[ix],self.neurons[ix-1]]))
            self.names.append("layer%s" %(ix+1))
            self.trainable_weights.append(w)

        initializer = self.config['InitWeights'][len(self.neurons)]
        w = tf.Variable(initializer(shape=[self.n_features,self.neurons[-1]]))
        self.names.append("layer%s" %(len(self.neurons)+1))
        self.trainable_weights.append(w)

    def get_weights(self, name):

        idx = self.names.index("%s" %(name))
        w = self.trainable_weights[idx]
        
        return w

    def dense(self, inputs, weights):

        return tf.linalg.matvec(weights,inputs)


    def encoder(self, input):

        x = input
        for ix in range(self.latent_ix+1):
            x = self.dense(x, self.get_weights(name="layer%s" %(ix+1)))
            x = self.config['Activations'][ix](x)

        return x
        
    def decoder(self, input):

        x = input
        for ix in range(self.latent_ix+1, len(self.neurons)+1):
            x = self.dense(x, self.get_weights(name="layer%s" %(ix+1)))
            x = self.config['Activations'][ix](x)

        return x

    def training_step(self, input):

        with tf.GradientTape() as tape:

            inputhat = self.predict(input)
            loss = tf.reduce_mean(tf.square(input - inputhat))

        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        return inputhat, loss

    def fit(self, input, epochs = 1, optimizer = None, verbose = False):

        if optimizer is not None:
            self.optimizer = optimizer

        for epoch_ix in range(epochs):
            inputhat, loss = self.training_step(input)
            if verbose: print("epochs = %s, loss = %s" %(str(epoch_ix), str(loss.numpy())))



        return inputhat, loss

    def predict(self, input):

        latent = self.encoder(input)
        inputhat = self.decoder(latent)

        return inputhat

