import ae
from pyod.utils.data import generate_data
import numpy as np
import tensorflow as tf

# Build Config
config = dict()
config['HiddenLayers'] = [6,6,5,6,3]
config['Optimizer'] = tf.optimizers.Adam
config['Activations'] = tf.keras.activations.relu
config['InitWeights'] = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal', seed=None)
config['PreTrain'] = False

clean_config = ae.configure_config(config)



# Build Testdata
X_train, y_train, _, _ = generate_data(
    n_train=20000,  
    n_test=200,
    n_features=10,
    contamination=0.1,
    random_state=42)

X = X_train[np.where(y_train < 1)[0]].astype(np.float32)
X = X[np.where(y_train < 1)[0]]


AE = ae.AutoEncoder(n_features = X.shape[1], config = config)
AE.fit(X, epochs = 100, verbose=True)