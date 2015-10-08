
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities
from lasagne.updates import nesterov_momentum, rmsprop, adagrad
from nolearn.lasagne import NeuralNet
import numpy as np




class EarlyStopping(object):

    def __init__(self, patience=100, criterion='valid_loss',
                 criterion_smaller_is_better=True):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None
        self.criterion = criterion
        self.criterion_smaller_is_better = criterion_smaller_is_better

    def __call__(self, nn, train_history):
        current_valid = train_history[-1][self.criterion]
        current_epoch = train_history[-1]['epoch']
        if self.criterion_smaller_is_better:
            cond = current_valid < self.best_valid
        else:
            cond = current_valid > self.best_valid
        if cond is True:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            if nn.verbose:
                print("Early stopping.")
                print("Best valid loss was {:.6f} at epoch {}.".format(
                    self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            if nn.verbose:
                print("Weights set.")
            raise StopIteration()

    def load_best_weights(self, nn, train_history):
        nn.load_weights_from(self.best_weights)

def build_model():
    hyper_parameters = dict(
    #hidden4_regularization = regularization.l1,
    #hidden5_regularization = regularization.l2,
    
    # handlers
)
    
    L=[
        (layers.InputLayer, {'shape':(None, 3, 64, 64)}),
        (layers.Conv2DLayer, {'num_filters':32, 'filter_size':(2,2), 'pad':0}),
        (layers.MaxPool2DLayer, {'pool_size': (3, 3)}),
        (layers.Conv2DLayer, {'num_filters':32, 'filter_size':(2,2), 'pad':0}),
        (layers.MaxPool2DLayer, {'pool_size': (2, 2)}),
        (layers.Conv2DLayer, {'num_filters':16, 'filter_size':(2,2), 'pad':0}),
        (layers.MaxPool2DLayer, {'pool_size': (1, 1)}),
        (layers.DenseLayer, {'num_units': 512, 'nonlinearity':nonlinearities.leaky_rectify}),
        (layers.DropoutLayer, {'p':0.5}),
        (layers.DenseLayer, {'num_units': 512, 'nonlinearity':nonlinearities.leaky_rectify}),
        (layers.DropoutLayer, {'p':0.5}),
        (layers.DenseLayer, {'num_units': 256, 'nonlinearity':nonlinearities.tanh}),
        (layers.DropoutLayer, {'p':0.2}),
        (layers.DenseLayer, {'num_units': 18, 'nonlinearity':nonlinearities.softmax}),
    ]


    net = NeuralNet(
        layers=L,
        update=adagrad,
        update_learning_rate=0.01,
        use_label_encoder=True,
        verbose=1,
        max_epochs=30,
        on_epoch_finished=[EarlyStopping(patience=50, criterion='valid_loss')]
        )
    return net




class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model()

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X
    
    def preprocess_y(self, y):
        return y.astype(np.int32)

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
