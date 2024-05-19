import numpy as np
import pandas as pd
from requests import HTTPError

class AdalineSGD:
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    shuffle : bool (default: True)
      Shuffles training data every epoch if True to prevent cycles.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    b_ : Scalar
        Bias unit after fitting.
    losses_ : list
      Mean squared error loss function value averaged over all
      training examples in each epoch.

        
    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
        
    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : object

        """
        self._initialize_weights(X.shape[1]) # first intialize the weights to small random numbers
        self.losses_ = []
        for i in range(self.n_iter): # for the number of iterations
            if self.shuffle:
                # the resaon we shuffle is to prevent the model from getting stuck in a local minima
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi, target))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y)) # this is a good method for shuffling the data
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float_(0.)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi)) # first dot the input with the weights and then add the bias, then apply linear activation 
        # output is going to be an array of the same shape as the input [paramater1]
        # this is the changed value of feature vector
        error = (target - output) # this is error array of size 1
        self.w_ += self.eta * 2.0 * xi * (error) # this is the derivative of the loss function which we update the weights by
        self.b_ += self.eta * 2.0 * error # this is the derivative of the loss function which we update the bias by
        loss = error**2 # we compute the mean squared error loss which is just squaring the error
        return loss
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
    
def main():
    # need to set up the perceptron 

    try:
        s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        print('From URL:', s)
        df = pd.read_csv(s,
                        header=None,
                        encoding='utf-8')
    
    except HTTPError:
        s = 'iris.data'
        print('From local Iris path:', s)
        df = pd.read_csv(s,
                        header=None,
                        encoding='utf-8')


    # select setosa and versicolor
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 0, 1)

    # extract sepal length and petal length
    X = df.iloc[0:100, [0, 2]].values # now we get the first 100 rows the first and third feature columns


    X_std = np.copy(X)
    # we frist standardize the data to speed up the convergence
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
    ada_sgd.fit(X_std, y)
    # now lets say we wanted to predict data afte running the model,
    # all we need to do is normalize the data and then run the predict function
    new_data_point = np.array([5.1, 1.4])

    # Standardize the new data point using the same mean and std as the training data
    new_data_point_std = np.copy(new_data_point)
    new_data_point_std[0] = (new_data_point[0] - X[:, 0].mean()) / X[:, 0].std()
    new_data_point_std[1] = (new_data_point[1] - X[:, 1].mean()) / X[:, 1].std()

    # Predict the class label
    prediction = ada_sgd.predict(new_data_point_std.reshape(1, -1))
    print("Predicted class label:", prediction[0])

if __name__ == "__main__":
    print("Perceptron class created successfully.")
    main()