import numpy as np
import pandas as pd
from requests import HTTPError

class AdalineGD:
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
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
      Mean squared eror loss function values in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
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
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1]) # generate some random weights for the features
        self.b_ = np.float_(0.)
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X) # dot all data in the training set with the weigthts 
            output = self.activation(net_input) # we then apply the activation function which is the identity function
            errors = (y - output) # now take the differnence between the class labels and the output of the activation function
            
            
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]

            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self

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
    X = df.iloc[0:100, [0, 2]].values
    ppn = AdalineGD(eta=0.01, n_iter=50)

    ppn.fit(X, y)
    pass

if __name__ == "__main__":
    print("Perceptron class created successfully.")
    main()