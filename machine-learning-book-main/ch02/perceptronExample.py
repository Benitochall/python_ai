import numpy as np
import pandas as pd
from requests import HTTPError

class Perceptron:
    """Perceptron classifier.

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
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        
        # X is the training data along with their features
        # y are our target values
        """Fit training data.

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
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Return class label after unit step"""\
        # so we pass in an array [5.1,1.4] this is then dotted with the weights and the bias is added, we return 1 if greater than
        # or equal to 0, else 0, this then gives us the class label, 0 or 1, in this case it guessed 1 when target was 0 
        # so the change in w was shifted lower, and the bias was also shifted lower 
        # this method is basically just finding a decision boundary
        return np.where(self.net_input(X) >= 0.0, 1, 0)


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

    ppn = Perceptron(eta=0.1, n_iter=10)

    ppn.fit(X, y)

if __name__ == "__main__":
    print("Perceptron class created successfully.")
    main()