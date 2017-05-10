from sklearn import model_selection

import pandas as pd
import numpy as np
from bunch import Bunch

import tensorflow as tf


def load_iris():
    """
    $ floyd run --data GY3QRFFUA8KpbnqvroTPPW:training "python script.py"
    みたいな書き方で/trainingにマウントされるらしい。
    dataは[[a,b,c,d],[a,b,c,d],...,[a,b,c,d]]
    targetは[0,0,...,2]
    target_names = ["setosa", "versicolor", "virginica"]    
    """

    iris = pd.read_csv("/dataset/iris.data", header=None)
    iris = iris.replace({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})
    dataset = Bunch()
    dataset.data = iris.as_matrix([0, 1, 2, 3]).astype(np.float32)
    dataset.target = np.array(iris[4])

    return dataset


def main(args):
    # Load dataset
    iris = load_iris()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
      iris.data, iris.target, test_size=0.2, random_state=42)

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[10, 20, 10],
                                                n_classes=3,
                                                model_dir="/output")

    def get_train_inputs():
        x = tf.constant(x_train)
        y = tf.constant(y_train)
        return x, y

    # Fit and predict.
    classifier.fit(input_fn=get_train_inputs, steps=200)

    def get_test_inputs():
        x = tf.constant(x_test)
        y = tf.constant(y_test)
        return x, y

    score = classifier.evaluate(input_fn=get_test_inputs, steps=1)
    print("\nTest Accuracy: {0}\n".format(score['accuracy']))


if __name__ == '__main__':
    tf.app.run()
