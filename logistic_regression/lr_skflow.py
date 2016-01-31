#  Copyright 2015-present Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
This example showcases how simple it is to build image classification networks.
It follows description from this TensorFlow tutorial:
    https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html#deep-mnist-for-experts
"""

import random
from sklearn import datasets, cross_validation, metrics

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import skflow

### Download and load MNIST data.

mnist = input_data.read_data_sets('MNIST_data')

### Linear classifier.

classifier = skflow.TensorFlowLinearClassifier(
    n_classes=10, batch_size=100, steps=10000, learning_rate=0.01)
classifier.fit(mnist.train.images, mnist.train.labels)
score = metrics.accuracy_score(classifier.predict(mnist.test.images), mnist.test.labels)
print('Accuracy: {0:f}'.format(score))
