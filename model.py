#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Discourse Relation Sense Classifier

Feel free to change/restructure the code below
"""

__author__ = 'Jayeol Chun'
import numpy as np
import tensorflow as tf
import json
from preprocessing import sense_dict

ARG1  = 'Arg1'
ARG2  = 'Arg2'
CONN  = 'Connective'
SENSE = 'Sense'
TYPE  = 'Type'
TOKEN = 'TokenList'
KEYS  = [ARG1, ARG2, CONN, SENSE]
TEXT  = 'RawText'
ID = 'DocID'
TYPE = 'Type'
FEATURE = "Feature"

class DRSClassifier(object):
  """TODO: Implement a FeedForward Neural Network for Discourse Relation Sense
      Classification using Tensorflow/Keras (tensorflow 2.0)"""
  def __init__(self):
    self.build()
    self.inverted_sense = self.invert_dict(sense_dict)

  def build(self):
    #CNN
    #1D
    '''
    self.model = tf.keras.models.Sequential([
      tf.keras.layers.Conv1D(250, 4, padding='valid', activation='relu',
                             strides=1, input_shape=(43, 50)),
      tf.keras.layers.GlobalMaxPooling1D(),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(100),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(21, activation='softmax')
    ])
    '''
    #2D
    '''
    self.model = tf.keras.models.Sequential([
      tf.keras.layers.Reshape((43, 50, 1), input_shape=(43, 50)),
      tf.keras.layers.Conv2D(250, (3, 50), padding='valid', activation='relu'),
      tf.keras.layers.GlobalMaxPooling2D(),
      tf.keras.layers.Dense(100),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(21, activation='softmax')
    ])
    '''

    #FFN

    self.model = tf.keras.models.Sequential([
      # `input_shape` only because it's the first layer, and it needs to know what dimensions to expect
      tf.keras.layers.Flatten(input_shape=(43, 50)),
      tf.keras.layers.Dense(512, activation='relu'),  # first hidden layer
      tf.keras.layers.Dense(256, activation='tanh'),  # second hidden layer
      tf.keras.layers.Dense(21, activation='softmax')  # output layer
    ])

    self.model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),  # `adam` is a good initial choice when experimenting a new neural network
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

  def train(self, train_instances, dev_instances):
    """TODO: Train the classifier on `train_instances` while evaluating
        periodically on `dev_instances`

    Args:
      train_instances: list
      dev_instances: list
    """
    train_x = np.asarray([instance["Feature"] for instance in train_instances])
    train_y = np.asarray([instance["Sense"] for instance in train_instances])

    val_x = np.asarray([instance["Feature"] for instance in dev_instances])
    val_y = np.asarray([instance['Sense'] for instance in dev_instances])

    self.model.fit(train_x, train_y, batch_size = 64, verbose=1, epochs=5, validation_data=(val_x, val_y))

  def predict(self, instances, export_file="./preds.json"):
    """TODO: Given a trained model, make predictions on `instances` and export
        predictions to a json file

    Args:
      instances: list
      export_file: str, where to save your model's predictions on `instances`

    Returns:

    """
    #print(sense_dict)
    #print(self.inverted_sense)
    test_x = np.asarray([instance[FEATURE] for instance in instances])
    #preds = self.model.predict(test_x)
    #self.model.predict(test_x)
    y_hat = [np.argmax(x) for x in self.model.predict(test_x)]
    #print(y_hat)

    with open(export_file, 'w') as f:
      for i, instance in enumerate(instances):
        output = {ARG1: {TOKEN: []}, ARG2: {TOKEN: []}, CONN: {TOKEN: []}}
        output[ARG1][TOKEN] = instance[ARG1][TOKEN]
        output[ARG2][TOKEN] = instance[ARG2][TOKEN]
        output[CONN][TOKEN] = instance[CONN][TOKEN]
        output[ID] = instance[ID]
        output[SENSE] = [self.inverted_sense[y_hat[i]]]
        output[TYPE] = instance[TYPE]
        #print(output)
        json.dump(output, f, ensure_ascii=False)
        f.write('\n')
        #f.write('{}\n'.format(json.dump(output, f, ensure_ascii=False)))
    #return preds

  def invert_dict(self, input_dict):
    return dict([[v,k] for k, v in input_dict.items()])
#if __name__ == "__main__":
#  print(sense_dict)