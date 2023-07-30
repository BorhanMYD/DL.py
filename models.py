import tensorflow as tf
from tensorflow.keras import layers as ksl
import numpy as np
from matplotlib import pyplot as plt
from keras.layers.serialization import activation
class Model:
  def __init__(self):
    self.net=None
    self.inputShape=[784]
  def buildmodel(self):
    self.net=tf.keras.Sequential([ksl.Dense(256,activation="tanh",
                                         input_shape=self.inputShape),  
                               ksl.Dense(128,activation="tanh"),
                               ksl.Dense(64,activation="tanh"),
                               ksl.Dense(10,activation="softmax")])
  def compileModel(self):
    tf.keras.utils.plot_model(self.net,"model.png")
    self.net.summary()
    loss=tf.keras.losses.CategoricalCrossentropy()
    optim=tf.keras.optimizers.SGD(learning_rate=0.001)
    self.net.compile(loss=loss)
model=Model()
model.buildmodel()
model.compileModel()
