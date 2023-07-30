import tensorflow as tf
from tensorflow.keras import layers as ksl
import numpy as np
from matplotlib import pyplot as plt
class Utils:
 def __init__(self):
    self.xTrain=None
    self.yTrain=None
    self.xTest=None
    self.yTest=None
 def loadData(self):
    mnist=tf.keras.datasets.mnist
    (self.xTrain,self.yTrain),(self.xTest,self.yTest)=mnist.load_data()   
 def ploSomeData(self):
  idx=np.random.choice(60000,9)           
  imgs=self.xTrain[idx,:,:]                
  _,axis=plt.subplots(3,3,figsize=[12,12])      
  axis = axis.flatten()
  for i,ax in enumerate(axis):
    ax.imshow(imgs[i])
  plt.show
  utils=Utils()
utils.loadData()
utils.ploSomeData()
