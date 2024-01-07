import tensorflow as tf
from tensorflow import keras
import numpy as np
def cofi_cost_func_v(X, W, b, Y, lambda_):
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)
    j = tf.where(tf.math.is_nan(j), 0.0, j)
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J

class Y_scale:
    def __init__(self,Y):
        self.Y = Y
        self.y_mean , self.y_diff= self.fit(Y)
        
    def fit(self,Y):
        self.y_mean = np.nanmean(Y, axis = 1 , keepdims = True)
        self.y_max = np.nanmax(Y, axis = 1, keepdims = True)
        self.y_min = np.nanmin(Y, axis = 1, keepdims = True)
        self.y_diff = self.y_max - self.y_min
        return self.y_mean, self.y_diff
        
    def Scale(self,Y):
        self.y_norm = (Y - self.y_mean)/self.y_diff
        return self.y_norm
    
    def inverse_scale(self,Y):
        self.Y_original =  ( Y * self.y_diff) + self.y_mean
        return self.Y_original