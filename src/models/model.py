import tensorflow as tf
from tensorflow import keras
import sys
sys.path.append("/path/to/Recommender_system")
from src.data.preprocess import Y_scale,cofi_cost_func_v
class Model:
    def __init__(self,Y,num_features, iterations, lambda_,learning_rate):
        self.Y = Y
        self.num_movies, self.num_users = Y.shape
        self.num_features  = num_features
        self.iterations = iterations
        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        self.Data = Y_scale(Y)
        self.Ynorm = self.Data.Scale(Y)
        
    def Build_Train_model(self):
        tf.random.set_seed(1234)
        self.W = tf.Variable(tf.random.normal((self.num_users,  self.num_features),dtype=tf.float64),  name='W')
        self.X = tf.Variable(tf.random.normal((self.num_movies, self.num_features),dtype=tf.float64),  name='X')
        self.b = tf.Variable(tf.random.normal((1,          self.num_users),   dtype=tf.float64),  name='b')
        self.optimizer = keras.optimizers.Adam(learning_rate = self.learning_rate)
        for i in range(self.iterations):
            with  tf.GradientTape() as tape:
                cost_value = cofi_cost_func_v(self.X, self.W, self.b, self.Ynorm, self.lambda_)
                grads = tape.gradient(cost_value, [self.X,self.W,self.b])
                self.optimizer.apply_gradients(zip(grads, [self.X,self.W,self.b]))
                if i%20 ==0 :
                    print(f"Training loss at iteration {i}: {cost_value:0.1f}")
        return self.W, self.X, self.b