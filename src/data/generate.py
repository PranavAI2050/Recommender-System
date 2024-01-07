import pandas as pd
import numpy as np

class Prepare_data:
    def __init__(self, ratings_path):
        self.ratings = pd.read_csv(ratings_path)
    
    def process(self):
        self.Y = pd.pivot_table(self.ratings, values = "rating", columns = ["userId"], index = ["movieId"], aggfunc   = np.sum)
        self.R = pd.notna(self.Y).astype(int)
        self.W = np.random.rand(self.Y.shape[1],10)
        self.b = np.random.rand(1,self.Y.shape[1])
        self.X = np.random.rand(self.Y.shape[0],10)
        self.Y_mat = self.Y.values
        self.R_mat = self.R.values
        return self.Y, self.R, self.W, self.b, self.X, self.Y_mat, self.R_mat