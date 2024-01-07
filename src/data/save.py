import os
import numpy as np
import pandas as pd

class handle_data:
    def __init__(self, processed_data_dir):
        self.processed_data_dir = processed_data_dir
    def save_useful_variables(self,W, X, b, Y):
        np.save(os.path.join(self.processed_data_dir, "User_vector.npy"), W)    
        np.save(os.path.join(self.processed_data_dir, "movie_vector.npy"), X)
        np.save(os.path.join(self.processed_data_dir, "weights.npy"), b)   
        Y.to_csv(os.path.join(self.processed_data_dir, "Training_array.csv"), index=False)
    def load_useful_variables(self):
        W = np.load(os.path.join(self.processed_data_dir, "User_vector.npy"))
        X = np.load(os.path.join(self.processed_data_dir, "movie_vector.npy"))
        b = np.load(os.path.join(self.processed_data_dir, "weights.npy"))
        Y = pd.read_csv(os.path.join(self.processed_data_dir, "Training_array.csv"))
        return W,X,b,Y