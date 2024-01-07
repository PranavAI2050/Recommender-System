from src.data.generate import Prepare_data
from src.models.model import Model
from src.data.save import handle_data
import pandas as pd
import os
import pickle
from config import Config


def main():
    data_dir = "data/raw_data"   
    path = "ratings.csv"
    data = Prepare_data(os.path.join(data_dir, path))
    Y,R,W,b,X,Y_mat,R_mat = data.process()

    model = Model(Y, Config.n_features, Config.iterations,Config.lambda_,Config.Learning_rate)
    W,X,b = model.Build_Train_model()

    processed_data_dir = "data/processed"
    h_data = handle_data(processed_data_dir)
    h_data.save_useful_variables(W,X,b,Y)

    path_movies = os.path.join(data_dir, "movies.csv")
    data_movies = pd.read_csv(path_movies)
    movies = data_movies.title.values
    df_titles = pd.DataFrame({"movies":movies})
    df_titles.to_csv(os.path.join(processed_data_dir,'movies_list.csv'), index = False)
     
     

if __name__ == "__main__":
    main()
     
   
     

