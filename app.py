from flask import Flask, render_template, request
from src.predict import Predict_for_user, predict_for_movie
from src.data.save import handle_data
import os
import pandas as pd

app = Flask(__name__)

raw_data_dir = "data/raw_data"
path_ratings = os.path.join(raw_data_dir, "ratings.csv")
path_movies = os.path.join(raw_data_dir, "movies.csv")
processed_data_dir = "data/processed"

h_data = handle_data(processed_data_dir)
W, X, b, Y = h_data.load_useful_variables()

User_predictor = Predict_for_user(X, Y, W, path_movies, path_ratings)
movies_predictor = predict_for_movie(Y, X, path_movies)

titles = pd.read_csv(os.path.join(processed_data_dir,'movies_list.csv'))
movies = titles.movies.values


@app.route('/')
def index():
    return render_template('index.html', movies = movies)
 
@app.route('/predict_movie', methods=['POST'])
def predict_movie():
    if request.method == 'POST':
        movie_name = request.form["movie_name"]
        prediction = movies_predictor.recommend(movie_name)
        recommendations_movie = prediction.tolist()  
        return render_template('index.html', recommendations_movie=recommendations_movie,movies = movies)

if __name__ == '__main__':
    app.run(debug=True)
