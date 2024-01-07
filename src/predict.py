import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
class Predict_for_user:
    def __init__(self,X,Y,W,movie_path,ratings_path):
        self.Y = Y
        self.W = W
        self.X = X
        self.df_users  =  pd.DataFrame({"user_id": Y.columns, "user_vector":W.tolist()})
        self.similarity_matrix  = self.compute_similarity()
        self.movies_df = pd.read_csv(movie_path)
        self.ratings_df = pd.read_csv(ratings_path)
        self.movies_df = self.movies_df.set_index('movieId')
        self.ratings_df['num_ratings'] =self.ratings_df.groupby("movieId")['rating'].transform('count')
        self.ratings_df = self.ratings_df.query('num_ratings > 20')
            
    def compute_similarity(self):
        self.cosine_sim_matirx = cosine_similarity(self.df_users.user_vector.tolist(),self. df_users.user_vector.tolist())
        return self.cosine_sim_matirx
    
    def recommend(self, user_id):
        index = self.df_users.query("user_id==@user_id").index.values[0]

        self.scores_df = pd.DataFrame({"user_id":self.Y.columns, "scores":self.similarity_matrix[index,:]})
        self.scores_df = self.scores_df.sort_values(by=['scores'], ascending=False).reset_index(drop = True)
        self.scores_df = self.scores_df.drop([0]) 
        
        self.similar_users =  self.scores_df.user_id[:20].values
        user_ratings_df = self.ratings_df.query("userId in @self.similar_users")
        
        self.average_ratings = user_ratings_df.groupby('movieId')['rating'].mean()
        self.average_ratings  = self.average_ratings.sort_values(ascending=False)
        self.top_10_movie_ids = self.average_ratings.head(10).index
        
        return self.movies_df.title[self.top_10_movie_ids].values


class predict_for_movie:
    def __init__(self,Y,X,movie_path):
        self.movies_df = pd.read_csv(movie_path)
        self.df_items = pd.DataFrame({"item_id": Y.index, "item_vector":X.tolist()})
        self.similarity_matrix  = self.compute_similarity()
        self.Y = Y
        self.X = X
        
    def compute_similarity(self):
        self.cosine_sim_matirx = cosine_similarity(self.df_items.item_vector.tolist(), self.df_items.item_vector.tolist())
        return self.cosine_sim_matirx
    
    def recommend(self, movie_name):
        index = self.movies_df.query("title == @movie_name").index.values[0]
        
        self.scores_df = pd.DataFrame({"movie_id":self.Y.index, "scores":self.similarity_matrix[index,:]})
        self.scores_df = self.scores_df.sort_values(by=['scores'], ascending=False).reset_index(drop = True)
        self.scores_df = self.scores_df.drop([0]) 
        
        self.top_10_movie_ids = self.scores_df.movie_id[:10].values
        
        return self.movies_df[self.movies_df.movieId.isin(self.top_10_movie_ids)].title.values
        