# 🎬 Movie Recommender System using Collaborative Filtering

Welcome to the Movie Recommender System repository! This project leverages collaborative filtering to provide personalized movie recommendations based on user preferences and similarities between movies. The recommendation engine is trained on the MovieLens dataset, capturing user ratings for various movies.

## Features

- **User-based Recommendations**: Input a user_id to receive personalized movie recommendations based on similar users' preferences.

- **Movie-based Recommendations**: Enter a movie title to get recommendations for similar movies, calculated using collaborative filtering.

## Getting Started

Follow these steps to set up and run the Movie Recommender System:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/PranavAI2050/Recommender-System.git

2. Navigate to the project directory:
    ```bash
    cd recommender-system

3. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv

4. Activate the virtual environment:
   ```bash
   venv\Scripts\activate

5. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
6. Run the main script to train the recommendation model:
   ```bash
   python main.py

7. Finally, run the Flask app to interact with the recommendation system:
   ```bash
   python app.py

