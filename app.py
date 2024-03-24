import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re
import zipfile

# Function to load data
def load_data():
    with zipfile.ZipFile('ml-latest-small.zip', 'r') as zip_ref:
        zip_ref.extractall('data')

    movies_df = pd.read_csv('data/ml-latest-small/movies.csv')
    ratings_df = pd.read_csv('data/ml-latest-small/ratings.csv')
    return movies_df, ratings_df

# Function to preprocess movies data
def preprocess_movies(movies_df):
    movies_df['movie_name'] = movies_df['title'].apply(lambda x: re.match(r'^(.*?)\s*\(\d{4}\)$', x).group(1) if re.match(r'^(.*?)\s*\(\d{4}\)$', x) else x)
    return movies_df

# Function to create user-item matrix
def create_user_item_matrix(ratings_df, movies_df):
    merged_data = pd.merge(ratings_df, movies_df, on='movieId')
    user_item_matrix = merged_data.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    return user_item_matrix

# Function to calculate item similarity
def calculate_item_similarity(user_item_matrix):
    item_similarity = cosine_similarity(user_item_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)
    return item_similarity_df

# Function to get movie ID
def get_movie_id(movies_df, movie_title):
    movie_id = movies_df[movies_df['movie_name'] == movie_title]['movieId'].values
    if len(movie_id) > 0:
        return movie_id[0]
    else:
        st.warning(f"Movie '{movie_title}' not found in the dataset.")
        return None

# Function to create ID to title mapping
def create_id_to_title_mapping(movies_df):
    id_to_title_mapping = dict(zip(movies_df['movieId'], movies_df['movie_name']))
    return id_to_title_mapping

# Function to get recommendations for a given movie
def get_recommendations(movie_title, item_similarity_df, id_to_title_mapping, movies_df, n=3):
    movie_id = get_movie_id(movies_df, movie_title)
    if movie_id is None:
        return None
    
    similar_scores = item_similarity_df[movie_id].sort_values(ascending=False)
    similar_scores = similar_scores.drop(movie_id)
    top_similar_movies_ids = similar_scores.head(n).index
    top_similar_movies_titles = [id_to_title_mapping[movie_id] for movie_id in top_similar_movies_ids]

    return top_similar_movies_titles

# Main function to run Streamlit app
def main():
    # Load data
    movies_df, ratings_df = load_data()
    movies_df = preprocess_movies(movies_df)
    user_item_matrix = create_user_item_matrix(ratings_df, movies_df)
    item_similarity_df = calculate_item_similarity(user_item_matrix)
    id_to_title_mapping = create_id_to_title_mapping(movies_df)

    # Streamlit UI
    st.title("Movie Recommendation System")

    movie_name = st.text_input("Enter a movie name:")
    if st.button("Get Recommendations"):
        recommendations = get_recommendations(movie_name, item_similarity_df, id_to_title_mapping, movies_df)
        if recommendations:
            st.success("Recommendations:")
            for title in recommendations:
                st.write(title)

if __name__ == "__main__":
    main()
