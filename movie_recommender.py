# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import requests
# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # OMDb API Key
# OMDB_API_KEY = os.getenv("OMDB_API_KEY")

# # Function to fetch movie poster from OMDb
# def fetch_poster(title):
#     try:
#         url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&t={title}"
#         res = requests.get(url).json()
#         if res.get("Poster") and res["Poster"] != "N/A":
#             return res["Poster"]
#         else:
#             return None
#     except Exception as e:
#         st.error(f"Error fetching poster for {title}: {str(e)}")
#         return None

# # Load movie data from u.item
# @st.cache_data
# def load_data():
#     column_names = ['movie_id', 'title', 'release_date', 'video_release_date',
#                     'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
#                     "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
#                     'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
#                     'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

#     df = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1',
#                      names=column_names, usecols=list(range(0, 5)) + list(range(5, 24)))

#     def get_genres(row):
#         genres = []
#         for genre, val in row.items():
#             if val == 1:
#                 genres.append(genre)
#         return ' '.join(genres)

#     genre_cols = column_names[5:]
#     df['genres'] = df[genre_cols].apply(get_genres, axis=1)
#     df = df[['movie_id', 'title', 'genres']]
#     return df

# # Get movie recommendations based on cosine similarity
# def get_recommendations(title, cosine_sim, indices, movies):
#     if title not in indices:
#         return []
#     idx = indices[title]
#     sim_scores = list(enumerate(cosine_sim[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     sim_scores = sim_scores[1:6]  # Top 5
#     movie_indices = [i[0] for i in sim_scores]
#     return movies['title'].iloc[movie_indices].tolist()

# # Load and process data
# movies = load_data()
# tfidf = TfidfVectorizer(stop_words='english')
# tfidf_matrix = tfidf.fit_transform(movies['genres'])
# cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
# indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# # Streamlit UI
# st.title("🎬 Movie Recommendation System")
# st.write("Pick a movie and get similar movies based on your favourite genre!")

# # Dropdown with placeholder
# movie_list = ['Select a movie'] + sorted(movies['title'].tolist())
# selected_movie = st.selectbox("Choose a movie:", movie_list)

# # Recommend button
# if selected_movie != 'Select a movie' and st.button("Recommend"):
#     recommendations = get_recommendations(selected_movie, cosine_sim, indices, movies)
#     st.subheader("Top 5 Recommendations:")

#     for movie in recommendations:
#         poster_url = fetch_poster(movie)
#         col1, col2 = st.columns([1, 3])
#         with col1:
#             if poster_url:
#                 st.image(poster_url, width=100)
#             else:
#                 st.write("No image available")
#         with col2:
#             st.markdown(f"**{movie}**")
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
movies = pd.read_csv("movies.csv", sep="|", encoding="latin-1", header=None)
movies.columns = [
    "movie_id", "title", "release_date", "video_release_date", "IMDb_URL",
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

# Convert genres into a list of genres for each movie
def combine_features(row):
    genres = []
    for genre in ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", 
                  "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", 
                  "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]:
        if row[genre] == 1:
            genres.append(genre)
    return row['title'] + " " + " ".join(genres)

movies["combined_features"] = movies.apply(combine_features, axis=1)

# Vectorize combined features
vectorizer = CountVectorizer().fit_transform(movies["combined_features"])
cosine_sim = cosine_similarity(vectorizer)

# Recommendation function
def recommend_movie(movie_title, top_n=5):
    movie_title = movie_title.lower()
    
    # Get index of the movie that matches the title exactly
    indices = movies[movies['title'].str.lower().str.contains(movie_title)]
    
    if indices.empty:
        return "Movie not found!"
    
    # If multiple matches, pick the first one
    idx = indices.index[0]
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort and exclude the movie itself
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Remove the searched movie from the recommendations
    sim_scores = [score for score in sim_scores if score[0] != idx]
    
    # Get top N recommendations
    sim_scores = sim_scores[:top_n]
    
    # Filter results that are from the same franchise (contain "Harry Potter" in title if that's the input)
    if "harry potter" in movie_title:
        recommended_indices = [
            i for i, score in sim_scores if "harry potter" in movies.iloc[i].title.lower()
        ]
        # If not enough HP matches, fall back to top similar ones
        if len(recommended_indices) < top_n:
            recommended_indices += [i for i, score in sim_scores if i not in recommended_indices][:top_n - len(recommended_indices)]
    else:
        recommended_indices = [i for i, score in sim_scores]

    # Return movie titles
    return movies.iloc[recommended_indices][['title', 'release_date']]

# Test the function
results = recommend_movie("Harry Potter", top_n=5)
print(results)
