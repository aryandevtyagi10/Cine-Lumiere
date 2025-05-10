import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OMDb API Key (replace with your actual OMDb API key in .env file)
OMDB_API_KEY = os.getenv("OMDB_API_KEY")  # This will load the API key from the .env file

# Function to fetch movie poster from OMDb
def fetch_poster(title):
    try:
        url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&t={title}"
        res = requests.get(url).json()
        if res.get("Poster") and res["Poster"] != "N/A":
            return res["Poster"]
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching poster for {title}: {str(e)}")
        return None

# Load movie data from u.item
@st.cache_data
def load_data():
    column_names = ['movie_id', 'title', 'release_date', 'video_release_date',
                    'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                    "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
                    'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                    'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    df = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1',
                     names=column_names, usecols=list(range(0, 5)) + list(range(5, 24)))

    def get_genres(row):
        genres = []
        for genre, val in row.items():
            if val == 1:
                genres.append(genre)
        return ' '.join(genres)

    genre_cols = column_names[5:]
    df['genres'] = df[genre_cols].apply(get_genres, axis=1)
    df = df[['movie_id', 'title', 'genres']]
    return df

# Get movie recommendations based on cosine similarity
def get_recommendations(title, cosine_sim, indices, movies):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Get top 5 recommendations
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# Load and process movie data
movies = load_data()
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Streamlit app UI
st.title("🎬 Movie Recommendation System")
st.write("Pick a movie and get similar movies based on your favourite genre!")

# Movie selection
selected_movie = st.selectbox("Choose a movie:", sorted(movies['title'].tolist()))

# Recommend button
if st.button("Recommend"):
    recommendations = get_recommendations(selected_movie, cosine_sim, indices, movies)
    st.subheader("Top 5 Recommendations:")

    for movie in recommendations:
        poster_url = fetch_poster(movie)
        col1, col2 = st.columns([1, 3])
        with col1:
            if poster_url:
                st.image(poster_url, width=100)
            else:
                st.write("No image available")
        with col2:
            st.markdown(f"**{movie}**")
