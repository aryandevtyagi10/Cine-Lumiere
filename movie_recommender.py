import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os
from dotenv import load_dotenv

load_dotenv()
OMDB_API_KEY = os.getenv("OMDB_API_KEY")

def fetch_poster(title):
    try:
        url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&t={title}"
        res = requests.get(url).json()
        if res.get("Poster") and res["Poster"] != "N/A":
            return res["Poster"]
    except:
        pass
    return None

@st.cache_data
def load_data():
    column_names = ['movie_id', 'title', 'release_date', 'video_release_date',
                    'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                    "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
                    'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                    'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    df = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1',
                     names=column_names, usecols=list(range(0, 5)) + list(range(5, 24)))
    genre_cols = column_names[5:]
    df['genres'] = df[genre_cols].apply(lambda row: ' '.join([g for g, v in row.items() if v == 1]), axis=1)
    return df[['movie_id', 'title', 'genres']]

def get_all_recommendations(title, cosine_sim, indices, movies):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx]  # exclude self
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# Setup
movies = load_data()
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.write("Pick a movie and get similar movies based on your favourite genre!")

movie_list = ['Select a movie'] + sorted(movies['title'].tolist())
selected_movie = st.selectbox("Choose a movie:", movie_list)

# Init session state
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'display_count' not in st.session_state:
    st.session_state.display_count = 0
if 'full_list' not in st.session_state:
    st.session_state.full_list = []
if 'current_movie' not in st.session_state:
    st.session_state.current_movie = None

# Recommend button
if st.button("Recommend"):
    if selected_movie != 'Select a movie':
        st.session_state.current_movie = selected_movie
        st.session_state.full_list = get_all_recommendations(selected_movie, cosine_sim, indices, movies)
        st.session_state.recommendations = []
        st.session_state.display_count = 0

# Display recommendations in batches
if st.session_state.current_movie:
    full_list = st.session_state.full_list
    next_count = st.session_state.display_count + 5
    st.session_state.recommendations = full_list[:next_count]
    st.session_state.display_count = next_count

    st.subheader(f"Recommendations for **{st.session_state.current_movie}**:")
    for movie in st.session_state.recommendations:
        poster_url = fetch_poster(movie)
        col1, col2 = st.columns([1, 3])
        with col1:
    if poster_url:
        st.image(poster_url, width=100)
    else:
        st.write("No image available")

        with col2:
            st.markdown(f"**{movie}**")

    # Load more button
    if st.session_state.display_count < len(st.session_state.full_list):
        st.button("Load More")
