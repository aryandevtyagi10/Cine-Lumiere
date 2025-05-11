import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os
import time
from dotenv import load_dotenv

# Load environment variables
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

def get_recommendations(title, cosine_sim, indices, movies, start_idx=0, batch_size=5):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx]  # Exclude the searched movie
    sim_scores = sim_scores[start_idx:start_idx + batch_size]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# Load data
movies = load_data()
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Session state
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'recommend_index' not in st.session_state:
    st.session_state.recommend_index = 0
if 'auto_scroll' not in st.session_state:
    st.session_state.auto_scroll = False
if 'should_rerun' not in st.session_state:
    st.session_state.should_rerun = False

st.title("🎬 Movie Recommendation System")
st.write("Pick a movie and get similar movies based on your favourite genre!")

movie_list = ['Select a movie'] + sorted(movies['title'].tolist())
selected = st.selectbox("Choose a movie:", movie_list)

if st.button("Recommend"):
    if selected != 'Select a movie':
        st.session_state.selected_movie = selected
        st.session_state.recommendations = []
        st.session_state.recommend_index = 0
        st.session_state.auto_scroll = True

# Show recommendations
if st.session_state.selected_movie:
    new_recs = get_recommendations(
        st.session_state.selected_movie,
        cosine_sim,
        indices,
        movies,
        start_idx=st.session_state.recommend_index,
        batch_size=5
    )
    st.session_state.recommendations.extend(new_recs)
    st.session_state.recommend_index += len(new_recs)

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

    if len(new_recs) == 5 and st.session_state.auto_scroll:
        st.session_state.should_rerun = True
        time.sleep(1)  # Wait before rerun (simulate scroll)

    if not st.session_state.auto_scroll:
        if st.button("Load More"):
            st.session_state.auto_scroll = True
            st.experimental_rerun()

# Trigger rerun only outside Streamlit layout
if st.session_state.should_rerun:
    st.session_state.should_rerun = False
    st.experimental_rerun()
