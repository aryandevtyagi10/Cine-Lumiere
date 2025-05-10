Cinlumeria - Movie Recommender System
Cinelumeri is a movie recommendation system built using Streamlit, which allows users to receive personalized movie suggestions based on genre similarity. Users can select a movie from a curated list and discover five similar movies using cosine similarity between genre features.

The app fetches movie posters using the OMDb API and displays them alongside the recommended titles for an engaging user experience.

Features:
🎬 Personalized movie recommendations based on genre.

🌐 Fetches movie posters and titles from the OMDb API.

💻 Built using Streamlit for an interactive web-based experience.

🧠 Leverages cosine similarity and TF-IDF vectorization for content-based filtering.

How it Works:
Users select a movie from the dropdown.

The system recommends the top 5 similar movies based on genre similarity.

Movie posters are displayed alongside titles for better visualization.

Tech Stack:
Streamlit - Frontend framework for the app.

Python - Core backend logic.

Scikit-learn - For building the recommendation algorithm.

OMDb API - Fetching movie metadata and posters.

How to Run:
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/cinelumeria.git
Install dependencies:

nginx
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app:

arduino
Copy
Edit
streamlit run app.py
