import streamlit as st
import pandas as pd
import kagglehub
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load Dataset ---
path = kagglehub.dataset_download("parasharmanas/movie-recommendation-system")
df = pd.read_csv(path + "/movies.csv")

# --- Data Preparation ---
def combine_features(row):
    features = []
    for col in ['genres', 'keywords', 'tagline', 'cast', 'director']:
        if col in row and pd.notna(row[col]):
            features.append(str(row[col]))
    return " ".join(features)

df['combined_features'] = df.apply(combine_features, axis=1)

# --- Use a smaller sample for memory efficiency ---
df_small = df.sample(2000, random_state=42)  # only 2000 movies
cv = CountVectorizer(stop_words='english')
count_matrix = cv.fit_transform(df_small['combined_features'])
cosine_sim = cosine_similarity(count_matrix)

# --- Recommendation Function ---
def get_recommendations(title, num=5):
    if title not in df_small['title'].values:
        return ["Movie not found in dataset."]
    
    idx = df_small[df_small['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[df_small.index.get_loc(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num+1]  # Skip first (same movie)
    movie_indices = [df_small.index[i[0]] for i in sim_scores]
    return df.loc[movie_indices, 'title'].tolist()

# --- Streamlit UI ---
st.title("Movie Recommendation System")
st.write("Select a movie you like, and we'll recommend similar movies.")

# Check if dataset is loaded successfully
if not df.empty:
    movie_list = df['title'].dropna().unique()
    selected_movie = st.selectbox("Choose a movie:", movie_list)
    
    if st.button("Recommend"):
        recommendations = get_recommendations(selected_movie)
        
        if isinstance(recommendations, list) and recommendations and recommendations[0] != "Movie not found in dataset.":
            st.subheader(f"Movies similar to '{selected_movie}':")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.error("Movie not found in dataset.")
else:
    st.error("Failed to load movie dataset.")
