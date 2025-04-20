import streamlit as st
import pandas as pd
from recommender_backend import recommend, data

st.title("🎬 Movie Recommendation Engine")
movie_list = data['title'].dropna().unique().tolist()
movie_list.sort()

selected_movie = st.selectbox("Choose a movie you like:", movie_list)
n = st.slider("Number of recommendations:", 1, 20, 5)

if st.button("Recommend"):
    try:
        result = recommend(selected_movie, n)
        st.dataframe(result.reset_index(drop=True))
    except Exception as e:
        st.error(f"Error: {e}")
