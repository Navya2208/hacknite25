
# Movie Recommendation Engine

*TRACK AI/ML*

This is a content-based movie recommendation system that suggests similar titles based on attributes like genre, cast, director, and description. The software uses TF-IDF vectorization and cosine similarity to match users' preferred movies with similar options from the dataset.  
 

*Links*
- Demo Video: https://youtu.be/u6pJzbRjqmQ    
- Dataset: https://www.kaggle.com/code/abdurahmanmrezk/netflix-analysis


## Contents
Features/Functionality 
- Recommends 1-20 similar movies/TV shows  
- Processes multiple features including plot descriptions  
- Interactive web interface with dropdown selection  
- Fast similarity matching using sklearn  

## Contributors

POTINI SAHITI - BT2024163

NAVYA SHARMA - BT2024237
## Installation

1. Ensure Python 3.8+ is installed  

2. Install requirements:  
    pip install streamlit pandas scikit-learn
    pandas
    sklearn.feature_extraction.text.        TfidfVectorizer
    sklearn.metrics.pairwise.cosine_similarity
    streamlit

3. Place `netflix_processed.csv` in your project folder  

4. Run:  
`streamlit run app.py`  

5. Access at: `http://localhost:8501` 