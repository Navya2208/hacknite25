import pandas as pd


ratings_df = pd.read_csv("ratings.csv")
print(ratings_df.head())


tags_df = pd.read_csv("tags.csv")
print(tags_df.head())

movies_df = pd.read_csv("movies.csv")
print(movies_df.head())

links_df = pd.read_csv("links.csv")
print(links_df.head())


# Example: Extract useful variables
user_ids = ratings_df['userId'].unique()
movie_ids = ratings_df['movieId'].unique()
average_ratings = ratings_df.groupby('movieId')['rating'].mean()

# Tagging info per movie
movie_tags = tags_df.groupby('movieId')['tag'].apply(list)

# Print summaries
print(f"\nTotal users: {len(user_ids)}")
print(f"Total movies rated: {len(movie_ids)}")
print(f"Sample average ratings:\n{average_ratings.head()}")

print(f"\nTags per movie (sample):\n{movie_tags.head()}")