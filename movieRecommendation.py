import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie ratings data
data = {
    'User': ['User1', 'User2', 'User3', 'User4', 'User5'],
    'Movie1': [4, 5, 0, 0, 1],
    'Movie2': [0, 0, 4, 5, 2],
    'Movie3': [1, 2, 3, 0, 0],
    'Movie4': [0, 3, 0, 4, 5],
}
df = pd.DataFrame(data)

# Calculate user similarity based on movie ratings
user_ratings = df.set_index('User')
user_similarity = cosine_similarity(user_ratings)

# Function to get movie recommendations for a user
def get_movie_recommendations(user, num_recommendations=3):
    user_index = df[df['User'] == user].index[0]
    sim_scores = user_similarity[user_index]
    
    # Sort movies by similarity
    similar_users = list(enumerate(sim_scores))
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)
    
    # Get movie recommendations
    recommendations = []
    for i in range(1, len(similar_users)):
        similar_user_index = similar_users[i][0]
        for movie in df.columns[1:]:
            if user_ratings.at[df.index[similar_user_index], movie] > 0 and user_ratings.at[user, movie] == 0:
                recommendations.append((movie, user_ratings.at[df.index[similar_user_index], movie]))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
    return recommendations[:num_recommendations]

# Example: Get movie recommendations for 'User1'
user = 'User1'
recommendations = get_movie_recommendations(user)
print(f"Movie recommendations for {user}:")
for movie, rating in recommendations:
    print(f"{movie}: Rating {rating}")
