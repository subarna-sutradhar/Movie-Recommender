#importing all the libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask,request,jsonify
import json


#loading the dataset
data = pd.read_csv("movies.csv")

#splitting the required fields
df = data[['id','overview','title','genres','keywords','tagline','cast','director']]

#dropping na values
df.dropna(inplace=True)


# data preprocessing
df['overview'] = df['overview'].apply(lambda x: str(x).lower().replace(","," ").replace(":"," ").replace("!", " ").replace("...", " ").replace(".", " ").replace("?", " "))

df['title'] = df['title'].apply(lambda x: str(x).lower().replace(","," ").replace(":"," ").replace("!", " ").replace("...", " ").replace(".", " ").replace("?", " "))

df['genres'] = df['genres'].apply(lambda x:str(x).lower())

df['keywords']=df['keywords'].apply(lambda x:str(x).lower())

df['tagline'] = df['tagline'].apply(lambda x: str(x).lower().replace(","," ").replace(":"," ").replace("!", " ").replace("...", " ").replace(".", " ").replace("?", " "))

df['cast'] = df['cast'].apply(lambda x: str(x).lower().replace("\u00e9a", " "))

df['director']=df['director'].apply(lambda x:str(x).lower())


#combining all the required fields
df["combined"] = df["overview"]+ " "+df["genres"]+ " "+ df["keywords"]+" "+df["tagline"]+ " "+df["cast"]+" "+df["director"]

#keeping the required fields
df_new=df.drop(columns = ['overview','genres','keywords','tagline','cast','director'])

#encode the data
vectorizer = TfidfVectorizer(stop_words='english')
encoded = vectorizer.fit_transform(df['combined']).toarray()

# #recommender function
# def recommend_movies(movie_name,top_rec):
#     movie_name = str(movie_name).lower().replace(",", " ").replace(":", " ")

#     #encode the input
#     inp_encoded = vectorizer.transform([movie_name])

#     similarity = cosine_similarity(encoded,inp_encoded)

#     #getting top rec to top indices
#     topindices = similarity.argsort()[::-1][:top_rec]
    
#     # return the recommended movies
#     x = data.iloc[topindices][["title"]]
#     print(type(x))
#     return x

# def recommend_movies(movie_name, top_rec):
#     movie_name = str(movie_name).lower().replace(",", " ").replace(":", " ")

#     # Encode the input
#     inp_encoded = vectorizer.transform([movie_name])
#     encoded_movies = vectorizer.transform(df["combined"])
#     similarity_scores = cosine_similarity(inp_encoded, encoded_movies)
#     # df["similarity"] = similarity_scores.flatten()
#     # df_sorted = df.sort_values("similarity", ascending=False)
#     # top_similar_movies = df_sorted[].head(top_rec)
#     # print(top_similar_movies)

#     top_indices = similarity_scores.argsort()[0][-top_rec:][::-1]
#     top_movies = data.iloc[top_indices]["title"] 
#     print(top_movies)




#     # similarity = cosine_similarity(encoded, inp_encoded.T)  # Transpose inp_encoded

#     # # Getting top recommendations by sorting similarity scores
#     # top_indices = similarity.argsort(axis=0)[::-1][:top_rec].flatten()

#     # # Return the recommended movies
#     # recommended_movies = df.iloc[top_indices]["title"]
#     # return recommended_movies

# def recommend_movies(movie_name,top_rec):
#     movie_name = str(movie_name).lower().replace(",", " ").replace(":", " ")
#     similarity = cosine_similarity(encoded)
#     # indices = df[df['title'] == movie_name].index[0]
#     # dist = sorted(list(enumerate(similarity[indices])),reverse=True,key = lambda x: x[1])

#     # for i in dist[:top_rec]:
#     #     print(df.iloc[i[0]].title)
#     # getting top rec to top indices
#     topindices = similarity.argsort()[:-1][:top_rec]
    
#     # return the recommended movies
#     x = df_new.iloc[topindices][["title"]]
#     # print(type(x))
#     # return x
#     print(x)

def recommend_movies(movie_name, top_rec):

    #preprocess the input data
    movie_name = str(movie_name).lower().replace(",", " ").replace(":", " ")

    #calculate the similarities
    similarity = cosine_similarity(encoded)
    
    # Find the index of the movie
    indices = df_new[df_new['title'] == movie_name].index[0]

    # Calculate the similarity scores for the given movie
    movie_scores = similarity[indices]

    # Sort the similarity scores and get the top recommendations
    top_indices = movie_scores.argsort()[::-1][:top_rec]

    # Return the recommended movies
    x = df_new.iloc[top_indices][['title']].to_dict(orient='records')
    return x

def get_recommendation():
    movie_name = input("Enter the name of the movie: ")
    top_rec = 10
    result_dict = recommend_movies(movie_name, top_rec)
    
    with open('recommended_movies.json', 'w') as file:
        json.dump(result_dict, file)
    # Return the response as JSON
    # return jsonify(results=result_dict)
get_recommendation()