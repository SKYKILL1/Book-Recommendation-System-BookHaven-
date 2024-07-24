import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
books = pd.read_csv('./Dataset/books.csv')
books_new = pd.read_csv('./Dataset/books_new.csv')

# Merging both the datasets
df = pd.concat([books, books_new])

# Dropping duplicate rows if any
df.drop_duplicates(inplace=True)

# Filling Null Values with 'Unknown' Value
df['SubGenre'] = df['SubGenre'].fillna('Unknown')
df['Author'] = df['Author'].fillna('Unknown Author')
df['Publisher'] = df['Publisher'].fillna('Unknown Publisher')

# Copying data into another variable to try some ideas
df1 = df.copy()

# Label encoding the categorical fields
labelencoders = {}
categorical_columns = ['Title', 'Author', 'Genre', 'Publisher', 'SubGenre']

for column in categorical_columns:
    labelencoders[column] = LabelEncoder()
    df1[column] = labelencoders[column].fit_transform(df1[column])

# Creating different variables to distinguish null valued subgenre rows and non-null value rows
books_unknown = df1[df1['SubGenre'] == 0]
books_known = df1[df1['SubGenre'] != 0]

# Label encoding subgenre column in known subgenre variable
subgenre_encoder = LabelEncoder()
subgenre_encoded = subgenre_encoder.fit_transform(books_known['SubGenre'])

# Preparing training and testing data for training and testing
X = books_known.drop(columns=['SubGenre'])
y = books_known['SubGenre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Training the model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Testing the model
X_unknown = books_unknown.drop(columns=['SubGenre'])
books_unknown.loc[:, 'SubGenre'] = subgenre_encoder.inverse_transform(clf.predict(X_unknown))

# Combined the datasets back
df2 = pd.concat([books_known, books_unknown])

# Convert back the categorical features to original labels
for column in categorical_columns:
    df2[column] = labelencoders[column].inverse_transform(df2[column])

df2['GenreCombined'] = df2['Genre'] + ',' + df2['SubGenre']

# Again copying the dataset
df3 = df2.copy()

# Combining Genre and SubGenre columns
df3['GenreCombined'].replace(r',Unknown', r'', regex=True, inplace=True)

# Dropping Genre and SubGenre columns because they are not needed anymore
df3.drop(columns=['Genre', 'SubGenre'], inplace=True)

# Used TF-IDF Vectorizer to convert genre_combined into a matrix of token counts
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df3['GenreCombined'])

# Finding similarity between all books based on Genre
cosine_sim_genre = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Encode the Author's name
label_encoder = LabelEncoder()
df3['Author_encoded'] = label_encoder.fit_transform(df3['Author'].astype(str))

# Use TF-IDF Vectorizer to convert GenreCombined into a matrix of token counts
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df3['GenreCombined'])

# Reshape Author_encoded to be a 2D array (required for cosine_similarity)
Author_encoded_reshaped = df3['Author_encoded'].values.reshape(-1, 1)

# Compute cosine similarity between all books based on GenreCombined and Authors
cosine_sim_Author = cosine_similarity(Author_encoded_reshaped, Author_encoded_reshaped)

# Function to get book recommendations based on genre, title, and Author
def get_recommendations(title, input_genre=None, input_Author=None, df3=df3, cosine_sim_genre=cosine_sim_genre, cosine_sim_Author=cosine_sim_Author):
    # Check if the title exists in the dataset
    if title not in df3['Title'].values:
        print(f"The book titled '{title}' was not found in the dataset.")
        return pd.DataFrame()

    # Get the index of the book that matches the title
    idx = df3[df3['Title'] == title].index[0]

    # Initialize a list to store the final similarity scores
    final_sim_scores = []

    # Calculate similarity scores based on genre
    if input_genre:
        genre_books = df3[df3['GenreCombined'].str.contains(input_genre, case=False)]
        genre_indices = genre_books.index
        genre_sim_scores = list(enumerate(cosine_sim_genre[idx]))
        genre_sim_scores = [score for score in genre_sim_scores if score[0] in genre_indices]
        final_sim_scores.extend(genre_sim_scores)

    # Calculate similarity scores based on Author
    if input_Author:
        Author_books = df3[df3['Author'] == input_Author]
        Author_indices = Author_books.index
        Author_sim_scores = list(enumerate(cosine_sim_Author[idx]))
        Author_sim_scores = [score for score in Author_sim_scores if score[0] in Author_indices]
        final_sim_scores.extend(Author_sim_scores)

    # Calculate similarity scores based on title (same genre and Author)
    title_sim_scores = list(enumerate((cosine_sim_genre[idx] + cosine_sim_Author[idx]) / 2))
    final_sim_scores.extend(title_sim_scores)

    # Remove duplicates and sort the books based on the combined similarity scores
    final_sim_scores = list(set(final_sim_scores))
    final_sim_scores = sorted(final_sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top 6 most similar books
    sim_indices = [i[0] for i in final_sim_scores[1:7]]

    # Return the top 6 most similar books
    return df3.iloc[sim_indices]

# Example: Get recommendations for a specific book title, genre, and Author
book_title = 'Data Smart'
input_genre = 'data_science'
input_Author = 'John'

recommendations = get_recommendations(book_title,input_genre,input_Author)

recommendations= np.array(recommendations)

# def book_recommendations(book_title, input_genre=None, input_Author=None):
#     recommendations = get_recommendations(book_title, input_genre, input_Author).reset_index()
#     recommendations.drop(columns=['index','Author_encoded'],axis=1,inplace=True)
#     return recommendations
