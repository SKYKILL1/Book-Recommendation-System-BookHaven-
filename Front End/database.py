import sqlite3
import pandas as pd

# Connect to the SQLite database (create it if it doesn't exist)
conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# Create a table named 'books' with appropriate columns
create_table_query = """
CREATE TABLE IF NOT EXISTS books (
    Title TEXT NOT NULL,
    Author TEXT NOT NULL,
    Height INTEGER,
    Publisher TEXT,
    Genre TEXT
);
"""
cursor.execute(create_table_query)
conn.commit()

# Read data from CSV file
books_df = pd.read_csv('Books_merged.csv')

# Insert data into the books table
for index, row in books_df.iterrows():
    insert_query = """
    INSERT INTO books (Title, Author, Height, Publisher, Genre)
    VALUES (?, ?, ?, ?, ?);
    """
    cursor.execute(insert_query, (row['Title'], row['Author'], row['Height'], row['Publisher'], row['Genre']))

conn.commit()

# Close the connection
conn.close()
