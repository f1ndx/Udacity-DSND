"""
Disaster Response Pipeline Project
Summary:
    - Loads the messages and categories datasets
    - Merges the two datasets
    - Cleans the data
    - Stores it in a SQLite database

Sample Script Execution:
> python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`
Args:
    - CSV file containing messages (disaster_messages.csv)
    - CSV file containing categories (disaster_categories.csv)
    - SQLite destination database (DisasterResponse.db)
"""

import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
 
def load_data(messages_filepath, categories_filepath):
    """
    Load Data function that loads and merges two datasets.
    
    Args:
        messages_filepath - path to messages csv file
        categories_filepath - path to categories csv file
    Output:
        df - loaded data as a pandas DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df 

def clean_data(df):
    """
    Clean Data function that splits 'categories' column into separate category columns
    
    Args:
        df - raw data DataFrame
    Outputs:
        df - cleaned data DataFrame
    """
    # Create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';', expand=True)
    # Select the first row of the categories dataframe
    firstrow = categories.iloc[0,:]
    category_colnames = firstrow.apply(lambda x:x[:-2])
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
         # Set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(np.int)
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
    """
    Save Data function
    
    Args:
        df - Clean data Pandas DataFrame
        database_filename - database file (.db) destination path
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('DisasterResponse', engine, index=False)
    pass  


def main():
    print(sys.argv)
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()