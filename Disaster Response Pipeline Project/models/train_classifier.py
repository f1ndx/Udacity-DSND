"""
Disaster Response Pipeline Project
Summary:
    - Loads data from the SQLite database
    - Splits the dataset into training and test sets
    - Builds a text processing and machine learning pipeline
    - Trains and tunes a model using GridSearchCV
    - Outputs results on the test set
    - Exports the final model as a pickle file

Sample Script Execution:
> python train_classifier.py ../data/DisasterResponse.db classifier.pkl
Args:
    - SQLite db path (containing pre-processed data)
    - pickle file name to save ML model
"""

# import libraries
import sys
import pandas as pd
import numpy as np
import nltk
import pickle
import re
import warnings

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, fbeta_score, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

warnings.filterwarnings('ignore')
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    """
    Load Data function that loads a SQLite database into DataFrames (features and target)
    
    Args:
        database_filepath - path to SQLite db containing cleaned message data
    Output:
        X - features DataFrame
        Y - labels DataFrame
        category_names - list of category names
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize function that processes text data by removing capitalization/special characters and lemmatizing text
    
    Args:
        text - text message for processing
    Output:
        clean_tokens - list of normalized/tokenized texts
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """
    Build Model function that outputs a Scikit ML Pipeline

    Args:
        None
    Output:
        cv: GridSearch model object that transforms the data, creates the model object and finds the optimal model parameters
    """
    # Create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Specify parameters for grid search
    # This takes over an hour to run using these params
    # parameters = {
    #     'vect__min_df': [1, 5], 
    #     'tfidf__use_idf': [True, False],
    #     'clf__estimator__n_estimators': [10, 25], 
    #     'clf__estimator__min_samples_split': [2, 5, 10]}

    # Testing other params
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__min_samples_split': [2, 4]
    }

    # Create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate Model function that returns classification report for the model
    
    Args:
        model - fitted model object
        X_test - test features DataFrame
        Y_test - test labels DataFrame
        category_names - list of category names
    """
    # Predict on test data
    Y_pred = model.predict(X_test)

    # Print scores
    print(classification_report(Y_test.iloc[:,1:].values, np.array([x[1:] for x in Y_pred]), 
        target_names=category_names))

    print("**** Accuracy scores for each category *****\n")
    for i in range(36):
        print("Accuracy score for " + Y_test.columns[i], accuracy_score(Y_test.values[:,i],Y_pred[:,i]))


def save_model(model, model_filepath):
    """
    Save Model function that saves trained model as Pickle file    

    Args:
        model - fitted model object
        model_filepath - destination path for saved .pkl file
    
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()