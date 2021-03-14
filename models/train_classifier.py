import sys

import re
from sqlalchemy import create_engine
import pickle

import pandas as pd
import numpy as np

# Import necessary nltk modules
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# Import custom classes
from custom_transformer import GetMessageLength

# Import necessary sklearn functions and classes
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report


def load_data(db_filepath):
    """Load the data fram db
    
    Args:
    db_filepath {str}: path to the database

    Returns:
    X {dataframe}: contains the messages in full
    y {np.array}: contains the 36 categories for messages
    labels {list of str}: the category names
    """
    # Make sure the path ends with '.db'
    if db_filepath[-3:] != '.db':
        db_filepath += '.db'
    # Load data from database
    engine = create_engine(f'sqlite:///{db_filepath}')
    # print('=====================> ', engine.table_names())
    df = pd.read_sql_table('disaster_tbl', engine)
    # Organize the dataframe into separate entities for modeling
    X = df['message']
    y = df.iloc[:, 3:]
    labels = y.columns

    return X, y.values, labels


def tokenize(msg):
    """Tokenize:
    1. Nomalize the message: lowercase and remove non-alphanumeric chars
    2. Split the message into its words
    3. Filter out the stopwords (most common used words)
    4. Lemmatize: reduce each word to its dictionary format
    
    Args:
    msg {str}: a single message

    Returns:
    tokens_clean {list of string}: a list of words for that message
    """
    # Convert to lowercase, and remove non-alphanumeric characters
    msg = re.sub(r"[^a-zA-Z0-9]", " ", msg.lower())
    # Split the message to list of words
    words = word_tokenize(msg)
    
    # Remove stop words, as thye are non-informative in our case
    stop_words = stopwords.words('english')
    tokens = [word for word in words if word not in stop_words]
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Lemmatize - reduce words to their dictionary form
    ## and remove the white spaces - leading & trailing
    tokens_clean = [lemmatizer.lemmatize(token.strip()) for token in tokens]
    # Lemmatize the verbs as well
    tokens_clean = [lemmatizer.lemmatize(token, pos = 'v') for token in tokens_clean]
    
    return tokens_clean


def build_pipeline():
    """Build the final model:
    1. Create the features
    2. Instantiate the classifier

    Args: None

    Returns:
    cv {GridSearchCV}
    """
    pipe = Pipeline([
        ('features', FeatureUnion([
            
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer = tokenize,
                                         max_features = 5000,
                                         max_df = .75)),
                ('tfidf', TfidfTransformer())
            ])),

            ('count_words', GetMessageLength())
        ])),
    
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = {
        #'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        #'features__text_pipeline__tfidf__sublinear_tf': [False, True],
        #'features__text_pipeline__tfidf__norm': ['l1', 'l2'],
        'clf__estimator__n_estimators': [20, 10],
        #'clf__estimator__learning_rate': [1, .1]
    }

    # create grid search object
    cv = GridSearchCV(pipe, parameters)
    
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """Predict the categories for testset & print the scores.
    
    Args:
    model: the final pipeline
    X_test {dataframe}: has the message column
    y_test {np.array}: has 36 categories
    category_names {list of str}: the names of categories
    
    Return: None
    Simply print the report
    """
    # Get the predictions for test set
    y_test_preds = model.predict(X_test)

    # Get the scores -- precision, recall & f1score, and accuracy
    for i, label in enumerate(category_names):
        s = f'{label} {"=" * 55}'
        print(s[:55])
        y_true = y_test[:, i]
        y_pred = y_test_preds[:, i]
        print(classification_report(y_true, y_pred))


def save_model(model, model_filepath):
    """Save the final model as pickle file."""
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size= 0.2,
                                                            random_state= 42)
        
        print('Building model...')
        pipe = build_pipeline()
        
        print('Training model...')
        print('This takes a while, you could use a cup of coffee...')
        pipe.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(pipe, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(pipe, model_filepath)

        print('Trained model saved!')

    else:
        print(
            'Please provide the filepath of the disaster messages database '
            'as the first argument and the filepath of the pickle file to '
            'save the model to as the second argument. \n\nExample: python '
            'train_classifier.py ../datasets/DisasterResponse.db classifier.pkl'
        )


if __name__ == '__main__':
    main()
