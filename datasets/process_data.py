import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Read the messages & categories.
    Remove the duplicates in each dataframe.
    Merge the two datasets.

    Args:
    messages_filepath {str}: path to the messages dataframe
    categories_filepath {str}: path to categories dataframe

    Returns:
    df {dataframe}: merged dataframe 
    it has both messages & the categories together
    """
    dfmsg = pd.read_csv(messages_filepath)
    dfcats = pd.read_csv(categories_filepath)

    # Remove the duplicate rows
    dfmsg = dfmsg.drop_duplicates()
    dfcats = dfcats.drop_duplicates()
    # Merge the two dataframes
    df = dfmsg.merge(dfcats, how= 'left', on= 'id')


def clean_data(df):
    """ Clean the dataframe by:
    1. Create 36 categories from the string `categories` column
    2. Make sure the categories are binary and numeric
    3. Only keep the `message` and their corresponding categories
    4. Save the dataframe into database

    Args:
    df {dataframe}: the main dataframe with messages & categories

    Returns:
    df {dataframe}: cleaned dataframe
    """
    # Split the `categories` column into 36 different categories:
    dflabels = df['categories'].str.split(';', expand= True)
    # Get the category names, and set them as columns' name
    labels = [x.split('-')[0] for x in dflabels.loc[0, ].values.tolist()]
    dflabels.columns = labels

    # Convert the values for each category/label to binary 
    ## 1: if belongs to that category, 0: otherwise
    for col in dflabels.columns:
        dflabels[col] = dflabels[col].str.get(-1)

    # Merge these newly created categories with the original dataset,
    ## Since the `categories` column - from which the labels/categories-
    ## have been created, is not needed anymore, drop it from the dataset
    ## Also drop the column `original`, since we use the `message` only.
    df = pd.concat([df, dflabels.reindex(df.index)], axis=1)
    df = df.drop(['categories', 'original'], axis= 1)

    # Convert the category/label columns to numeric-binary
    df[labels] = df[labels].apply(pd.to_numeric, axis = 1)

    # Category `related` has extra value `2` meaning: indirectly related
    # For the requirement of this project, I'll convert the 2s as 1s.
    df.loc[df['related'] == 2, 'related'] = 1
    
    return df


def save_data(df, database_filename='DisasterResponseDB'):
    """Save the clean dataset into an sqlite database."""
    engine = create_engine(f'sqlite:///{database_filename}.db')
    df.to_sql('DisasterResponse', engine, index=False)


def main():
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
