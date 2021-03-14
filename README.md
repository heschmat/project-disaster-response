# Disaster Response Project

Analyze disaster data from [Figure Eight][https://appen.com/] to build a model for an API that classifies disaster messages.


# Project Components
There are three components in this project.

1. ETL Pipeline
`process_data.py` is responsible for data cleaning pipeline:

- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

2. ML Pipeline
`train_classifier.py` builds a machine learning pipeline that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. Flask Web App
The results will be displayed in a flask web app. To run the app, go to the `app` directory and run `python run.py`. Then you have to go to the following link: `http://localhost:3001/`.
```
(disaster_env) C> python run.py
Data Done!
 * Serving Flask app "run" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
 * Restarting with stat
Data Done!
 * Debugger is active!
 * Debugger PIN: 101-974-328
 * Running on http://0.0.0.0:3001/ (Press CTRL+C to quit)
```


# Project Interface
1. To run the app, simply go to the `app` directory and run `python run.py` in the command line.

2. To process a new dataset you need to go to the `datasets` directory. An example to do ETL would be `python process_data.py messages.csv categories.csv DisasterResponse.db`. 
Here, first and 2nd arguments - after the script - are the message and categories dataset. The last argument is the path to save the cleaned data into a database. 
```
(disaster_env) C> python process_data.py messages.csv categories.csv DisasterResponseDB.db
Loading data...
    MESSAGES: messages.csv
    CATEGORIES: categories.csv
Cleaning data...
Saving data...
    DATABASE: DisasterResponseDB.db
Cleaned data saved to database!
```

3. To train the model, go to the `models` directory. To train the classifier run `python train_classifier.py ../datasets/DisasterResponse.db classifier.pkl`.
Here, first argument is where the cleaned data stored in the database, and the last argument is the name with which you want to save the newly created classifier. 

# Libraries
- sqlalchemy
- pandas
- sklearn
- flask
- nltk

# Challenges
There are 36 categories in this dataset; 33 of the categories are flaged less than 20%. And 29 of them are flaged less than 10%. For this reason, `accuracy` is not a viable metric to judge the model performance. In this case, depending on the nature of the task, we may want to optimize the performance for `precision` or `recall`; or simply use `f1score` as a singular metric that takes both into account. 
