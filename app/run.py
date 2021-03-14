import json
import plotly
import pandas as pd
import joblib

from sqlalchemy import create_engine

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib

from plots import return_figures

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemma = WordNetLemmatizer()

    tokens_clean = [lemma.lemmatize(tok.lower().strip()) for tok in tokens]

    return tokens_clean

# load data
engine = create_engine('sqlite:///../datasets/DisasterResponse.db')
df = pd.read_sql_table('disaster_tbl', engine)
print('Data Done!')

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    figures = return_figures()
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(figures)]
    # Convert the plotly figures to JSON for JS in HTML template
    graphJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
