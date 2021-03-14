from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import plotly.graph_objs as go


# load data
engine = create_engine('sqlite:///../datasets/DisasterResponse.db')
df = pd.read_sql_table('disaster_tbl', engine)


def return_figures():
    """Creates four plotly visualizations
    Returns:
        list (dict): list containing the plotly visualizations
    """
    # ======================================================================
    # 1. Display `genre` 
    # ======================================================================
    graph_one = []

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    graph_one.append(go.Bar(
        x = genre_names,
        y = genre_counts
    ))

    layout_one = dict(
        title = 'Distribution of Message Genres',
        xaxis = dict(title = 'Genre'),
        yaxis = dict(title = 'Count')
    )

    # ======================================================================
    # 2. plot categories 
    # ======================================================================
    graph_two = []
    res = df.iloc[:, 3:].apply(np.mean, axis= 0).round(2)

    graph_two.append(go.Bar(
        x = res.index.tolist(),
        y = res.values.tolist()
    ))

    layout_two = dict(
        title = 'Distribution of each category in the dataset',
        xaxis = dict(title = 'Category'),
        yaxis = dict(title = 'Frequency')
    )

    # append all charts to the figures list
    figures = []
    figures.append(dict(data = graph_two, layout = layout_two))
    figures.append(dict(data = graph_one, layout = layout_one))
    

    return figures
