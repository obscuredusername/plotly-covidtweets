import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK stopwords
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('Corona_NLP_train.csv', encoding='latin')

# Preprocess the data
stop_words = set(stopwords.words('english'))
df['cleaned_tweet'] = df['OriginalTweet'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

# Create the Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.H1('COVID-19 Tweet Analysis'),
    dcc.Tabs([
        dcc.Tab(label='Sentiment Analysis', children=[
            dcc.Graph(id='sentiment-chart')
        ]),
        dcc.Tab(label='Word Cloud', children=[
            html.Img(id='word-cloud-image')
        ])
    ])
])

# Define the callback functions
@app.callback(
    Output('sentiment-chart', 'figure'),
    [Input('sentiment-chart', 'hoverData')])
def update_sentiment_chart(hoverData):
    # Utilize the function to generate sentiment line chart
    fig = generate_sentiment_line_chart()
    return fig

@app.callback(
    Output('word-cloud-image', 'src'),
    [Input('word-cloud-image', 'id')])
def update_word_cloud(id):
    text = ' '.join(df['cleaned_tweet'])
    wordcloud = WordCloud().generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('word_cloud.png', bbox_inches='tight')
    return 'word_cloud.png'

def generate_sentiment_line_chart():
    # Read the CSV file into a DataFrame
    df = pd.read_csv("Corona_NLP_train.csv", encoding='latin1')

    # Convert the 'TweetAt' column to datetime
    df['TweetAt'] = pd.to_datetime(df['TweetAt'], format="%d-%m-%Y")

    # Group the data by date and sentiment, and count the occurrences
    sentiment_counts = df.groupby(['TweetAt', 'Sentiment']).size().reset_index(name='Count')

    # Combine negative and extremely negative sentiments
    sentiment_counts.loc[sentiment_counts['Sentiment'].isin(['Negative', 'Extremely Negative']), 'Sentiment'] = 'Negative'

    # Combine positive and extremely positive sentiments
    sentiment_counts.loc[sentiment_counts['Sentiment'].isin(['Positive', 'Extremely Positive']), 'Sentiment'] = 'Positive'

    # Create a line chart using Plotly
    fig = px.line(sentiment_counts, x='TweetAt', y='Count', color='Sentiment',
                  labels={'Count': 'Number of Tweets', 'TweetAt': 'Date'},
                  title='Number of Tweets Over Time by Sentiment')

    # Customize the chart layout
    fig.update_layout(
        legend_title='Sentiment',
        xaxis_tickformat='%b %Y'  # Format the x-axis tick labels as month and year
    )

    return fig


if __name__ == "__main__":
  generate_sentiment_line_chart()



    
