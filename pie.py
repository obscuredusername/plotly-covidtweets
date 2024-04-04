import pandas as pd
import plotly.express as px

def generate_sentiment_pie_chart():
    # Read the CSV file into a DataFrame
    df = pd.read_csv("Corona_NLP_train.csv", encoding='latin1')

    # Filter out rows with missing location data
    df = df.dropna(subset=['Location'])

    # Group the data by sentiment and count the occurrences
    sentiment_counts = df['Sentiment'].value_counts().reset_index()

    # Rename the columns
    sentiment_counts.columns = ['Sentiment', 'Count']

    # Define a color mapping for each sentiment category
    color_mapping = {
        'Positive': 'lightgreen',
        'Extremely Positive': 'darkgreen',
        'Negative': 'orange',
        'Extremely Negative': 'darkred',
        'Neutral': 'lightgrey'
    }

    # Map each sentiment category to a color
    sentiment_counts['Color'] = sentiment_counts['Sentiment'].map(color_mapping)

    # Create a pie chart using Plotly
    fig_pie = px.pie(sentiment_counts,
                     values='Count',
                     names='Sentiment',
                     color='Sentiment',
                     color_discrete_map=color_mapping,
                     title='Sentiment Distribution')

    # Show the pie chart
    fig_pie.show()

# Example usage
if __name__ == "__main__":
    generate_sentiment_pie_chart()
