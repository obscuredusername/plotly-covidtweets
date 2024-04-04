import pandas as pd
import plotly.express as px

def generate_sentiment_choropleth():
    # Load the dataset
    df = pd.read_csv("Corona_NLP_train.csv", encoding='latin1')

    # Define color mapping
    color_map = {'Positive': 'lightgreen', 'Extremely Positive': 'darkgreen', 
                 'Negative': 'orange', 'Extremely Negative': 'red', 'Neutral': 'grey'}

    # Create a choropleth map using Plotly Express
    fig = px.choropleth(df, 
                         locations="Location", 
                         locationmode="country names",
                         color="Sentiment",
                         color_discrete_map=color_map,
                         projection="natural earth",
                         title="Sentiment Choropleth Map",
                         hover_name="Location",
                         labels={"Sentiment": "Sentiment"}
                        )

    # Show the figure
    fig.show()

# Example usage
if __name__ == "__main__":
    generate_sentiment_choropleth()
