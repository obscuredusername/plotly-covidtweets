import pandas as pd
import plotly.graph_objects as go
from collections import Counter

# Read the CSV file into a DataFrame
df = pd.read_csv("Corona_NLP_train.csv", encoding='latin1')

# Concatenate the text data
text = ' '.join(df['OriginalTweet'])

# Define stop words
stop_words = [
    "the", "and", "to", "a", "of", "in", "that", "but", "this", "what", "for", "like", 
    "was", "is", "an", "are", "can", "you", "all", "as", "why", "i", "with", "my",
    "it's", "your", "so", "not", "on", "am", "dont", "he", "she", "which", 
    "we", "us", "will", "were", "should", "would",
    "or", "about", "have", "had", "be", "me", "no", "yes", "just", "been", "much",
    "more", "their", "there", "they", "them", "don't", "every", "could", "from", 
    "after", "got", "go", "get", "been", "have", "has", "do", "does", "did", 
    "make", "made", "come", "came", "take", "took", "give", "gave", "find", 
    "found", "know", "knew", "think", "thought", "say", "said", "see", "saw", 
    "want", "wanted", "use", "used", "try", "tried", "thing", "things", "stuff", 
    "item", "items", "person", "people", "man", "woman", "child", "children", 
    "one", "ones", "place", "places", "time", "times", "day", "days", "year", 
    "years", "good", "bad", "better", "best", "big", "small", "large", "old", 
    "young", "new", "first", "last", "high", "low", "great", "little", "many", 
    "few", "some", "any", "other", "same", "different", "while", "since", 
    "until", "than", "to", "at", "by", "from", "into", "under", "over", 
    "between", "through", "before", "among", "against", "within", "without",
    "I", "he", "she", "it", "we", "they", "him", "her", "us", "them", "an", 
    "very", "really", "just", "still", "already", "again", "too", "never", 
    "always", "often", "sometimes", "now", "here", "oh", "wow", "hey", "hi", 
    "hello", "ouch", "oops", "ugh", "how","when","because","its","u","i'm", "getting"
    ,"where","even","if","up","down","going","lot","who","using","it.","in","lol","please"
    ,"doing","i've","you're","doesn't","?","me","im","can't","i'm","those","me"
    ,"didn't","didnt","well","then","our","-","his","gonna","on","me.","i’m","these","i'm","isn't",
    "kya","ki","being","also","tell"
]

# Tokenize the text into words and filter out stop words
words = [word for word in text.split() if word.lower() not in stop_words]

# Count the word frequencies
word_counts = dict(Counter(words))

# Sort the word frequencies in descending order
sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

# Extract the most common words and their frequencies
top_words = [word[0] for word in sorted_word_counts[:50]]
word_frequencies = [word[1] for word in sorted_word_counts[:50]]

# Create a bar chart using Plotly
fig = go.Figure(data=[go.Bar(x=top_words, y=word_frequencies)])

# Customize the chart layout
fig.update_layout(
    title='Word Frequencies (Excluding Stop Words)',
    xaxis_title='Words',
    yaxis_title='Frequency'
)

# Show the bar chart
fig.show()
