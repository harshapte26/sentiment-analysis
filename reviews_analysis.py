import re
import string
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from wordcloud import WordCloud

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    print("Reading CSV file......")
    df = pd.read_csv(file_path)
    return df

def preprocess_text(text):
    """
    Preprocess a given text by converting to lowercase, removing punctuations, emojis, brackets, and stopwords.

    Parameters:
    text (str): The text to preprocess.

    Returns:
    str: The preprocessed text.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove emojis
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002700-\U000027BF"  # other symbols
        u"\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
        u"\U00002000-\U00002BFF"  # miscellaneous symbols
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # Remove brackets and their contents
    text = re.sub(r'\[.*?\]|\{.*?\}|\(.*?\)|\<.*?\>', '', text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)

    return text

def preprocess_df(input_df, rating_threshold, age_bins, labels):
    """
    Preprocess the DataFrame by filling NaN values, merging title and review columns, 
    creating age groups, cleaning text, and applying sentiment polarity.

    Parameters:
    input_df (pd.DataFrame): The input DataFrame.
    rating_threshold (int): The threshold for determining positive and negative sentiment.
    age_bins (list): The bins for categorizing age groups.
    labels (list): The labels for the age groups.

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    df_filled = input_df.copy()

    df_filled = df_filled.fillna({'Title':' ', 'Text_Review':' ', 'Type':'Misc'})

    # Merging Title and Review Columns
    df_filled['Combined_Title_Review'] = df_filled['Title'] + ' ' + df_filled['Text_Review']
    # Create a new column 'Age_Group'
    df_filled['Age_Group'] = pd.cut(df_filled['Age'], bins=age_bins, labels=labels, right=False)

    df_filled = df_filled.drop(['Title', 'Text_Review', 'Age'], axis=1)

    # Cleaning text in Reviews Column
    df_filled['Combined_Title_Review'] = df_filled['Combined_Title_Review'].apply(preprocess_text)

    # Applying sentiment polarity based upon rating
    df_filled['sentiment'] = df_filled['Rating'].apply(lambda rating: 'Negative' if rating < rating_threshold else 'Positive')

    return df_filled

def age_distribution_plot(df):
    """
    Get the frequency of reviews by age group.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.Series: The frequency of reviews by age group.
    """
    input_df = df.copy()

    # Get the frequency of reviews by age group
    review_count_by_age_group = input_df["Age_Group"].value_counts().sort_index()

    return review_count_by_age_group

def sentiment_count_plot(df, rating_threshold, column):
    """
    Calculate positive and negative review counts for each type.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    rating_threshold (int): The threshold for determining positive and negative sentiment.
    column (str): The column to group by for sentiment counts.

    Returns:
    tuple: Two pd.Series, one for positive review counts and one for negative review counts.
    """
    input_df = df.copy()

    # Calculate positive and negative review counts for each type
    positive_reviews_cnt = input_df[input_df['Rating'] > rating_threshold][column].value_counts().sort_index()
    negative_reviews_cnt = input_df[input_df['Rating'] <= rating_threshold][column].value_counts().sort_index()

    # Align indices of positive and negative reviews
    all_types = positive_reviews_cnt.index.union(negative_reviews_cnt.index)
    positive_reviews_cnt = positive_reviews_cnt.reindex(all_types, fill_value=0)
    negative_reviews_cnt = negative_reviews_cnt.reindex(all_types, fill_value=0)

    return positive_reviews_cnt, negative_reviews_cnt

def review_length_plot(df, rating_threshold):
    """
    Calculate the average review length grouped by age group and sentiment type.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    rating_threshold (int): The threshold for determining positive and negative sentiment.

    Returns:
    tuple: Two pd.DataFrames, one for positive reviews and one for negative reviews.
    """
    input_df = df.copy()

    # Calculate review length
    input_df['Review_Length'] = input_df['Combined_Title_Review'].apply(lambda x: len(str(x).split()))

    # Determine sentiment polarity based on 'Rating' column
    input_df['Sentiment_Type'] = input_df['Rating'].apply(lambda rating: 'Negative' if rating < rating_threshold else 'Positive')

    # Group by age group and sentiment type, then calculate average review length
    avg_review_length = input_df.groupby(['Age_Group', 'Sentiment_Type'])['Review_Length'].median().reset_index()

    # Create separate data frames for positive and negative sentiment types
    positive_reviews = avg_review_length[avg_review_length['Sentiment_Type'] == 'Positive'].reset_index(drop=True)
    negative_reviews = avg_review_length[avg_review_length['Sentiment_Type'] == 'Negative'].reset_index(drop=True)

    return positive_reviews, negative_reviews

def get_top_features_by_category_sentiment(df, category_col, text_col, sentiment_col, n_features=10):
    """
    Get the top features by category and sentiment using TF-IDF.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    category_col (str): The column name for categories.
    text_col (str): The column name for text data.
    sentiment_col (str): The column name for sentiment data.
    n_features (int): The number of top features to extract.

    Returns:
    dict: A dictionary with categories and sentiments as keys and top features as values.
    """
    categories = df[category_col].unique()
    sentiments = df[sentiment_col].unique()
    top_features = defaultdict(list)
    stop_words = set(stopwords.words('english'))

    for category in categories:
        for sentiment in sentiments:
            # Filter data by category and sentiment
            category_reviews = df[(df[category_col] == category) & (df[sentiment_col] == sentiment)][text_col]

            # Check for empty category-sentiment combinations (no reviews)
            if len(category_reviews) == 0:
                top_features[(category, sentiment)] = []
                continue

            # TF-IDF Vectorizer and processing
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            X = vectorizer.fit_transform(category_reviews)

            # Get feature names and sort by TF-IDF score
            feature_array = vectorizer.get_feature_names_out()
            tfidf_sorting = X.toarray().sum(axis=0).argsort()[::-1]

            top_n = feature_array[tfidf_sorting][:n_features]
            top_features[(category, sentiment)] = top_n.tolist()  # Convert to list for dictionary

    return top_features

if __name__ == "__main__":
    file_path = 'Raw_Reviews.csv'
    df = load_data(file_path)
    age_bins = [0, 18, 25, 35, 45, 55, 65, 100]
    labels = ['0-18', '18-25', '25-35', '35-45', '45-55', '55-65', '65+']
    preprocessed_df = preprocess_df(df, rating_threshold=3, age_bins=age_bins, labels=labels)