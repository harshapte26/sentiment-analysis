import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from reviews_analysis import *

# Streamlit app
st.title('Reviews Analysis')

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Load and preprocess data
    df = pd.read_csv(uploaded_file)
    age_bins = [0, 18, 25, 35, 45, 55, 65, 100]
    labels = ['0-18', '18-25', '25-35', '35-45', '45-55', '55-65', '65+']
    preprocessed_df = preprocess_df(df, rating_threshold=3, age_bins=age_bins, labels=labels)

    # Sidebar for navigation
    st.sidebar.title('Navigation')
    options = ['Frequency of Ratings-Reviews', 'Age Distribution of Reviews', 'Sentiment Count', 'Review Length Analysis', "What's working and What's not working(Pain Points)?"]
    choice = st.sidebar.radio('Go to', options)

    if choice == 'Frequency of Ratings-Reviews':
        st.header('Frequency of Each Rating')
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(preprocessed_df['Rating'].value_counts().index, preprocessed_df['Rating'].value_counts().values)
        ax.set_xlabel('Rating')
        ax.set_ylabel('Frequency')
        ax.set_title('Frequency of Each Rating')
        st.pyplot(fig)

    elif choice == 'Age Distribution of Reviews':
        st.header('Age Distribution Plot')
        age_distribution = age_distribution_plot(preprocessed_df)

        fig, ax = plt.subplots()
        ax.bar(age_distribution.index, age_distribution.values)
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Count')
        ax.set_title('Age Distribution Plot')
        st.pyplot(fig)

    elif choice == 'Sentiment Count':
        st.header('Sentiment Count Plot')
        rating_threshold = 3  # Example threshold

        # Add a dropdown for column selection
        column = st.selectbox('Select Column', ['Type', 'Age_Group'])

        positive_reviews_cnt, negative_reviews_cnt = sentiment_count_plot(preprocessed_df, rating_threshold, column)

        positive_reviews_df = pd.DataFrame({
            column: positive_reviews_cnt.index,
            'Positive Reviews': positive_reviews_cnt.values
        })
        negative_reviews_df = pd.DataFrame({
            column: negative_reviews_cnt.index,
            'Negative Reviews': negative_reviews_cnt.values
        })

        # Merge the dataframes on the selected column
        merged_df = pd.merge(positive_reviews_df, negative_reviews_df, on=column)

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(merged_df[column]))  # the label locations
        width = 0.35  # the width of the bars

        positive_bars = ax.bar(x - width/2, merged_df['Positive Reviews'], width, label='Positive Reviews')
        negative_bars = ax.bar(x + width/2, merged_df['Negative Reviews'], width, label='Negative Reviews')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel(column)
        ax.set_ylabel('Count')
        ax.set_title('Sentiment Count Plot')
        ax.set_xticks(x)
        ax.set_xticklabels(merged_df[column], rotation=45)
        ax.legend()

        fig.tight_layout()
        st.pyplot(fig)

    elif choice == 'Review Length Analysis':
        st.header("Review Length Analysis")
        rating_threshold = 3  # Example threshold
        positive_reviews, negative_reviews = review_length_plot(preprocessed_df, rating_threshold)
        st.subheader('Average Review Length by Age Group and Sentiment Type')
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(positive_reviews['Age_Group']))
        width = 0.35
        positive_bars = ax.bar(x - width/2, positive_reviews['Review_Length'], width, label='Positive')
        negative_bars = ax.bar(x + width/2, negative_reviews['Review_Length'], width, label='Negative')
        ax.set_xlabel('Age Group', fontsize=12)
        ax.set_ylabel('Average Review Length', fontsize=12)
        ax.set_title('Average Review Length by Age Group and Sentiment Type', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(positive_reviews['Age_Group'], rotation=45)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

    elif choice == "What's working and What's not working(Pain Points)?":
        st.header('Top Features by Category and Sentiment')
        category_col = 'Type'  # Example category column
        text_col = 'Combined_Title_Review'  # Example text column
        sentiment_col = 'sentiment'  # Example sentiment column
        top_features = get_top_features_by_category_sentiment(preprocessed_df, category_col, text_col, sentiment_col)
        categories = list(set([category for category, _ in top_features.keys()]))
        sentiments = list(set([sentiment for _, sentiment in top_features.keys()]))
        selected_category = st.selectbox('Select Category', categories)
        selected_sentiment = st.selectbox('Select Sentiment', sentiments)
        features = top_features.get((selected_category, selected_sentiment), [])
        if len(features) != 0:
            st.subheader(f'Top Features for {selected_category} - {selected_sentiment}')
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(features))
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.write('No features available for the selected category and sentiment.')

else:
    st.write("Please upload a CSV file to proceed.")