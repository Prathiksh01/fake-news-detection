import tweepy
import pandas as pd

# Function to fetch tweets based on a keyword
def fetch_tweets(api_key, api_key_secret, access_token, access_token_secret, keyword):
    # Authenticate to Twitter
    auth = tweepy.OAuth1UserHandler(api_key, api_key_secret, access_token, access_token_secret)
    api = tweepy.API(auth)

    # Fetch tweets containing the keyword
    tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang='en').items(10)  # Change the number as needed

    # Create a DataFrame to store the tweets
    tweet_data = []
    for tweet in tweets:
        tweet_data.append({'text': tweet.text, 'created_at': tweet.created_at})
    
    return pd.DataFrame(tweet_data)
