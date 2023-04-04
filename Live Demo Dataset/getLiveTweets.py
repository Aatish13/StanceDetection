import pandas as pd
import snscrape.modules.twitter as sntwitter
from time import sleep
from tqdm import tqdm
import snscrape.modules.twitter as sntwitter
import pandas

# https://betterprogramming.pub/how-to-scrape-tweets-with-snscrape-90124ed006af

tweetLst = []
tweetNumber = 1000
tweetRetrieved_count = 0

# Using TwitterSearchScraper to scrape data and append tweets to list
for i, tweet in tqdm(enumerate(
    sntwitter.TwitterSearchScraper('climate change since:2023-01-01').get_items()
), total=tweetNumber):
    if tweetRetrieved_count > tweetNumber - 1:
        break
    if tweet.lang == "en" and len(tweet.rawContent) > 0:
        tweetLst.append({"ID": tweet.id, "Date": tweet.date, "Tweet": tweet.rawContent})
        tweetRetrieved_count = tweetRetrieved_count + 1

live_demo_tweets = pd.DataFrame(tweetLst)
live_demo_tweets.to_csv("./live_demo_tweets_raw.csv", index=False)
print(live_demo_tweets)

