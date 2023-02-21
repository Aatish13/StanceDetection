import pandas as pd
import snscrape.modules.twitter as sntwitter
from time import sleep

# Citations:
# https://github.com/JustAnotherArchivist/snscrape/issues/137
# https://www.freecodecamp.org/news/python-web-scraping-tutorial/
# https://medium.com/machine-learning-mastery/how-to-scrape-millions-of-tweets-using-snscraper-aa47cee400ec
# https://towardsdatascience.com/learn-how-to-easily-hydrate-tweets-a0f393ed340e
# https://betterprogramming.pub/how-to-scrape-tweets-with-snscrape-90124ed006af

# Note: Tried Twitter API, but found out there is application, which I applied for, 
# but takes at least a week. When researching how to do it, this snscrape module
# popped up claiming to be better than the API based on the articles as the 
# API has limitations with number of tweets, so I used it.

def getTweet(tweetID):
    for tweet in sntwitter.TwitterTweetScraper(tweetId=str(tweetID)).get_items():
        return tweet.rawContent 
    sleep(5)

def exportDF(df):
    df["Tweet"] = df["id"].apply(getTweet)
    sleep(1)
    df.to_csv('Cleaned_Climate_Change_Twitter_Posts.csv', mode='a', index=False)
    sleep(1)

df = pd.read_csv(".\Datasets\Climate Change Twitter Dataset.csv")
df = df[["id", "Tweet", "stance"]]
t_num = 0
for i in range(0, len(df), 100000):
    df["Tweets"][t_num:i] = df["id"][t_num].apply(getTweet)
    t_num = i
    sleep(300)

