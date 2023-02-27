import pandas as pd
import snscrape.modules.twitter as sntwitter
from time import sleep
from tqdm import tqdm

# Citations:
# https://github.com/JustAnotherArchivist/snscrape/issues/137
# https://www.freecodecamp.org/news/python-web-scraping-tutorial/
# https://medium.com/machine-learning-mastery/how-to-scrape-millions-of-tweets-using-snscraper-aa47cee400ec
# https://towardsdatascience.com/learn-how-to-easily-hydrate-tweets-a0f393ed340e
# https://betterprogramming.pub/how-to-scrape-tweets-with-snscrape-90124ed006af
# https://cloud.google.com/vertex-ai/docs/text-data/sentiment-analysis/prepare-data
# https://github.com/JustAnotherArchivist/snscrape/issues/634
# https://github.com/JustAnotherArchivist/snscrape/issues/291
# https://stackoverflow.com/questions/73994971/how-do-i-filter-english-tweets-only-in-snscrape
# https://www.geeksforgeeks.org/delete-duplicates-in-a-pandas-dataframe-based-on-two-columns/

# Note: Tried Twitter API, but found out there is application, which I applied for,
# but takes at least a week. When researching how to do it, this snscrape module
# popped up claiming to be better than the API based on the articles as the
# API has limitations with number of tweets, so I used it.

# Database link:
# https://www.kaggle.com/datasets/deffro/the-climate-change-twitter-dataset?resource=download

counter = 1


def get_specific_tweet(tweet_id):
    global counter
    print("Analysing Tweet #:" + str(counter), end="\r")
    counter = counter + 1
    try:
        for i, tweet in enumerate(
            sntwitter.TwitterTweetScraper(tweetId=tweet_id, mode=sntwitter.TwitterTweetScraperMode.SINGLE).get_items()
        ):
            if tweet.lang == "en" and len(tweet.rawContent) > 0:
                return tweet.rawContent
            else:
                return "Error!!!***"
    except:
        return "Error!!!***"


print("Reading CSV File")
df = pd.read_csv(".\Datasets\Climate Change Twitter Dataset.csv")
df = df[["id", "stance"]]  # .tail(100000)

start = len(df) - 125000
end = len(df) - 100000
df = df[start:end]

print("Getting Tweets")
df["Tweet"] = df["id"].apply(get_specific_tweet)

print("Preparing Dataframe and Exporting {0} Results".format(len(df)))
df["id"] = df["id"].astype("str")
df = df[df["Tweet"] != "Error!!!***"]
df = df.dropna().drop_duplicates(subset="Tweet", keep="last").reset_index()
df.to_csv("Cleaned_Climate_Change_Tweets.csv", mode="a", index=False, header=False)
