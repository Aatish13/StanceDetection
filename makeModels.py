from sklearn.model_selection import train_test_split
import pandas as pd


# Citation:
# https://www.geeksforgeeks.org/how-to-do-train-test-split-using-sklearn-in-python/

# Split Data into train and test to be used for the models
def splitData(twitter_df):
    x = twitter_df["Tweet"]
    y = twitter_df["stance"]
    return train_test_split(x,y,random_state=0, test_size=0.20, shuffle=True)

# Caleb's Model
def transferLearning(x_train, x_test, y_train, y_test):
    pass

# Main Code to call other functions
path = "./Cleaned_Climate_Change_Tweets_Model.csv"
twitter_df = pd.read_csv(path).dropna()
x_train, x_test, y_train, y_test = splitData(twitter_df)