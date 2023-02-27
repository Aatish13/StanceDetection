import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Citation:
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

import Bert_Model_Baseline
import Bert_Model

# Found out when doing transfer learning that the classes should be equal
def balance_data(df):
    df_b = df[df["stance"]=='believer']
    df_d = df[df["stance"]=='denier']    
    df_balanced = pd.DataFrame()
    if len(df_b) > len(df_d):
        df_b_dwnsmpld = df_b.sample(df_d.shape[0])
        df_balanced = pd.concat([df_b_dwnsmpld, df_d])
    elif len(df_b) < len(df_d):
        df_d_dwnsmpld = df_d.sample(df_b.shape[0])
        df_balanced = pd.concat([df_d_dwnsmpld, df_b])
    else:
        return df
    return df_balanced

# Based on some research, it looks like f1 is the best score for binary 
# classification
def analyze_results(test_labels, prediction_labels):
    return f1_score(test_labels, prediction_labels, average="binary")

# Main function call the other scripts
def main():
    # Import Model Dataset where Model is 99% of all the tweets collected
    df = pd.read_csv("./Climate_Change_Tweets_Model.csv")
    # Most examples with Bert classification examples so only 0, 1 (not sure how we can change it)
    df = df[df["stance"] != "neutral"]
    df = balance_data(df)
    df["stance_label"] = df["stance"].map({"believer": 1, "denier": 0})

    # Extract X: Tweets and Y: Stance_Label (0,1)
    model_texts = df["Tweet"].to_list()
    model_labels = df["stance_label"].to_list()

    # Split the train and test data, and use stratify to match the new balanced stances
    train_texts, test_texts, train_labels, test_labels = train_test_split(model_texts,model_labels, stratify=model_labels, test_size=.2)

    # Runs Huggingface Piple Robert Model Transformer that has been trained on climate change tweets for stance detection
    prediction_labels = Bert_Model_Baseline.roberta_model_baseline(test_texts)
    score = analyze_results(test_labels, prediction_labels)

    # Runs the Transformer Bert from Tensoflow hub and trains the model to do stance detection
    prediction_labels = Bert_Model.robert_model_tf(train_texts, test_texts, train_labels, test_labels)
    score = analyze_results(test_labels, prediction_labels)



main()
