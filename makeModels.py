import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Citation:
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

import Bert_Model_Baseline
import Bert_Model


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

def analyze_results(test_labels, prediction_labels):
    return f1_score(test_labels, prediction_labels, average="binary")

def main():
    df = pd.read_csv("./Climate_Change_Tweets_Model.csv")
    df = df[df["stance"] != "neutral"]
    df = balance_data(df)
    df["stance_label"] = df["stance"].map({"believer": 1, "denier": 0})

    model_texts = df["Tweet"].to_list()
    model_labels = df["stance_label"].to_list()

    train_texts, test_texts, train_labels, test_labels = train_test_split(model_texts,model_labels, stratify=model_labels, test_size=.2)

    prediction_labels = Bert_Model_Baseline.roberta_model_baseline(test_texts)
    score = analyze_results(test_labels, prediction_labels)

    prediction_labels = Bert_Model.robert_model_tf(train_texts, test_texts, train_labels, test_labels)
    score = analyze_results(test_labels, prediction_labels)




