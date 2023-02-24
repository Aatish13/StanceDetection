from transformers import pipeline
import pandas as pd

# Citation:
# https://towardsdatascience.com/transfer-learning-in-nlp-for-tweet-stance-classification-8ab014da8dde
# https://huggingface.co/cardiffnlp/twitter-roberta-base-stance-climate

# Caleb Panikulam

# Runs Huggingface Piple Robert Model Transformer that has been trained on climate change tweets for stance detection
def roberta_model_baseline(text_list):
    pipe = pipeline(model="cardiffnlp/twitter-roberta-base-stance-climate")
    results_df = pd.DataFrame(pipe(text_list))
    results_df["stance"] = results_df["label"].map({"favor": 1, "against": 0}) 

    return results_df["stance"].to_list()
