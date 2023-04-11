import pandas as pd
from helper import *

def balance_data(df):
    df_believer = df[df["stance"]=='believer']
    df_denier = df[df["stance"]=='denier']
    df_neutral = df[df["stance"]=='neutral']
    min_count = min(len(df_neutral), len(df_believer), len(df_denier))
    return pd.concat([df_believer.sample(min_count), df_denier.sample(min_count), df_neutral.sample(min_count)])


def convert_labels_to_values(df):
  mapper = [("denier", 0), ("believer", 1), ("neutral", 2)]
  for key, value in mapper:
    df=df.replace(key,value)

  return df

if __name__ == "__main__":
    df = pd.read_csv('../Dataset/Merged_Dataset.csv')
    df = balance_data(df)

    df['Tweet'] = df['Tweet'].apply(convert_to_lower)
    df['Tweet'] = df['Tweet'].apply(clean_ascii)
    df['Tweet'] = df['Tweet'].apply(remove_mentions)
    df['Tweet'] = df['Tweet'].apply(remove_links)
    df['Tweet'] = df['Tweet'].apply(remove_punctuation)
    df['Tweet'] = df['Tweet'].apply(remove_stopwords)
    df['Tweet'] = df['Tweet'].apply(open_contractions)

    df.to_csv("../Dataset/Preprocessed_Data.csv")