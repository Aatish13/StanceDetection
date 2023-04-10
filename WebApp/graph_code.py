import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

model = tf.keras.models.load_model('./BERT_Trained_Model.h5', custom_objects={"TFBertModel": TFBertModel})
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

from sklearn.metrics import multilabel_confusion_matrix


def get_prediction(df):  
    # Encode Text for the Model
    encoded_texts = tokenizer(
        text=df['Tweets'].values.tolist(),
        add_special_tokens=True,
        max_length=70,
        truncation=True,
        # padding=True,
        padding="max_length",
        return_tensors="tf",
        return_token_type_ids=False,
        return_attention_mask=True,
        verbose=True,
    )

    # Run the Prediction
    results = model.predict({"input_ids": encoded_texts["input_ids"], "attention_mask": encoded_texts["attention_mask"]}, batch_size=int(len(df['Tweets'].values)/20))
    return np.argmax(results, axis=1).tolist()

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
# Copied from: https://stackoverflow.com/questions/67303001/plot-confusion-matrix-with-keras-data-generator-using-sklearn
def get_confusionMatrix(expected, predictions):
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(expected, predictions), display_labels=["Non-Believer: 0", "Neutral: 1", "Believer: 2"])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

# https://stackoverflow.com/questions/56870373/getting-the-accuracy-from-classification-report-back-into-a-list
def get_skLearnEvaluation(expected, predictions):
    return classification_report(expected, predictions, output_dict=True)
    """
    Example Output:
    {'0': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 1},
    '1': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 1},
    '2': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 1},
    'micro avg': {'precision': 0.6666666666666666,
    'recall': 0.6666666666666666,
    'f1-score': 0.6666666666666666,
    'support': 3},
    'macro avg': {'precision': 0.6666666666666666,
    'recall': 0.6666666666666666,
    'f1-score': 0.6666666666666666,
    'support': 3},
    'weighted avg': {'precision': 0.6666666666666666,
    'recall': 0.6666666666666666,
    'f1-score': 0.6666666666666666,
    'support': 3},
    'samples avg': {'precision': 0.75,
    'recall': 0.75,
    'f1-score': 0.6666666666666666,
    'support': 3}}
    """

df = pd.read_csv(r"../TransferLearning/demo_tweets.csv")
predictions = get_prediction(df)
expected = df["Label"].values.tolist()
get_confusionMatrix(expected, predictions)

# ==================================
#  Prediction

# https://medium.com/voice-tech-global/machine-learning-confidence-scores-all-you-need-to-know-as-a-conversation-designer-8babd39caae7
def score2Txt(score):
    # "Over 0.7: the prediction is a strong candidate for answering the user query."
    if score >= 0.7:
        return "Model is very confident in the predictions." # Progress Bar color is green

    # "Between 0.3 and 0.7: the prediction can partially answer the request." 
    if score >= 0.3 and score < 0.7:
        return "Model is almost confident in the predictions." # Progress Bar is yellow

    # "Below 0.3: the prediction is probably not a good choice."
    else:
        return "Model is not very confident in the predictions" # Progress Bar is red

def get_confidence_score(results):
    # Produce Confidence Score
    df_conf_all = pd.DataFrame({"Label": np.argmax(results, axis=1).tolist() , "Confidence Sum": np.max(results, axis=1).tolist()})
    df_conf = df_conf_all.groupby("Label").sum().reset_index()[["Label", "Confidence Sum"]]
    df_conf["Confidence Score"] = df_conf["Confidence Sum"] / df_conf_all["Confidence Sum"].sum()
    new_df = df_conf[["Label", "Confidence Score"]]
    new_df["Confidence Text"] = new_df["Confidence Text"].apply(score2Txt)
    # Progress Bar is score rounded to percentage 0.3513 => 35%
    return new_df

# Tensorflow Training vs Testing graph saved as model_results_plot.png
