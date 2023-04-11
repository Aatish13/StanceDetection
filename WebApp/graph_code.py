import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('./models/BERT_Trained_Model.h5', custom_objects={"TFBertModel": TFBertModel})
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
    # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(expected, predictions), display_labels=["Non-Believer: 0", "Neutral: 1", "Believer: 2"])
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(expected, predictions, normalize="all"), display_labels=["Non-Believer: 0", "Neutral: 1", "Believer: 2"])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

# https://stackoverflow.com/questions/56870373/getting-the-accuracy-from-classification-report-back-into-a-list
def get_skLearnEvaluation(expected, predictions):
    # print(classification_report(expected, predictions))
    result = classification_report(expected, predictions, output_dict=True)
    # print(result)
    for i in range(0, 3):
        print("F1 Score for Label {0}: {1}".format(i, result[str(i)]["f1-score"]*100))
    print("Model Accuracy: {0}".format(result["accuracy"]*100))
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

def model_trainVtestgraph():
    # pd.DataFrame(model.history).plot(figsize=(8,5))
    # plt.show()
    print(pd.DataFrame.from_dict(model.history))
    

# df = pd.read_csv(r"../TransferLearning/demo_tweets.csv")
# predictions = get_prediction(df)
# expected = df["Label"].values.tolist()
# get_confusionMatrix(expected, predictions)
# get_skLearnEvaluation(expected, predictions)
# Note: For the training, focused on val_balanced_accuracy
model_trainVtestgraph()

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
