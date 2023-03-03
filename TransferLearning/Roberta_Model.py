# Citation:
# https://www.analyticsvidhya.com/blog/2021/12/multiclass-classification-using-transformers/

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification, AutoTokenizer,TFBertModel, TFAutoModel, TFAutoModelForSequenceClassification
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense

# Caleb Panikulam

def balance_data(df):
    # Balance Dataframe
    stance_count, stance_id = df.groupby("stance_map").count().reset_index(0).sort_values("stance").head(1)[["stance", "stance_map"]].values.tolist()[0]
    ids = df["stance_map"].unique().tolist()
    ids.remove(stance_id)
    new_df = df[df["stance_map"]==stance_id]
    for id in ids:
        temp_df = df[df["stance_map"]==id].sample(stance_count)
        new_df = pd.concat([new_df, temp_df])
    
    return new_df

def preprocess_data(file_name = "./Climate_Change_Tweets_Model.csv", md_split):
    # Import Data
    df = pd.read_csv(file_name)
    df["stance_map"] = df["stance"].map({"believer": 2, "denier": 1, "neutral": 0})

    df = balance_data(df)
    
    #md_split = len(df) - 10
    model_tweets = df[0:md_split]["Tweet"].to_list()
    #demo_tweets = df[md_split:]["Tweet"].to_list()

    # Keras One-Hot Encoding Equilavent
    model_labels = tf.keras.utils.to_categorical(df[0:md_split]["stance_map"])
    #demo_labels = tf.keras.utils.to_categorical(df[md_split:]["stance_map"])

    x_train, x_test, y_train, y_test = train_test_split(model_tweets, model_labels, test_size=.2)
    return x_train, x_test, y_train, y_test

def get_BertModel(model_name):
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    # bert = TFBertModel.from_pretrained("bert-base-cased")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert = TFAutoModel.from_pretrained(model_name)

    return tokenizer, bert

def encode_data(tokenizer, data, max_length = 70):

    return tokenizer(
            text=data,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            #padding=True, 
            padding='max_length',
            return_tensors='tf',
            return_token_type_ids = False,
            return_attention_mask = True,
            verbose = True)


def create_model(tokenizer, bert, max_length = 70):
    tf.keras.backend.clear_session()

    device_type = '/CPU:0'
    if len(tf.config.list_physical_devices('GPU')) > 0: 
        device_type = '/GPU:0'

    # tokenizer, bert = get_BertModel("roberta-base")
    x_train, x_test, y_train, y_test = preprocess_data()
    x_train_token = encode_data(tokenizer, x_train) 
    x_test_token = encode_data(tokenizer, x_test)

    with tf.device(device_type):
        input_ids = Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
        input_mask = Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")
        embeddings = bert(input_ids,attention_mask = input_mask)[0] 
        out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
        out = Dense(128, activation='relu')(out)
        out = tf.keras.layers.Dropout(0.1)(out)
        out = Dense(32,activation = 'relu')(out)
        y = Dense(3,activation = 'sigmoid')(out)
        model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
        model.layers[2].trainable = True
        optimizer = Adam(
            learning_rate=5e-05, # this learning rate is for bert model , taken from huggingface website 
            epsilon=1e-08,
            decay=0.01,
            clipnorm=1.0)
        # Set loss and metrics
        loss =CategoricalCrossentropy(from_logits = True)
        metric = CategoricalAccuracy('balanced_accuracy'),
        # Compile the model
        model.compile(
            optimizer = optimizer,
            loss = loss, 
            metrics = metric)
        
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

        train_history = model.fit(
            x ={'input_ids':x_train_token['input_ids'],'attention_mask':x_train_token['attention_mask']} ,
            y = y_train,
            validation_data = (
            {'input_ids':x_test_token['input_ids'],'attention_mask':x_test_token['attention_mask']}, y_test
            ),
        epochs=100,
            batch_size=36,
            callbacks=[callback]
        )
    return model

def analyze_model(tokenizer, md_split, model,df):
    
    demo_tweets = df[md_split:]["Tweet"].to_list()
    demo_labels = tf.keras.utils.to_categorical(df[md_split:]["stance_map"])
    demo_token = encode_data(tokenizer, demo_tweets)
    
    results = model.predict({'input_ids':demo_token['input_ids'],'attention_mask':demo_token['attention_mask']})
    # print(predicted_raw[0])
    
    y_predict = np.argmax(results, axis = 1)
    y_actual = df[md_split:]["stance_map"]

    print(classification_report(y_actual, y_predict))




def predict_model(tokenizer, model, tweets):
    tweets_encoded = encode_data(tokenizer, tweets)
    
    results = model.predict({'input_ids':tweets_encoded['input_ids'],'attention_mask':tweets_encoded['attention_mask']})
    
    return np.argmax(results, axis = 1)




def test_funcs():



