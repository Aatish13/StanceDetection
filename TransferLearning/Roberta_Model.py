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


tf.keras.backend.clear_session()

# Import Data
df = pd.read_csv("./Climate_Change_Tweets_Model.csv")
# df = df.head(int(len(df)*0.10))
df["stance_map"] = df["stance"].map({"believer": 2, "denier": 1, "neutral": 0})

# Balance Datafram
stance_count, stance_id = df.groupby("stance_map").count().reset_index(0).sort_values("stance").head(1)[["stance", "stance_map"]].values.tolist()[0]
print(stance_count, stance_id)
ids = df["stance_map"].unique().tolist()
ids.remove(stance_id)
ids
new_df = df[df["stance_map"]==stance_id]
for id in ids:
    temp_df = df[df["stance_map"]==id].sample(stance_count)
    new_df = pd.concat([new_df, temp_df])
print(len(new_df))
df = new_df


# Split between Model and Prediction DataFrames
md_split = len(df) - 10

model_tweets = df[0:md_split]["Tweet"].to_list()
demo_tweets = df[md_split:]["Tweet"].to_list()

# Keras One-Hot Encoding Equilavent
model_labels = tf.keras.utils.to_categorical(df[0:md_split]["stance_map"])
demo_labels = tf.keras.utils.to_categorical(df[md_split:]["stance_map"])

# Test Train Split for Model Data
x_train, x_test, y_train, y_test = train_test_split(model_tweets, model_labels, test_size=.2)


# Tokenizer and Pretrained for any Bert type model

# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# bert = TFBertModel.from_pretrained("bert-base-cased")

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
bert = TFAutoModel.from_pretrained("roberta-base")

# Encode the Data
x_train_token = tokenizer(
    text=x_train,
    add_special_tokens=True,
    max_length=70,
    truncation=True,
    #padding=True, 
    padding='max_length',
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

x_test_token = tokenizer(
    text=x_test,
    add_special_tokens=True,
    max_length=70,
    truncation=True,
    #padding=True, 
    padding='max_length',
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

demo_token = tokenizer(
    text=demo_tweets,
    add_special_tokens=True,
    max_length=70,
    truncation=True,
    #padding=True, 
    padding='max_length',
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)


device_type = '/CPU:0'
if len(tf.config.list_physical_devices('GPU')) > 0: 
    device_type = '/GPU:0'

with tf.device(device_type):
    max_len = 70
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
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

predicted_raw = model.predict({'input_ids':demo_token['input_ids'],'attention_mask':demo_token['attention_mask']})
predicted_raw[0]

y_predicted = np.argmax(predicted_raw, axis = 1)
y_true = df[md_split:]["stance_map"]

print(classification_report(y_true, y_predicted))


