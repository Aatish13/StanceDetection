# Citation:
# https://www.analyticsvidhya.com/blog/2021/12/multiclass-classification-using-transformers/
# https://datascience.stackexchange.com/questions/92955/keras-earlystopping-callback-why-would-i-ever-set-restore-best-weights-false
# https://stackoverflow.com/questions/63851453/typeerror-singleton-array-arraytrue-cannot-be-considered-a-valid-collection
# https://stackoverflow.com/questions/41908379/keras-plot-training-validation-and-test-set-accuracy
# https://www.kdnuggets.com/2021/02/saving-loading-models-tensorflow.html
# https://stackoverflow.com/questions/73557769/valueerror-unknown-layer-tfbertmodel-please-ensure-this-object-is-passed-to-t
# https://stackoverflow.com/questions/73557769/valueerror-unknown-layer-tfbertmodel-please-ensure-this-object-is-passed-to-t
# https://wandb.ai/ayush-thakur/dl-question-bank/reports/What-s-the-Optimal-Batch-Size-to-Train-a-Neural-Network---VmlldzoyMDkyNDU
# https://datascience.stackexchange.com/questions/37186/early-stopping-on-validation-loss-or-on-accuracy
# https://stackoverflow.com/questions/67715646/model-summary-and-plot-model-showing-nothing-from-the-built-model-in-tensorf

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, TFBertModel, TFAutoModel, TFRobertaModel
import tensorflow as tf

# Caleb Panikulam

def balance_data(df):
    stance_count, stance_id = (
        df.groupby("stance_map").count().reset_index(0).sort_values("stance").head(1)[["stance", "stance_map"]].values.tolist()[0]
    )

    ids = df["stance_map"].unique().tolist()
    ids.remove(stance_id)
    new_df = df[df["stance_map"] == stance_id]
    for id in ids:
        temp_df = df[df["stance_map"] == id].sample(stance_count)
        new_df = pd.concat([new_df, temp_df])
    
    print("Database")
    print(df.groupby("stance_map").count().reset_index(0)[["stance"]])
    print("\n")
    print("Balanced Database")
    print(new_df.groupby("stance_map").count().reset_index(0)[["stance"]])
    print("\n")

    return new_df

def preprocess_data(file_name="../Dataset/Preprocessed_Data_Added_More.csv"):
    df = pd.read_csv(file_name)
    df["stance_map"] = df["stance"].map({"neutral": 0, "denier": 1, "believer": 2})

    df = balance_data(df)

    model_tweets, demo_tweets, model_labels, demo_labels = train_test_split(
        df["Tweet"].to_list(), df["stance_map"].to_list(), test_size=0.02, stratify=df["stance_map"].to_list()
    )

    # Keras One-Hot Encoding Equilavent
    model_labels = tf.keras.utils.to_categorical(model_labels)
    demo_labels = tf.keras.utils.to_categorical(demo_labels)

    return model_tweets, demo_tweets, model_labels, demo_labels

def get_BertModel(model_name = ""):
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    # bert = TFBertModel.from_pretrained("bert-base-cased")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = TFBertModel.from_pretrained("bert-base-uncased")

    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # bert = TFAutoModel.from_pretrained(model_name)

    return tokenizer, bert

def encode_data(tokenizer, data, max_length=70):
    return tokenizer(
        text=data,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        # padding=True,
        padding="max_length",
        return_tensors="tf",
        return_token_type_ids=False,
        return_attention_mask=True,
        verbose=True,
    )

def create_model(tokenizer, bert, model_tweets, model_labels, max_length=70, file="./models/BERT_Trained_Model.h5"):
    tf.keras.backend.clear_session()

    device_type = "/CPU:0"
    if len(tf.config.list_physical_devices("GPU")) > 0:
        device_type = "/GPU:0"
    print("Note: Using {0}".format(device_type))
    x_train, x_test, y_train, y_test = train_test_split(model_tweets, model_labels, test_size=0.2, stratify=model_labels)
    x_train_token = encode_data(tokenizer, x_train)
    x_test_token = encode_data(tokenizer, x_test)

    with tf.device(device_type):
        input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
        input_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")
        embeddings = bert(input_ids, attention_mask=input_mask)[0]
        out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
        out = tf.keras.layers.Dense(128, activation="relu")(out)
        out = tf.keras.layers.Dropout(0.1)(out)
        out = tf.keras.layers.Dense(32, activation="relu")(out)
        y = tf.keras.layers.Dense(3, activation="sigmoid")(out)
        model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
        model.layers[2].trainable = True
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=5e-05,
            epsilon=1e-08,
            decay=0.01,
            clipnorm=1.0,  # this learning rate is for bert model , taken from huggingface website
        )
        # Set loss and metrics
        loss = tf.keras.losses.CategoricalCrossentropy()  # from_logits=True)
        metric = tf.keras.metrics.CategoricalAccuracy("balanced_accuracy")
        # Compile the model
        model.compile(optimizer=optimizer, loss=loss, metrics=metric)

        callback = tf.keras.callbacks.EarlyStopping(monitor="val_balanced_accuracy", patience=3, restore_best_weights=True)#, min_delta=0.0001)

        train_history = model.fit(
            x={"input_ids": x_train_token["input_ids"], "attention_mask": x_train_token["attention_mask"]},
            y=y_train,
            validation_data=(
                {"input_ids": x_test_token["input_ids"], "attention_mask": x_test_token["attention_mask"]},
                y_test,
            ),
            epochs=100,
            batch_size=32,
            callbacks=[callback],
        )
    results = model.evaluate({"input_ids": x_test_token["input_ids"], "attention_mask": x_test_token["attention_mask"]}, y_test, verbose=1)    
    print("Evaluation of Model:\n{0}\n".format(dict(zip(model.metrics_names, results))))
    # pd.DataFrame(train_history.history).plot(figsize=(8,5))
    # plt.show()
    model.save(file)
    print("Saved Model: {0}".format(file))
    tf.keras.backend.clear_session()
    del model
    # return model

def import_model(file, transformer_model):
    return tf.keras.models.load_model(file, custom_objects=transformer_model)

def analyze_model(tokenizer, file, demo_tweets, demo_labels):
    # model = tf.keras.models.load_model(file, custom_objects={"TFRobertaModel": TFRobertaModel})
    model = import_model(file, {"TFBertModel": TFBertModel})
    demo_token = encode_data(tokenizer, demo_tweets)

    results = model.predict({"input_ids": demo_token["input_ids"], "attention_mask": demo_token["attention_mask"]})
    # print(predicted_raw[0])

    y_predict = np.argmax(results, axis=1)
    y_actual = np.argmax(demo_labels, axis=1).tolist()

    print(classification_report(y_actual, y_predict))

def predict_model(tokenizer, tweets, file="./models/BERT_Trained_Model.h5"):
    model = import_model(file, {"TFBertModel": TFBertModel})
    tweets_encoded = encode_data(tokenizer, tweets)

    results = model.predict({"input_ids": tweets_encoded["input_ids"], "attention_mask": tweets_encoded["attention_mask"]})

    return np.argmax(results, axis=1)

def test_funcs():
    print("Test")

    #tokenizer, bert = get_BertModel("roberta-base")
    # tokenizer, bert = get_BertModel("bert-base-cased")
    tokenizer, bert = get_BertModel()
    model_tweets, demo_tweets, model_labels, demo_labels = preprocess_data()
    print("Count - Model: {0} & Demo: {1}".format(len(model_tweets), len(demo_tweets)))

    # print("Create_Model()")
    # create_model(tokenizer, bert, model_tweets, model_labels)

    print("Analyze_Model()")
    analyze_model(tokenizer, "./models/BERT_Trained_Model.h5", demo_tweets, demo_labels)

    # print("Predict_Model()")
    # results = predict_model(tokenizer, demo_tweets[1:5])
    # print(results)


if __name__ == "__main__":
    tf.keras.backend.clear_session()
    test_funcs()
    tf.keras.backend.clear_session()


"""
Batch Size: 16, Bert Uncased, Learning Rate 5E5
{'loss': 0.5221635699272156, 'balanced_accuracy': 0.8054231405258179}
f1 score: 0.89 accuracy for 1028

Batch Size: 32, Bert Uncased, Learning Rate 5E5
{'loss': 0.5252397656440735, 'balanced_accuracy': 0.8102900385856628}
f1 score: 0.90 accuracy for 1028

Batch Size: 32, Roberta, Learning Rate 5E5
{'loss': 0.5152681469917297, 'balanced_accuracy': 0.8009535074234009
f1 score: 0.86 accuracy for 1028

"""
