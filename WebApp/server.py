from flask import Flask, request,render_template,Response
from PIL import Image
import base64
import os
import datetime
import pickle
from io import BytesIO
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

model = tf.keras.models.load_model('./models/BERT_Trained_Model.h5', custom_objects={"TFBertModel": TFBertModel})
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/getOutputCSV")
def getPlotCSV():
    with open("./output.csv") as fp:
        csv = fp.read()
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=output.csv"})

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'GET':
         return render_template('index.html');
    else:
        # Read uploaded file
        file = request.files['file']
        df = pd.read_csv(file)
        print(df)
        # Encode Text for the Model
        encoded_texts = tokenizer(
            text=df['Tweet'].values.tolist(),
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
        results = model.predict({"input_ids": encoded_texts["input_ids"], "attention_mask": encoded_texts["attention_mask"]}, batch_size=int(len(df['Tweet'].values)/20))
        print(results)
        
        # Convert the Prediction to Labels via the max index of each row
        print(np.argmax(results, axis=1))

        # Produce Confidence Score
        df_conf_all = pd.DataFrame({"Label": np.argmax(results, axis=1).tolist() , "Confidence Sum": np.max(results, axis=1).tolist()})
        df_conf = df_conf_all.groupby("Label").sum().reset_index()[["Label", "Confidence Sum"]]
        df_conf["Confidence Score"] = df_conf["Confidence Sum"] / df_conf_all["Confidence Sum"].sum()
        print(df_conf[["Label", "Confidence Score"]])

        predictions = np.argmax(results, axis=1)
        mapper = ["denier", "neutral","believer"]
        df["stance"] = [mapper[p] for p in predictions] 
        df.to_csv("./output.csv")
        return render_template('index.html', fileReady = 1)


@app.route("/predictText", methods=["POST","GET"])
def predictText():
    if request.method == 'GET':
         return render_template('index.html')
    else:


        text = request.form['text']
        if text=="":
            return render_template('index.html')
        data = [text]
    
    # Create the pandas DataFrame with column name is provided explicitly
        df = pd.DataFrame(data, columns=['Tweet'])
        print(df)

        # Encode Text for the Model
        encoded_texts = tokenizer(
            text=df['Tweet'].values.tolist(),
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
        results = model.predict({"input_ids": encoded_texts["input_ids"], "attention_mask": encoded_texts["attention_mask"]}, batch_size=int(len(df['Tweet'].values)/20))
        print(results)
        
        # Convert the Prediction to Labels via the max index of each row
        print(np.argmax(results, axis=1))

        # Produce Confidence Score
        df_conf_all = pd.DataFrame({"Label": np.argmax(results, axis=1).tolist() , "Confidence Sum": np.max(results, axis=1).tolist()})
        df_conf = df_conf_all.groupby("Label").sum().reset_index()[["Label", "Confidence Sum"]]
        df_conf["Confidence Score"] = df_conf["Confidence Sum"] / df_conf_all["Confidence Sum"].sum()
        print(df_conf[["Label", "Confidence Score"]])

        predictions = np.argmax(results, axis=1) 
        print(predictions[0])
        return render_template('index.html', textPredictions = predictions[0],text = text)
    


if __name__ == "__main__":
    app.run("0.0.0.0", port=3000, debug=True)
