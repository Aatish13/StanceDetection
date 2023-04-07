from flask import Flask, request,render_template
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

model = tf.keras.models.load_model('./BERT_Trained_Model.h5', custom_objects={"TFBertModel": TFBertModel})
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():

    # Read uploaded file
    file = request.files['file']
    df = pd.read_csv(file)
    print(df)

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
    print(results)
    
    # Convert the Prediction to Labels via the max index of each row
    print(np.argmax(results, axis=1))

    # Produce Confidence Score
    df_conf_all = pd.DataFrame({"Label": np.argmax(results, axis=1).tolist() , "Confidence Sum": np.max(results, axis=1).tolist()})
    df_conf = df_conf_all.groupby("Label").sum().reset_index()[["Label", "Confidence Sum"]]
    df_conf["Confidence Score"] = df_conf["Confidence Sum"] / df_conf_all["Confidence Sum"].sum()
    print(df_conf[["Label", "Confidence Score"]])


    return {'predictions': np.argmax(results, axis=1)}#predictions.tolist()}

    # data = request.form
    # file_name = str(datetime.datetime.now())

    # decoded_img = base64.b64decode(data["image"])
    # img = Image.open(BytesIO(decoded_img))

    # file_name = file_name + ".jpg"
    # isExist = os.path.exists(data["category"])
    # if not isExist:
    #     os.makedirs(data["category"])

    # img.save(data["category"] + "/" + file_name, "jpeg")

    # status = "Image has been successfully sent to the server."
    # response = app.response_class(response=status, status=200)
    # return response

# loaded_model = pickle.load(open("./model.pkl", 'rb'))

# @app.route("/identify-image/", methods=["POST"])
# def identify_image():

#     data = request.form
#     file_name = str(datetime.datetime.now())

#     decoded_img = base64.b64decode(data["image"])
#     img = Image.open(BytesIO(decoded_img))
#     imgToSave = img
#     img = img.convert('L')
#     img = img.resize((28,28))
#     image_np = np.array(img)
#     image_np = np.where(image_np >= 90, 0.0, 255.0)
#     #plt.imshow(image_np)
#     text_image = np.array([image_np])
#     text_image = text_image.reshape(text_image.shape[0], text_image.shape[1], text_image.shape[2], 1)
#     Y_pred = loaded_model.predict(text_image)
#     Y_pred_classes = np.argmax(Y_pred,axis = 1)
#     category = str(Y_pred_classes[0])

#     file_name = file_name + ".jpg"
#     isExist = os.path.exists(category)
#     if not isExist:
#         os.makedirs(category)

#     imgToSave.save(category + "/" + file_name, "jpeg")

#     status = "Image has been successfully sent to the server."
#     response = app.response_class(response=category, status=200)
#     return response


if __name__ == "__main__":
    app.run("0.0.0.0", port=3000, debug=True)
