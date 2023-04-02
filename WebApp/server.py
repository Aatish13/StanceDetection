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
from transformers import BertTokenizer

model = tf.keras.models.load_model('./WebApp/BERT_Trained_Model.h5')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('/views/index.html')

@app.route("/predict/", methods=["POST"])
def predict():

    file = request.files['file']
    df = pd.read_csv(file)

    encoded_texts = tokenizer.batch_encode_plus(
        df['text'].values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=128,
        return_tensors='tf'
    )

    predictions = model.predict(encoded_texts['input_ids'])

    return {'predictions': predictions.tolist()}

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
    app.run("0.0.0.0", port=3000)
