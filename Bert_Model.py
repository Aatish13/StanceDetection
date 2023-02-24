import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
import numpy as np

# Citation:
# https://www.section.io/engineering-education/classification-model-using-bert-and-tensorflow/#making-predictions
# https://towardsdatascience.com/a-practical-introduction-to-early-stopping-in-machine-learning-550ac88bc8fd
# https://stackoverflow.com/questions/51857831/does-earlystopping-in-keras-save-the-best-model

# Caleb Panikulam

# Runs the Transformer Bert from Tensoflow hub and trains the model to do stance detection
def robert_model_tf(train_texts, test_texts, train_labels, test_labels):
    # Get the Bert Models
    bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

    # Make the Custom Tensorflow Bert Layer
    # "Bert layers"
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)

    # "Neural network layers"
    # Dropout is used to prevent overfitting
    l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
    l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)

    # "Use inputs and outputs to construct a final model"
    model = tf.keras.Model(inputs=[text_input], outputs = [l])

    print(model.summary())

    METRICS = [tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]

    model.compile(optimizer='adam',loss='binary_crossentropy', metrics=METRICS)

    # Early Stopping to prevent underfitting and overfitting
    # by using a lot of epochs
    callback = tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
    model.fit(train_texts, test_texts, epochs=100, callbacks=[callback])

    results = model.predict(test_texts).flatten() 
    
    return np.where(results > 0.5, 1, 0)

