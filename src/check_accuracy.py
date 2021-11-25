import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from .functions import get_training_data, sqldb

def check_accuracy():
    model_dir = os.path.join('assets','model')
    model = tf.keras.models.load_model(model_dir)

    training_data = sqldb.fetch_data(table='training_data')
    y_orignal = [eval(e[3]) for e in training_data]
    x, y = get_training_data()
    y_pred = model.predict(x)
    i=0
    while i < len(y_pred):
        y_pred[i] = y_pred[i]*10
        print(y_pred[i])
        i += 1

    mse = mean_squared_error(y_orignal, y_pred)
    return mse