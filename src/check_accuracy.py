import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from functions import get_training_data, sqldb

def check_accuracy():
    model_dir = os.path.join('assets','model')
    model = tf.keras.models.load_model(model_dir)

    training_data = sqldb.fetch_data(table='training_data')
    y_orignal = [eval(e[3]) for e in training_data]
    x, y = get_training_data()
    y_pred = model.predict(x)
    y_pred_ = []
    for y_pred_i in y_pred:
        weights = []
        i=0
        while i < 88:
            temp = []
            for j in range(i, i+11):
                temp.append(y_pred_i[j])
            weights.append(np.argmax(np.asarray(temp)))
            i += 11
        y_pred_.append(weights)

    mse = mean_squared_error(y_orignal, y_pred_)
    return mse