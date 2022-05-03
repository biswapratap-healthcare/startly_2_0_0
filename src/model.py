import os
import shutil
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from functions import get_training_data, prepare_training_data, sqldb


def get_training_model():
    # define two sets of inputs
    input_image_features = Input(shape=(131072, ))
    input_style_vector = Input(shape=(512 * 512, ))
    input_similarity_percentage = Input(shape=(8, ))

    # the first branch operates on the first input (feature vector of image)
    a = Dense(128, activation="relu")(input_image_features)
    a = Dense(64, activation="relu")(a)
    a = Dense(32, activation="relu")(a)
    a = Dense(16, activation="relu")(a)
    a = Model(inputs=input_image_features, outputs=a)

    # the second branch operates on the second input (average vector for a style)
    b = Dense(128, activation="relu")(input_style_vector)
    b = Dense(64, activation="relu")(b)
    b = Dense(32, activation="relu")(b)
    b = Dense(16, activation="relu")(b)
    b = Model(inputs=input_style_vector, outputs=b)

    # the third branch operates on the third input (layer vector of the style)
    c = Dense(10, activation="relu")(input_similarity_percentage)
    c = Dense(16, activation="relu")(c)
    c = Model(inputs=input_similarity_percentage, outputs=c)

    # combine the output of the two branches
    combined = concatenate([a.output, b.output, c.output])

    # apply a FC layer and then a regression prediction on the combined outputs
    z = Dense(128, activation="relu")(combined)
    z = Dense(8)(z)
    z = Lambda(custom_layer)(z)
    # our model will accept the inputs of the two branches and then output a single value
    model = Model(inputs=[a.input, b.input, c.input], outputs=z)

    opt = Adam(learning_rate=0.0001)
    model.compile(loss="mae", optimizer=opt, metrics=['accuracy'])

    print(model.summary())

    return model


def custom_layer(tensor):
    tensor = tf.divide(
        tf.subtract(
            tensor,
            tf.reduce_min(tensor)
        ),
        tf.subtract(
            tf.reduce_max(tensor),
            tf.reduce_min(tensor)
        )
    )
    return tensor


def plot_distribution(train_data):
    x_data = list()
    y_data = list()
    for x, td in enumerate(train_data[:100]):
        try:
            y = td[2].index(1)
        except ValueError as e:
            y = 0
        x_data.append(x)
        y_data.append(y)
    import matplotlib.pyplot as plt
    plt.plot(x_data, y_data)
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.show()


def train_model():
    model = get_training_model()
    x, y = get_training_data()
    if len(x) == 0:
        return None
    try:
        hist = model.fit(x=x,
                         y=y,
                         batch_size=10,
                         epochs=20,
                         validation_split=0.30,
                         verbose=1)
    except Exception as e:
        print(e)
        return 0
    model_dir = os.path.join('assets', 'model')
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    model.save(model_dir)
    accuracy = dict()
    accuracy['training'] = hist.history['accuracy'][-1]
    accuracy['validation'] = hist.history['val_accuracy'][-1]
    # sqldb.drop_table('training_data')
    return accuracy


def init_model():
    return train_model()


if __name__ == '__main__':
    print(init_model())
