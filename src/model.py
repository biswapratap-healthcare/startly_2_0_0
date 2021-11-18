import os


from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from check_accuracy import check_accuracy

from main import init
from functions import get_training_data, prepare_training_data, sqldb
from check_accuracy import check_accuracy

def get_training_model():
    # define two sets of inputs
    input_image_features = Input(shape=(131072,))
    input_style_vector = Input(shape=(512*512,))
    input_similarity_percentage = Input(shape=(8,))

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
    z = Dense(16, activation="relu")(combined)
    z = Dense(88, activation="sigmoid")(z)
    # our model will accept the inputs of the two branches and then output a single value
    model = Model(inputs=[a.input, b.input, c.input], outputs=z)

    opt = Adam(learning_rate=0.0001)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])

    return model


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
    model.fit(x=x,
              y=y,
              batch_size=10,
              epochs=50,
              validation_split=0.33,
              verbose=1)
    model_dir = os.path.join('assets','model')
    model.save(model_dir)
    sqldb.drop_table('training_data')


def init_model():
    init()
    prepare_training_data()
    train_model()
    mse = check_accuracy()
    print(f"Mean squared error is {mse}")


if __name__ == "__main__":
    init_model()