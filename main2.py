import os
import glob
import pickle
import random
import numpy as np
from main import get_model, get_feature_representations, gram_matrix, content_layers, style_layers

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop

layers = dict()
layers['block1_conv1'] = [1, 0, 0, 0, 0, 0, 0]
layers['block2_conv1'] = [0, 1, 0, 0, 0, 0, 0]
layers['block3_conv1'] = [0, 0, 1, 0, 0, 0, 0]
layers['block4_conv1'] = [0, 0, 0, 1, 0, 0, 0]
layers['block5_conv1'] = [0, 0, 0, 0, 1, 0, 0]
layers['block5_pool'] = [0, 0, 0, 0, 0, 1, 0]
layers['block5_conv2'] = [0, 0, 0, 0, 0, 0, 1]

train_data_block1_conv1_64 = list()
train_data_block2_conv1_128 = list()
train_data_block3_conv1_256 = list()
train_data_block4_conv1_512 = list()
train_data_block5_conv1_512 = list()
train_data_block5_pool_512 = list()
train_data_block5_conv2_512 = list()


def add_entry(ln, entry):
    if ln == 'block1_conv1':
        train_data_block1_conv1_64.append(entry)
    elif ln == 'block2_conv1':
        train_data_block2_conv1_128.append(entry)
    elif ln == 'block3_conv1':
        train_data_block3_conv1_256.append(entry)
    elif ln == 'block4_conv1':
        train_data_block4_conv1_512.append(entry)
    elif ln == 'block5_conv1':
        train_data_block5_conv1_512.append(entry)
    elif ln == 'block5_pool':
        train_data_block5_pool_512.append(entry)
    elif ln == 'block5_conv2':
        train_data_block5_conv2_512.append(entry)
    else:
        pass


def prepare_training_data():
    model = get_model()
    for f in glob.glob('data/**/*.*', recursive=True):
        style_features, content_features = get_feature_representations(model, f)
        gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
        gram_content_features = [gram_matrix(content_feature) for content_feature in content_features]
        for ln, c in zip(content_layers, gram_content_features):
            score = list()
            a = random.randint(0, 9)
            for pos in range(0, 10, 1):
                if pos == a:
                    score.append(1)
                else:
                    score.append(0)
            entry = [c.numpy(), layers.get(ln), score]
            add_entry(ln, entry)
        for ln, s in zip(style_layers, gram_style_features):
            score = list()
            a = random.randint(1, 10)
            for pos in range(0, 10, 1):
                if pos == a:
                    score.append(1)
                else:
                    score.append(0)
            entry = [s.numpy(), layers.get(ln), score]
            add_entry(ln, entry)
    file = open('train_data_block1_conv1_64.pkl', "xb")
    pickle.dump(train_data_block1_conv1_64, file)
    file.close()
    file = open('train_data_block2_conv1_128.pkl', "xb")
    pickle.dump(train_data_block2_conv1_128, file)
    file.close()
    file = open('train_data_block3_conv1_256.pkl', "xb")
    pickle.dump(train_data_block3_conv1_256, file)
    file.close()
    file = open('train_data_block4_conv1_512.pkl', "xb")
    pickle.dump(train_data_block4_conv1_512, file)
    file.close()
    file = open('train_data_block5_conv1_512.pkl', "xb")
    pickle.dump(train_data_block5_conv1_512, file)
    file.close()
    file = open('train_data_block5_pool_512.pkl', "xb")
    pickle.dump(train_data_block5_pool_512, file)
    file.close()
    file = open('train_data_block5_conv2_512.pkl', "xb")
    pickle.dump(train_data_block5_conv2_512, file)
    file.close()


def get_64_model():
    # define two sets of inputs
    input_a = Input(shape=(4096,))
    input_b = Input(shape=(7,))

    # the first branch operates on the first input
    x = Dense(32, activation="relu")(input_a)
    x = Dense(16, activation="relu")(x)
    x = Model(inputs=input_a, outputs=x)

    # the second branch operates on the second input
    y = Dense(32, activation="relu")(input_b)
    y = Dense(16, activation="relu")(y) 
    y = Model(inputs=input_b, outputs=y)

    # combine the output of the two branches
    combined = concatenate([x.output, y.output])

    # apply a FC layer and then a regression prediction on the combined outputs
    z = Dense(16, activation="relu")(combined)
    z = Dense(10, activation="linear")(z)

    # our model will accept the inputs of the two branches and then output a single value
    model = Model(inputs=[x.input, y.input], outputs=z)

    opt = Adam(lr=0.00001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

    return model


def train_data_block1_conv1_model():
    infile = open('train_data_block1_conv1_64.pkl', 'rb')
    train_data = pickle.load(infile, encoding='bytes')
    model_64 = get_64_model()
    x1 = list()
    x2 = list()
    y = list()
    for td in train_data:
        x1.append(td[0].flatten())
        x2.append(np.array(td[1]))
        y.append(np.array(td[2]))
    x1 = np.array(x1)
    x2 = np.array(x2)
    y = np.array(y)
    model_64.fit(x=[x1, x2],
                 y=y,
                 batch_size=50,
                 epochs=50,
                 validation_split=0.33,
                 verbose=1)


if __name__ == "__main__":
    # if not os.path.exists('training_data.pkl'):
    #    prepare_training_data()
    # exit(0)
    train_data_block1_conv1_model()
