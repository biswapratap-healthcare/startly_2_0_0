import os
import shutil
import glob
import pickle
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.engine import training
from main import get_loss, sqldb, init

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input

vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False
vgg = Model(inputs=vgg.input, outputs=vgg.output)

def add_entry(image_id, style_name, loss, score):
    sqldb.insert_training_data(image_id, style_name, loss, score)


def prepare_training_data():
    image_data = sqldb.fetch_data(table='image_data')
    image_ids = set([e[0] for e in image_data])
    training_data = sqldb.fetch_data(table='training_data')
    training_ids = set([e[0] for e in training_data])
    image_ids_to_train = image_ids - training_ids
    image_data = filter(lambda x: x[0] in image_ids_to_train, image_data)

    # get the vectors for each style
    average_vectors = sqldb.fetch_data(table='average_vector_data')
    style_vectors = dict()
    for data in average_vectors:
        style = data[0]
        vector_folder_path = data[1]
        style_vectors[style] = dict()
        for vector_file in glob.glob(os.path.join(vector_folder_path, '*.pkl')):
            f = open(vector_file, 'rb')
            vector = pickle.load(f)
            f.close()
            style_vectors[style][os.path.basename(vector_file)] = vector

    for f in image_data:
        id = f[0]
        image_vector_folder = sqldb.fetch_vector_paths(imageid=id)[0]
        image_vector = dict()
        for vector_file in glob.glob(os.path.join(image_vector_folder, '*.pkl')):
            file = open(vector_file, 'rb')
            vector = pickle.load(file)
            file.close()
            image_vector[os.path.basename(vector_file)] = vector

        f = f[1]
        
        for style in style_vectors.keys():
            losses = []
            score = []
            for vector_file in style_vectors[style].keys():
                vector_file = vector_file
                loss = get_loss(image_vector[vector_file], style_vectors[style][vector_file]).numpy()
                losses.append(loss)
                score.append(random.randint(0, 10))
            add_entry(id, style, str(losses), str(score))

def get_training_model():
    # define two sets of inputs
    input_image_features = Input(shape=(131072,))
    input_style_vector = Input(shape=(512*512,))
    input_similarity_percentage = Input(shape=(8,))

    # the first branch operates on the first input (feature vector of image)
    a = Dense(32, activation="relu")(input_image_features)
    a = Dense(16, activation="relu")(a)
    a = Model(inputs=input_image_features, outputs=a)

    # the second branch operates on the second input (average vector for a style)
    b = Dense(32, activation="relu")(input_style_vector)
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
    z = Dense(8, activation="linear")(z)
    z = BatchNormalization()(z)
    # our model will accept the inputs of the two branches and then output a single value
    model = Model(inputs=[a.input, b.input, c.input], outputs=z)

    opt = Adam(lr=0.0001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

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

def get_input1(image_id):
    imag_path = sqldb.fetch_image_paths(imageid=image_id)[0]
    features_path = os.path.join('assets', 'feature_data')
    style_dir = os.path.join(features_path, os.path.basename(os.path.dirname(imag_path)))
    filename = os.path.basename(imag_path).split('.')[0] + '.npy'
    filepath = os.path.join(style_dir, filename)
    if os.path.exists(filepath):
        return np.load(filepath)
    else:
        os.makedirs(features_path, exist_ok=True)
        os.makedirs(style_dir, exist_ok=True)

        img = image.load_img(imag_path, target_size=(512, 512))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        img = vgg.predict(img)
        img = img[0]
        np.save(filepath, img)
        return img

def get_input2(average_vectors, style):
    for e in average_vectors:
        if e[0] == style:
            style_vector_folder = e[1]
            style_vector_file = os.path.join(style_vector_folder, 'avergae.pkl')
            style_vector_file = open(style_vector_file, 'rb')
            style_vector = pickle.load(style_vector_file)
            style_vector_file.close()
            break
    return style_vector.numpy()

def train_model():
    model = get_training_model()
    x1 = list()
    x2 = list()
    x3 = list()
    y = list()
    td = sqldb.fetch_data(table='training_data')
    print(len(td))
    average_vectors = sqldb.fetch_data(table='average_vector_data')
    i=0
    # while i<len(td):
    while i<10:
        x1.append(get_input1(td[i][0]))
        x2.append(get_input2(average_vectors, td[i][1]))
        x3.append(np.array(eval(td[i][2])))
        y.append(np.array(eval(td[i][3])))
        i += 1
    
    x1 = np.array(x1)
    x2 = np.array(x2)
    x3 = np.array(x3)
    y = np.array(y)
    x1 = x1.reshape(x1.shape[0], -1)
    model.fit(x=[x1.reshape(x1.shape[0], -1), x2.reshape(x2.shape[0], -1), x3],
              y=y,
              batch_size=10,
              epochs=50,
              validation_split=0.33,
              verbose=1)
    model_dir = os.path.join('assets', 'model')
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    model.save(model_dir)
    # sqldb.drop_table('training_data')


if __name__ == "__main__":
    # init(sqldb)
    # prepare_training_data()
    train_model()
