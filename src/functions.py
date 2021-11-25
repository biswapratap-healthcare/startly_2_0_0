from src.sql import SqlDatabase
import os
import glob
import random
import pickle
import math
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import models
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input

vgg = VGG19(include_top=False, weights='imagenet')
vgg.trainable = False
vgg = Model(inputs=vgg.input, outputs=vgg.output)

sqldb = SqlDatabase()

content_layers = ['block5_conv2']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1',
                'block5_pool']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def insert_images(file_path, style):
    sqldb.insert_images(file_path, style)


def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def get_loss(base, target):
    g1 = base
    g2 = target
    return tf.reduce_mean(tf.square(g1 - g2))


def load_img(path_to_img):
    max_dim = 512
    img = Image.open(path_to_img)
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.ANTIALIAS)
    img = kp_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img


def load_and_process_img(path_to_img):
    img = load_img(path_to_img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img


def get_model():
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    return models.Model(vgg.input, model_outputs)


def get_feature_representations(model, path):
    preprocessed_image = load_and_process_img(path)
    outputs = model(preprocessed_image)
    style_features = [style_layer[0] for style_layer in outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in outputs[num_style_layers:]]
    return style_features, content_features


def generate_average_vectors(sqldb):
    average_vector_folder = 'assets//average_g'
    style_dict = {}
    os.makedirs(average_vector_folder, exist_ok=True)
    styles = sqldb.fetch_data(table='styles')
    for style in styles:
        average_dict = {}
        style_name = style[0]
        image_folder_list = sqldb.fetch_vector_paths(style=style_name)
        n = len(image_folder_list)
        for image_folder in image_folder_list:
            for layer_file in os.listdir(image_folder):
                if layer_file == ".DS_Store" or layer_file == "Icon\r":
                    continue
                layer = layer_file
                layer_file = os.path.join(image_folder, layer_file)
                layer_file = open(layer_file, 'rb')
                layer_matrix = pickle.load(layer_file)
                layer_file.close()
                try:
                    average_dict[layer] += layer_matrix
                except KeyError:
                    average_dict[layer] = layer_matrix

        n = tf.constant(float(n))
        for k, v in average_dict.items():
            average_dict[k] = tf.divide(v, n)

        style_dict[style_name] = average_dict

    for style_key, style_value in style_dict.items():
        f_path = os.path.join(average_vector_folder, style_key)
        os.makedirs(f_path, exist_ok=True)
        for k, v in style_value.items():
            file_path = os.path.join(f_path, k)
            if os.path.exists(file_path):
                os.remove(file_path)
            file = open(file_path, "xb")
            pickle.dump(v, file)
            file.close()
        sqldb.insert_average_vector_data(style_key, f_path)


def generate_gram_matrices():
    model = get_model()
    for layer in model.layers:
        layer.trainable = False
    image_table = sqldb.fetch_data(table="image_data")
    vector_data = sqldb.fetch_data(table="vector_data")
    image_ids = set([e[0] for e in vector_data])
    image_ids = set([e[0] for e in image_table]) - image_ids
    image_table = filter(lambda x: x[0] in image_ids, image_table)

    for image_data in image_table:
        f = image_data[1]
        style = image_data[2]
        f_folder = os.path.basename(f).split('.')[0]
        f_replace = os.path.join('assets//data_g', style, f_folder)
        os.makedirs(f_replace, exist_ok=True)
        try:
            average = tf.zeros(
                                (512, 512), dtype=tf.dtypes.float32, name=None
                            )
            style_features, content_features = get_feature_representations(model, f)
            gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
            gram_content_features = [gram_matrix(content_feature) for content_feature in content_features]
            for ln, c in zip(content_layers, gram_content_features):
                pad_size = int((512 - c.shape[0])/2)
                paddings = tf.constant([[pad_size, pad_size, ], [pad_size, pad_size]])
                average += tf.pad(c, paddings, 'CONSTANT', constant_values=0)
                pp = os.path.join(f_replace, ln + '.pkl')
                file = open(pp, "xb")
                pickle.dump(c, file)
                file.close()
            for ln, g in zip(style_layers, gram_style_features):
                pad_size = int((512 - g.shape[0])/2)
                paddings = tf.constant([[pad_size, pad_size, ], [pad_size, pad_size]])
                average += tf.pad(g, paddings, 'CONSTANT', constant_values=0)
                pp = os.path.join(f_replace, ln + '.pkl')
                file = open(pp, "xb")
                pickle.dump(g, file)
                file.close()
            average /= (num_style_layers + num_content_layers)
            pp = os.path.join(f_replace, 'average' + '.pkl')
            file = open(pp, "xb")
            pickle.dump(average, file)
            file.close()
            sqldb.insert_vector_data(image_id=image_data[0], vector_path=f_replace)
        except Exception as e:
            print(e)
            break
    generate_average_vectors(sqldb)


# Functions for random training
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

    count = 0
    for f in image_data:
        if count > 20:
            break
        count += 1
        idd = f[0]
        image_vector_folder = sqldb.fetch_vector_paths(imageid=idd)[0]
        image_vector = dict()
        for vector_file in glob.glob(os.path.join(image_vector_folder, '*.pkl')):
            file = open(vector_file, 'rb')
            vector = pickle.load(file)
            file.close()
            image_vector[os.path.basename(vector_file)] = vector

        f = f[1]

        for style in style_vectors.keys():
            losses = list()
            scores = np.random.dirichlet(np.ones(8), size=1) * 10
            scores = list(scores.flatten())
            scores = [math.ceil(score) / 10 for score in scores]
            for vector_file in style_vectors[style].keys():
                vector_file = vector_file
                loss = get_loss(image_vector[vector_file], style_vectors[style][vector_file]).numpy()
                losses.append(loss)
            add_entry(idd, style, str(losses), str(scores))


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
            style_vector_file = os.path.join(style_vector_folder, 'average.pkl')
            style_vector_file = open(style_vector_file, 'rb')
            style_vector = pickle.load(style_vector_file)
            style_vector_file.close()
            break
    return style_vector.numpy()


to_categorical = dict()
for i in range(11):
    to_categorical[i] = np.zeros(11)
    to_categorical[i][i] = 1


def to_categorical_array(x):
    i = 0
    while i < len(x):
        x[i] = to_categorical[x[i]]
        i += 1
    return x


def get_training_data():
    x1 = list()
    x2 = list()
    x3 = list()
    y = list()
    td = sqldb.fetch_data(table='training_data')
    average_vectors = sqldb.fetch_data(table='average_vector_data')
    idx = 0
    while idx < len(td):
        x1.append(get_input1(td[idx][0]))
        x2.append(get_input2(average_vectors, td[idx][1]))
        a = td[idx][2]
        b = eval(a)
        x3.append(np.array(b))
        a = td[idx][3]
        b = eval(a)
        y.append(np.array(b))
        idx += 1
    
    x1 = np.array(x1)
    x2 = np.array(x2)
    x3 = np.array(x3)
    y = np.array(y)
    x1 = x1.reshape(x1.shape[0], -1)
    x2 = x2.reshape(x2.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    return [x1, x2, x3], y
