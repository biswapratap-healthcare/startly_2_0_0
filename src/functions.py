import datetime

from sql import SqlDatabase
import os
import glob
import random
import pickle
import math
import numpy as np
import tensorflow as tf
import io
import base64
from PIL import Image
from tensorflow.keras import models
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
image_size = (512, 512)


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


def load_img(image_arr):
    max_dim = 512
    numpy_arr = pickle.loads(image_arr)
    img = Image.fromarray(np.uint8(numpy_arr)).convert('RGB')
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.ANTIALIAS)
    img = kp_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img


def load_and_process_img(image_arr):
    img = load_img(image_arr)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img


def get_model():
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    return models.Model(vgg.input, model_outputs)


def get_feature_representations(model, image_arr):
    preprocessed_image = load_and_process_img(image_arr)
    outputs = model(preprocessed_image)
    style_features = [style_layer[0] for style_layer in outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in outputs[num_style_layers:]]
    return style_features, content_features


def generate_average_vectors(sqldb):
    style_dict = {}
    styles = sqldb.fetch_data(table='styles')
    for style in styles:
        average_dict = {}
        style_name = style[1]
        vector_list = sqldb.fetch_vector_paths(style=style_name)
        n = len(vector_list)
        for vector_dict in vector_list:
            for layer, value in vector_dict.items():
                layer_matrix = pickle.loads(value)
                try:
                    average_dict[layer] += layer_matrix
                except KeyError:
                    average_dict[layer] = layer_matrix
        n = tf.constant(float(n))
        for k, v in average_dict.items():
            average_dict[k] = tf.divide(v, n)
        print("Calculated average vectors for style: " + str(style))
        style_dict[style_name] = average_dict
        sqldb.insert_average_vector_data(style_name, average_dict)
        print("Inserted average vectors for style: " + str(style_name))


def generate_gram_matrices():
    model = get_model()
    for layer in model.layers:
        layer.trainable = False
    image_table = sqldb.fetch_image_ids(table="image_table")
    vector_table = sqldb.fetch_image_ids(table="vector_table")
    vector_ids = set([e[0] for e in vector_table])
    image_ids = set([e[0] for e in image_table]) - vector_ids
    image_table = filter(lambda x: x[0] in image_ids, image_table)

    previous_style = ''
    pickle_strings = dict()
    for image_data in image_table:
        img_data = sqldb.fetch_image_data(image_id=image_data[0])
        image_arr = img_data[0]
        style = img_data[1]
        try:
            average = tf.zeros((512, 512), dtype=tf.dtypes.float32, name=None)
            style_features, content_features = get_feature_representations(model, image_arr)
            gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
            gram_content_features = [gram_matrix(content_feature) for content_feature in content_features]
            for ln, c in zip(content_layers, gram_content_features):
                pad_size = int((512 - c.shape[0])/2)
                paddings = tf.constant([[pad_size, pad_size, ], [pad_size, pad_size]])
                average += tf.pad(c, paddings, 'CONSTANT', constant_values=0)
                pickle_strings[ln] = pickle.dumps(c)
            for ln, g in zip(style_layers, gram_style_features):
                pad_size = int((512 - g.shape[0])/2)
                paddings = tf.constant([[pad_size, pad_size, ], [pad_size, pad_size]])
                average += tf.pad(g, paddings, 'CONSTANT', constant_values=0)
                pickle_strings[ln] = pickle.dumps(g)
            average /= (num_style_layers + num_content_layers)
            pickle_strings['average'] = pickle.dumps(average)
            sqldb.insert_vector_data(image_id=image_data[0], pickle_strings=pickle_strings)
            if previous_style != style:
                print("Inserted vector data for style: " + str(style))
                previous_style = style
        except Exception as e:
            print(e)
            break
    exit(0)
    generate_average_vectors(sqldb)


# Functions for random training
def add_entry(image_id, style_name, loss, score):
    return sqldb.insert_training_data(image_id, style_name, loss, score)


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
    training_threshold = 1
    x1 = list()
    x2 = list()
    x3 = list()
    y = list()
    td = sqldb.fetch_data(table='training_data')
    if len(td) < training_threshold:
        return [], []
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


def get_images(style):
    image_data = sqldb.fetch_data(table='image_data')
    image_ids = set([e[0] for e in image_data if e[2] == style])

    training_data = sqldb.fetch_data(table='training_data')
    training_ids = set([e[0] for e in training_data])

    image_ids_to_train = image_ids - training_ids
    image_data = list(filter(lambda x: x[0] in image_ids_to_train, image_data))

    img = image_data[0][1]
    pil_img = Image.open(img)
    pil_img.thumbnail(image_size)
    return image_to_byte_array(pil_img), str(image_data[0][0])


def get_style_images(style_id, page_num=None):
    styles = sqldb.fetch_data(table='styles')
    try:
        if page_num is None:
            page_num = 0
        else:
            page_num = int(page_num) - 1
        style = styles[style_id][1]
        images = list()
        style_images = sqldb.fetch_image_paths(style=style)
        for image_ in style_images[page_num * 40: page_num * 40 + 40]:
            image_ = Image.open(image_)
            image_.thumbnail(image_size)
            images.append(image_to_byte_array(image_))
        return str(images)
    except IndexError:
        return None


def image_to_byte_array(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image.format)
    img_byte_arr = img_byte_arr.getvalue()
    ret = base64.b64encode(img_byte_arr).decode("utf-8")
    return ret