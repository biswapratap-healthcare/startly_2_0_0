import os
import glob
import pickle
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras import models
from tensorflow.python.keras.preprocessing import image as kp_image
import sys
sys.path.extend(['/Users/divy/Desktop/Divy/Image Clasification(Biswa)/startly_2_0_0-cli'])
from src.sql import SqlDatabase

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
        style_dict[style_name] = average_dict

    for sk in style_dict.keys():
        f_path = os.path.join(average_vector_folder, sk)
        os.makedirs(f_path, exist_ok=True)
        for k in style_dict[sk].keys():
            file_path = os.path.join(f_path, k)
            if os.path.exists(file_path):
                os.remove(file_path)
            file = open(file_path, "xb")
            pickle.dump(average_dict[k], file)
            file.close()
        sqldb.insert_average_vector_data(sk, f_path)


def init(sqldb):
    model = get_model()
    for layer in model.layers:
        layer.trainable = False
    errors = list()
    current_paths = [e[1] for e in sqldb.fetch_data()]
    for f in glob.glob('assets//data/**/*.*', recursive=True):
        if f in current_paths:
            continue
        sqldb.insert_images(f, os.path.basename(os.path.dirname(f)))
    image_table = sqldb.fetch_data(table="image_data")
    vector_data = sqldb.fetch_data(table="vector_data")
    image_ids = set([e[0] for e in vector_data])
    image_ids = set([e[0] for e in image_table]) - image_ids
    image_table = filter(lambda x: x[0] in image_ids, image_table)

    for image_data in image_table:
        f = image_data[1]
        f_replace = f.replace('assets//data', 'assets//data_g').split('.')[0]
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
                pad_size = int((512 - c.shape[0])/2)
                paddings = tf.constant([[pad_size, pad_size, ], [pad_size, pad_size]])
                average += tf.pad(c, paddings, 'CONSTANT', constant_values=0)
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
            errors.append([e, f])
            print(e)
    generate_average_vectors(sqldb)


if __name__ == "__main__":
    init(sqldb)
