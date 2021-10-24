import os
import shutil

import cv2
import glob
import pickle
import numpy as np
import tensorflow as tf
from keras.applications import vgg19

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1',
                'block5_pool']


def preprocess_image(image_path):
    width, height = tf.keras.preprocessing.image.load_img(image_path).size
    img_n_rows = 400
    img_n_cols = int(width * img_n_rows / height)
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(img_n_rows, img_n_cols)
    )
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)


def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram


def get_model():
    model = vgg19.VGG19(weights="imagenet", include_top=False)
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    style_output_dict = {}
    for s in style_layers:
        style_output_dict[s] = outputs_dict[s]
    feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=style_output_dict)
    return feature_extractor


def init():
    feature_extractor = get_model()
    errors = list()
    for f in glob.glob('data/**/*.*', recursive=True):
        f_replace = f.replace('data', 'data_g')
        os.makedirs(f_replace, exist_ok=True)
        try:
            style_reference_image = preprocess_image(f)
            features = feature_extractor(style_reference_image)
            for layer_name in style_layers:
                pp = os.path.join(f_replace, layer_name + '.pkl')
                if os.path.exists(pp):
                    continue
                layer_features = features[layer_name]
                g = gram_matrix(layer_features[0])
                file = open(pp, "xb")
                pickle.dump(g, file)
                file.close()
        except Exception as ex:
            errors.append([ex, f])


def style_loss(style, combination, img_n_rows, img_n_cols):
    s = style
    c = combination
    channels = 3
    size = img_n_rows * img_n_cols
    return tf.reduce_sum(tf.square(s - c)) / (4.0 * (channels ** 2) * (size ** 2))


def search(ref_image):
    width, height = tf.keras.preprocessing.image.load_img(ref_image).size
    img_n_rows = 400
    img_n_cols = int(width * img_n_rows / height)
    feature_extractor = get_model()
    style_reference_image = preprocess_image(ref_image)
    features = feature_extractor(style_reference_image)
    comp_dict = dict()
    for layer_name in style_layers:
        ref_layer_features = features[layer_name]
        ref_gram_matrix = gram_matrix(ref_layer_features[0])
        for f in glob.glob('data_g/**/' + layer_name + '.pkl', recursive=True):
            fn = f.replace('data_g', 'data')
            fn = os.path.dirname(fn)
            infile = open(f, 'rb')
            comparison_gram_matrix = pickle.load(infile, encoding='bytes')
            try:
                s_loss = style_loss(ref_gram_matrix, comparison_gram_matrix, img_n_rows, img_n_cols).numpy()
                comp_dict[layer_name].append([fn, s_loss])
            except KeyError:
                s_loss = style_loss(ref_gram_matrix, comparison_gram_matrix, img_n_rows, img_n_cols).numpy()
                comp_dict[layer_name] = [[fn, s_loss]]
        comp_dict[layer_name] = sorted(comp_dict[layer_name], key=lambda x: x[1])

    average = dict()
    for k in comp_dict.keys():
        for e in comp_dict[k]:
            try:
                average[e[0]] += e[1]
            except KeyError:
                average[e[0]] = e[1]

    comp_dict["average6"] = list()
    for k in average.keys():
        comp_dict["average6"].append([k, average[k] / 6])
    file = open("comparison.pkl", "xb")
    pickle.dump(comp_dict, file)
    file.close()


if __name__ == "__main__":
    if not os.path.exists('data_g'):
        init()
    if not os.path.exists('comparison.pkl'):
        search(ref_image='ref.jpeg')
    comp_dict_file = open('comparison.pkl', 'rb')
    comparison_dict = pickle.load(comp_dict_file, encoding='bytes')
    os.makedirs('result', exist_ok=True)
    for k, v in comparison_dict.items():
        images = [x[0] for x in v[1:11]]
        p = os.path.join('result', k)
        os.makedirs(p, exist_ok=True)
        for image in images:
            shutil.copy(image, p)
