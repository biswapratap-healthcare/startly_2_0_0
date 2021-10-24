import os
import shutil
import glob
import pickle
import tensorflow as tf
import numpy as np
from PIL import Image
from keras import models
from tensorflow.python.keras.preprocessing import image as kp_image

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


def get_style_loss(base_style, gram_target):
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))


def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))


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


def init():
    model = get_model()
    for layer in model.layers:
        layer.trainable = False
    errors = list()
    for f in glob.glob('data/**/*.*', recursive=True):
        f_replace = f.replace('data', 'data_g')
        os.makedirs(f_replace, exist_ok=True)
        try:
            style_features, content_features = get_feature_representations(model, f)
            gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
            for ln, c in zip(content_layers, content_features):
                pp = os.path.join(f_replace, ln + '.pkl')
                if os.path.exists(pp):
                    continue
                file = open(pp, "xb")
                pickle.dump(c, file)
                file.close()
            for ln, g in zip(style_layers, gram_style_features):
                pp = os.path.join(f_replace, ln + '.pkl')
                if os.path.exists(pp):
                    continue
                file = open(pp, "xb")
                pickle.dump(g, file)
                file.close()
        except Exception as e:
            errors.append([e, f])


def search(ref_image):
    model = get_model()
    for layer in model.layers:
        layer.trainable = False
    style_features, content_features = get_feature_representations(model, ref_image)
    comp_dict = dict()
    # for ln, c in zip(content_layers, content_features):
    #     for f in glob.glob('data_g/**/' + ln + '.pkl', recursive=True):
    #         fn = f.replace('data_g', 'data')
    #         fn = os.path.dirname(fn)
    #         infile = open(f, 'rb')
    #         c_comp = pickle.load(infile, encoding='bytes')
    #         try:
    #             c_loss = get_content_loss(c, c_comp).numpy()
    #             comp_dict[ln].append([fn, c_loss])
    #         except KeyError:
    #             c_loss = get_content_loss(c, c_comp).numpy()
    #             comp_dict[ln] = [[fn, c_loss]]
    #     comp_dict[ln] = sorted(comp_dict[ln], key=lambda x: x[1])
    for ln, s in zip(style_layers, style_features):
        for f in glob.glob('data_g/**/' + ln + '.pkl', recursive=True):
            fn = f.replace('data_g', 'data')
            fn = os.path.dirname(fn)
            infile = open(f, 'rb')
            s_comp = pickle.load(infile, encoding='bytes')
            try:
                s_loss = get_style_loss(s, s_comp).numpy()
                comp_dict[ln].append([fn, s_loss])
            except KeyError:
                s_loss = get_style_loss(s, s_comp).numpy()
                comp_dict[ln] = [[fn, s_loss]]
        comp_dict[ln] = sorted(comp_dict[ln], key=lambda x: x[1])

    average = dict()
    for k1 in comp_dict.keys():
        for e in comp_dict[k1]:
            try:
                average[e[0]] += e[1]
            except KeyError:
                average[e[0]] = e[1]

    comp_dict["average"] = list()
    for k1 in average.keys():
        comp_dict["average"].append([k1, average[k1] / 7])
    file = open("comparison.pkl", "xb")
    pickle.dump(comp_dict, file)
    file.close()


if __name__ == "__main__":
    if not os.path.exists('data_g'):
        init()
    # exit(0)
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
