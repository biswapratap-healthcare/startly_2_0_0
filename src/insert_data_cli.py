import tensorflow as tf
import numpy as np
import os
import random
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import math

from model import init_model
from functions import get_loss, add_entry, get_input1, get_input2, sqldb


def get_img(style_vectors):
    model_dir = os.path.join('assets', 'model')
    model = tf.keras.models.load_model(model_dir)

    image_data = sqldb.fetch_data(table='image_data')
    image_ids = set([e[0] for e in image_data])
    
    training_data = sqldb.fetch_data(table='training_data')
    training_ids = set([e[0] for e in training_data])
    image_ids_to_train = image_ids - training_ids
    image_data = list(filter(lambda x: x[0] in image_ids_to_train, image_data))
    
    data = random.choice(image_data)
    id = data[0]
    image_vector_folder = sqldb.fetch_vector_paths(imageid=id)[0]
    layer_order = [e.replace('.pkl','') for e in os.listdir(image_vector_folder)]
    image_vector = dict()
    for vector_file in glob.glob(os.path.join(image_vector_folder, '*.pkl')):
        file = open(vector_file, 'rb')
        vector = pickle.load(file)
        file.close()
        image_vector[os.path.basename(vector_file)] = vector

    image_path = data[1]

    fig = plt.figure()
    plt.imshow(mpimg.imread(image_path))
    plt.title('Orignal image')
    plt.show()
    
    print('getting_features')
    x1 = get_input1(id)
    x1 = np.array([x1])
    print('got features')
    average_vectors = sqldb.fetch_data(table='average_vector_data')
    print('got_avg_data')
    for style in style_vectors.keys():
        print('in style')
        x2 = get_input2(average_vectors, style)
        x2 = np.array([x2])

        losses = []
        score = []

        image_paths = sqldb.fetch_image_paths(style=style)
        image_paths = random.sample(image_paths, 9)
        print('images_selected')
        fig = plt.figure()
        fig.suptitle(style)
        columns = 3
        rows = 3
        i=1
        for image_path in image_paths:
            img = mpimg.imread(image_path)
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
            i += 1
        plt.show()
        print('plotted')
        for vector_file in style_vectors[style].keys():
            vector_file = vector_file
            loss = get_loss(image_vector[vector_file], style_vectors[style][vector_file]).numpy()
            losses.append(math.log(loss, 10))
        losses = np.array(losses)
        max_ = losses.max()
        min_ = losses.min()
        delta = max_ - min_
        for i in range(len(losses)):
            losses[i] = (losses[i] - min_)*100/delta
            
        x3 = losses
        x3 = np.array([x3])
        y = model.predict([x1.reshape(x1.shape[0], -1), x2.reshape(x2.shape[0], -1), x3])
        
        weights = []
        i=0
        while i < 88:
            temp = []
            for j in range(i, i+11):
                temp.append(y[0][j])
            weights.append(np.argmax(np.asarray(temp)))
            i += 11
        p = 0
        w_total = 0
        for weight, loss in zip(weights, losses):
            p+= weight*loss
            w_total += weight
        p = p/w_total

        print('Layer_name: match% * weight = result')
        for layer, loss, weight in zip(layer_order, losses, weights):
            print(f'{layer}: {loss:.2f}% * {weight} = {loss*weight:.2f}%')
        print(f'Total percentage match(with weighted average) for {style} is {p:.2f}%')

        for vector_file in style_vectors[style].keys():
            score.append(int(input(f"Give score for {vector_file} loss for {style} layer out of 10: ")))
        
        add_entry(id, style, str(losses), str(score))


def init():
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
    get_img(style_vectors)


if __name__ == "__main__":
    init_model()
    init()
