import os
import pickle
from src.sql import SqlDatabase

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

sqldb = SqlDatabase()

layers = ['average',
          'block1_conv1',
          'block2_conv1',
          'block3_conv1',
          'block4_conv1',
          'block5_conv1',
          'block5_conv2',
          'block5_pool']

average_vectors = sqldb.fetch_data(params=['*'], table='average_vector_table')
style_vectors = dict()
for data in average_vectors:
    style = data[0]
    style_vectors[style] = dict()
    for layer, vector in zip(layers, data[1:]):
        style_vectors[style][layer] = pickle.loads(vector)
