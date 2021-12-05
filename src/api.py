import io
import os
import glob
import base64
import math
from PIL import Image
from flask import Flask
from waitress import serve
from flask_cors import CORS
from flask_restplus import Resource, Api, reqparse
import pickle
from functions import get_images, get_style_images, insert_images, add_entry, get_loss, sqldb
from main import init

import numpy as np
from dateutil.parser import parse


image_size = (512, 512)


def create_app():
    #Uncomment to refresh the database
    # sqldb.drop_all()
    # init()
    app = Flask("foo", instance_relative_config=True)

    api = Api(
        app,
        version='1.0.0',
        title='Startly Backend App',
        description='Startly Backend App',
        default='Startly Backend App',
        default_label=''
    )

    CORS(app)

    @api.route('/give_image')
    class GetImageService(Resource):
        @api.doc(responses={"response": 'json'})
        def get(self):
            try:
                rv = dict()
                rv['image'], rv['image_id'] = get_images()
                return rv, 200
            except Exception as e:
                rv = dict()
                rv['status'] = str(e)
                return rv, 404
    

    @api.route('/give_style_dict')
    class GetStyleDict(Resource):
        @api.doc(responses={"response": 'json'})
        def get(self):
            try:
                rv = dict()
                style_dict = dict()
                styles = sqldb.fetch_data(table='styles')
                for style in styles:
                    style_dict[style[0]] = style[1]
                rv['style_dict'] = style_dict
                return rv, 200
            except Exception as e:
                rv = dict()
                rv['status'] = str(e)
                return rv, 404

    get_style = reqparse.RequestParser()
    get_style.add_argument('style_id',
                           type=str,
                           help='Id of the style',
                           required=True)
    get_style.add_argument('page_num',
                           type=str,
                           help='Page number of the style images(10/page)')

    @api.route('/get_style_images')
    @api.expect(get_style)
    class GetStyleImage(Resource):
        @api.expect(get_style)
        @api.doc(responses={"response": 'json'})
        def post(self):
            try:
                args = get_style.parse_args()
            except Exception as e:
                rv = dict()
                rv['status'] = str(e)
                return rv, 404
            try:
                style_id = int(args['style_id'])
                page_no = args['page_num']
                rv = dict()
                rv['images'] = get_style_images(style_id, page_no)
                rv['status'] = 'Success'
                return rv, 200
            except Exception as e:
                rv = dict()
                rv['status'] = str(e)
                return rv, 404
    
    training_data = reqparse.RequestParser()
    training_data.add_argument('image_id',
                               type=str,
                               help='Id of the image',
                               required=True)
    training_data.add_argument('style_id',
                               type=str,
                               help='Id of the style',
                               required=True)
    training_data.add_argument('percentage_match',
                               type=str,
                               help='Percentage match given by user')

    average_vectors_g = None

    def change_global_var(value):
        global average_vectors_g
        average_vectors_g = value

    @api.route('/submit_result')
    @api.expect(training_data)
    class SubmitResult(Resource):
        @api.expect(training_data)
        @api.doc(responses={"response": 'json'})
        def post(self):
            try:
                args = training_data.parse_args()
            except Exception as e:
                rv = dict()
                rv['status'] = str(e)
                return rv, 404
            try:
                image_id = int(args['image_id'])
                style_id = int(args['style_id'])
                percentage_match = float(args['percentage_match'])
                average_vectors = sqldb.fetch_data(table='average_vector_data')
                if average_vectors_g != average_vectors:
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
                    change_global_var(average_vectors)
                image_vector_folder = sqldb.fetch_vector_paths(imageid=image_id)[0]
                image_vector = dict()
                for vector_file in glob.glob(os.path.join(image_vector_folder, '*.pkl')):
                    file = open(vector_file, 'rb')
                    vector = pickle.load(file)
                    file.close()
                    image_vector[os.path.basename(vector_file)] = vector            
                
                losses = []
                style = sqldb.fetch_style_name(style_id)
                for vector_file in style_vectors[style].keys():
                    vector_file = vector_file
                    loss = get_loss(image_vector[vector_file], style_vectors[style][vector_file]).numpy()
                    losses.append(math.log(loss, 10))
                losses = np.array(losses)
                match_percentage = losses.copy()
                max_ = match_percentage.max()
                min_ = match_percentage.min()
                delta = max_ - min_
                score = []
                for i in range(len(match_percentage)):
                    match_percentage[i] = 100 - ((match_percentage[i] - min_)*100/delta)
                    score.append(10 - abs(percentage_match - match_percentage[i])/10)
                add_entry(image_id, style_id, str(losses), str(score))

                rv = dict()
                rv['status'] = 'Success'
                return rv, 200
            except Exception as e:
                rv = dict()
                rv['status'] = str(e)
                return rv, 404
    return app


if __name__ == "__main__":
    serve(create_app(), host='0.0.0.0', port=8000, threads=20)
