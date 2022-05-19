import json
import os
import glob
import math
import pickle
import numpy as np
from main import init
from flask import Flask
from waitress import serve
from flask_cors import CORS
from flask_restplus import Resource, Api, reqparse
from functions import get_images, get_style_images, add_entry, get_loss, sqldb, filter_image_fn
from init_style_vector import style_vectors
from model import init_model

image_size = (512, 512)


def create_app():
    # Uncomment to refresh the database
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

    give_image_parser = reqparse.RequestParser()
    give_image_parser.add_argument('style',
                                   type=str,
                                   help='A sample image of this style',
                                   required=True)

    @api.route('/give_image')
    @api.expect(give_image_parser)
    class GetImageService(Resource):
        @api.expect(give_image_parser)
        @api.doc(responses={"response": 'json'})
        def get(self):
            try:
                args = give_image_parser.parse_args()
            except Exception as e:
                rv = dict()
                rv['status'] = str(e)
                return rv, 404
            try:
                rv = dict()
                style = args['style']
                rv['image'], rv['image_id'] = get_images(style)
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
                styles = sqldb.fetch_data(params=['*'], table='styles')
                for style in styles:
                    style_dict[style[0]] = style[1]
                rv['style_dict'] = style_dict
                return rv, 200
            except Exception as e:
                rv = dict()
                rv['status'] = str(e)
                return rv, 404

    @api.route('/train_model')
    class TrainModelService(Resource):
        @api.doc(responses={"response": 'json'})
        def post(self):
            try:
                rv = dict()
                init_model()
                rv['status'] = 'Success'
                return rv, 200
            except Exception as e:
                rv = dict()
                rv['status'] = str(e)
                return rv, 404

    filter_image = reqparse.RequestParser()
    filter_image.add_argument('style',
                              type=str,
                              help='The style filter.',
                              required=True)
    filter_image.add_argument('image',
                              type=str,
                              help='The image.',
                              required=True)

    @api.route('/filter_image')
    @api.expect(filter_image)
    class FilterImageService(Resource):
        @api.expect(filter_image)
        @api.doc(responses={"response": 'json'})
        def post(self):
            try:
                args = filter_image.parse_args()
            except Exception as e:
                rv = dict()
                rv['status'] = str(e)
                return rv, 404
            try:
                style = args['style']
                image = args['image']
                rv = dict()
                rv['result'] = filter_image_fn(style, image)
                rv['status'] = 'Success'
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
    training_data.add_argument('payload',
                               type=str,
                               help='{"image_id" : "1638803707364", "style_list":[{"style_id":"1","percentage":"30"},{"style_id":"2","percentage":"40"}]}',
                               required=True)

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
                rv = dict()
                rv_status = list()
                payload = args['payload']
                payload_dict = json.loads(payload)
                image_id = payload_dict['image_id']
                style_list = payload_dict['style_list']
                for style in style_list:
                    style_id = style['style_id']
                    percentage = int(style['percentage'])
                    image_vector_dict = sqldb.fetch_image_vectors(img_id=image_id)[0]
                    image_vector = dict()
                    for layer, vector in image_vector_dict.items():
                        image_vector[layer] = pickle.loads(vector)
                    losses = list()
                    style = sqldb.fetch_style_name(style_id)
                    for vector_file in style_vectors[style].keys():
                        vector_file = vector_file
                        loss = get_loss(image_vector[vector_file], style_vectors[style][vector_file]).numpy()
                        losses.append(math.log(loss, 10))
                    match_percentage = np.asarray(losses)
                    max_ = match_percentage.max()
                    min_ = match_percentage.min()
                    delta = max_ - min_
                    score = []
                    for i in range(len(match_percentage)):
                        match_percentage[i] = 100 - ((match_percentage[i] - min_)*100/delta)
                        score.append(10 - abs(percentage - match_percentage[i])/10)
                    rv_status.append(add_entry(image_id, style, str(list(match_percentage)), str(score)))
                rv['status'] = rv_status
                return rv, 200
            except Exception as e:
                rv = dict()
                rv['status'] = str(e)
                return rv, 404
    return app


if __name__ == "__main__":
    serve(create_app(), host='0.0.0.0', port=8000, threads=100)
