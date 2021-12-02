from PIL import Image

from waitress import serve
from flask_cors import CORS
from flask_restplus import Resource, Api, reqparse
from flask import Flask

import base64
from .functions import insert_images, sqldb

from PIL import Image
import io
import glob
import os

def image_to_byte_array(image:Image):
  imgByteArr = io.BytesIO()
  image.save(imgByteArr, format=image.format)
  imgByteArr = imgByteArr.getvalue()
  return base64.b64encode(imgByteArr)

def init():
    for f in glob.glob('assets/data/**/*.*', recursive=True):
        insert_images(f, os.path.basename(os.path.dirname(f)))

image_size = (512, 512)
def get_images():
    image_data = sqldb.fetch_data(table='image_data')
    image_ids = set([e[0] for e in image_data])
    
    training_data = sqldb.fetch_data(table='training_data')
    training_ids = set([e[0] for e in training_data])
    image_ids_to_train = image_ids - training_ids
    image_data = list(filter(lambda x: x[0] not in image_ids_to_train, image_data))
    return image_to_byte_array(Image.open(image_data[0][1]).thumbnail(image_size))


def get_style_images(style_id, page_num=None):
    styles = sqldb.fetch_data(table='styles')
    try:
        if page_num is None:
            page_num = 0
        else:
            page_num = int(page_num)  - 1
        style = styles[style_id][0]
        images = []
        style_images = sqldb.fetch_image_paths(style=style)
        for image in style_images[page_num*10:page_num*10+10]:
            image = Image.open(image)
            image.thumbnail(image_size)
            images.append(image_to_byte_array(image))
        return str(images)
    except IndexError:
        return None

def create_app():
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
                rv['image'] = get_images()
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

    @api.route('/get_stlye_images')
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
    return app




if __name__ == "__main__":
    serve(create_app(), host='0.0.0.0', port=8000, threads=20)