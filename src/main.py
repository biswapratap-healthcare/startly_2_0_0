import glob
from functions import generate_gram_matrices, insert_images

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# import tensorflow as tf


def init():
    for f in glob.glob('src/assets/data/**/*.*', recursive=True):
        insert_images(f, os.path.basename(os.path.dirname(f)))
    generate_gram_matrices()
    

if __name__ == "__main__":
    init()
