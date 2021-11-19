import os
import glob
from functions import generate_gram_matrices, insert_images


def init():
    for f in glob.glob('../../../startly_2_0_0/data/**/*.*', recursive=True):
        insert_images(f, os.path.basename(os.path.dirname(f)))
    generate_gram_matrices()
    

if __name__ == "__main__":
    init()
