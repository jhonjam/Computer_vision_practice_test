import os 
import glob
import sys
import os.path

import tensorflow
from skimage import io
import  PIL
from PIL import Image
import sys


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import argparse
import os.path

parser = argparse.ArgumentParser()


parser.add_argument(
    "--path_rgb_image",
    default="C:/Users/jhonj/Downloads/dataset_parafuso/",
    type=str,
    help="Path to the folder of image rgb for test",
)


parser.add_argument(
    "--res_dir",
    default="C:/Users/jhonj/Downloads/dataset_parafuso2",
    help="Path to the folder where the new image  rgb is save",
)




# Funcion general
def main(config):

    paths_img_rgb = os.path.join(config.path_rgb_image)
    paths = [p.replace("\\", '/') for p in glob.glob("{}/*.jpeg".format(paths_img_rgb))]


    #Funcion que genera nuevas imagenes
    datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.5,
        height_shift_range=0.5,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    
    for i,image in enumerate(paths):
        x = io.imread(image)
        x = x.reshape((1, ) + x.shape)

        for batch in datagen.flow(x, batch_size=16,save_to_dir=config.res_dir,save_prefix='Parafuso',save_format='jpg'):
            i += 1    
            if i > 50:
                break

    




if __name__ == "__main__":
    config = parser.parse_args()

    main(config)