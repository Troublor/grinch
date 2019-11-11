import re
import numpy as np
import glob
from scipy import ndimage
import imageio


class DataProcessor(object):
    def __init__(self, dirname='../aloi-500-balance'):
        self.dir = dirname

    def read_imgs(self):
        imageset = []
        index = []
        extension = '*.png'
        for image_path in glob.glob("{}/*/*.png".format(self.dir)):
            cls_index = re.findall(r'(\d*)_', image_path.split('\\')[-1])
            image = imageio.imread(image_path)
            image = np.reshape(image, (image.shape[0] * image.shape[1], 1))
            imageset.append(image)
            index.append(int(cls_index[0]))
            # print('index = {}'.format(cls_index))
            # print(image_path)

        assert len(index) == len(imageset)
        print('Read {} images.'.format(len(imageset)))
        print('Dimension of image: {}.'.format(len(imageset[0])))

        return imageset, index


# dprocessor = DataProcessor('../aloi-500-balance')
# # dprocessor.read_imgs()