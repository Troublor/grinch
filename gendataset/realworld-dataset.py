import numpy as np
import glob
from scipy import ndimage
import imageio


class DataProcessor(object):
    def __init__(self, dirname):
        self.dir = dirname

    def read_imgs(self):
        imageset = []
        for image_path in glob.glob("{}/*.png".format(self.dir)):
            # image = np.array(ndimage.imread(image_path, flatten=False))
            image = imageio.imread(image_path)
            # print(image.shape)
            image = np.reshape(image, (image.shape[0] * image.shape[1], 1))
            # print(image.shape)
            imageset.append(image)
        print('Read {} images.'.format(len(imageset)))
        print('Dimension of image: {}.'.format(len(imageset[0])))
        return imageset


# dprocessor = DataProcessor('../aloi-500')
# dprocessor.read_imgs()