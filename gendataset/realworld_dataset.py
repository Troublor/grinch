import re
import numpy as np
import glob
from scipy import ndimage
import imageio

from gendataset.shuffle import round_robin, random_shuffle


class DataProcessor(object):
    def __init__(self, dirname='aloi-500-balance'):
        self.dir = dirname

    def read_imgs(self, shuffle=None):
        imageset = []
        index = []
        extension = '*.png'
        for image_path in glob.glob("{}/*/{}".format(self.dir, extension)):
            cls_index = re.findall(r'(\d*)_', image_path.split('\\')[-1])
            image = imageio.imread(image_path)
            image = np.ndarray.tolist(np.reshape(image, image.size))
            imageset.append(image)
            index.append(int(cls_index[0]))
            # print('index = {}'.format(cls_index))
            # print(image_path)

        assert len(index) == len(imageset)
        print('Read {} images.'.format(len(imageset)))
        print('Dimension of image: {}.'.format(len(imageset[0])))

        if shuffle is None:
            return index, imageset

        clusters = []
        for i in range(len(index)):
            if len(clusters) <= index[i] - 1:
                while len(clusters) < index[i] - 1 + 1:
                    clusters.append([])
            clusters[index[i] - 1].append(imageset[i])

        return shuffle(clusters)
