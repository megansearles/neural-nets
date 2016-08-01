# this has a function which can show you an image within the h5 database
import h5py
import numpy as np
import matplotlib.pyplot as pp


class viewer :

    def __init__(self, datapath) :
        self.data = h5py.File(datapath, 'r')


    def show_image(self, *indices) :

        image = self.data

        for i in indices :
            image = image[i]

        image = np.array(image)

        image = image[:,:,(2,1,0)] #convert to rgb

        pp.imshow(image)
        pp.show()

