##This takes the forms dataset and pulls out a training and testing dataset
# just run it in the same directory as the forms.h5 file
import h5py
import numpy as np
from tflearn.data_utils import shuffle, to_categorical



def process_form_data(filename) :
    data = h5py.File(filename, 'r')
    output = h5py.File('forms_out.h5', 'w')

    test_image = output.create_dataset('test_image', (330, 256, 256, 3), dtype=np.uint8)
    train_image = output.create_dataset('train_image', (770, 256, 256, 3), dtype=np.uint8)
    test_label  = output.create_dataset('test_label', (330,11), dtype=np.int8)
    train_label  = output.create_dataset('train_label', (770,11), dtype=np.int8)

    image, labels = shuffle(data['image'], data['form'])

    onehot_labels = to_categorical(labels, 11)


    count = {}
    train_count = 0
    test_count = 0
    for i, l in enumerate(labels) :

        if l not in count :
            count[l] = 0

        if count[l] > 29 :
            train_image[train_count] = image[i]
            train_label[train_count] = onehot_labels[i]
            train_count += 1

        else :
            test_image[test_count] = image[i]
            test_label[test_count] = onehot_labels[i]
            test_count += 1

        count[l] += 1

    output.close()

process_form_data("forms.h5")

