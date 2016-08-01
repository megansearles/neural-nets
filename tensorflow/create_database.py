#This takes the scraped images and puts them all into a single h5py database resizing them and everything
#Right now if this is run it will create the forms.h5 database from a database.h5 in the same directory
import h5py
import numpy as np
import scipy.misc as misc
import unicodedata
import traceback


def filter_sep(file, sep=';', enclose='"', sub=',') :

    data = open(file).readlines()
    data = [l.split(enclose) for l in data]

    for l in data :
        for i in range(len(l)) :
            if ((i % 2) == 1) :
                l[i] = l[i].replace(sep, sub)

    data = [enclose.join(l) for l in data]

    open('parsed_' + file, 'w').writelines(data)

def count_categories(file) :
    data = open(file).readlines()

    data = [l.split(';') for l in data]

    count = {}

    for l in data :

        if l[7] not in count :
            count[l[7]] = 0

        count[l[7]] += 1

    return count

def get_image(imagepath, size=(256,256)) :
    image = misc.imread(imagepath, mode='RGB')
    image = misc.imresize(image, size, interp='bicubic')

    image = image[:,:,(2,1,0)] #convert to BGR
    #image = image.transpose(2,0,1) #depth first as well, to line up with imagenet
    # it turns out tflearn can not handle depth first it must be depth last

    return image

def create_database(catalog) :

    data = open(catalog).readlines()
    data = [d.strip().split(';') for d in data]

    database = h5py.File('database.h5', 'w')

    artist_name = database.create_dataset("artist_name", (len(data),), dtype='S50')
    born_died = database.create_dataset("born_died", (len(data),), dtype='S75')
    title = database.create_dataset("title", (len(data),), dtype='S140')
    date = database.create_dataset("date", (len(data),), dtype='S60')
    technique = database.create_dataset("technique", (len(data),), dtype='S125')
    location = database.create_dataset("location", (len(data),), dtype='S100')
    images = database.create_dataset("image", (len(data), 256, 256, 3), dtype=np.uint8)
    form = database.create_dataset("form", (len(data),), dtype='S15')
    genre = database.create_dataset("genre", (len(data),), dtype='S15')
    school = database.create_dataset("school", (len(data),), dtype='S15')
    timeframe = database.create_dataset("timeframe", (len(data),), dtype='S15')

    for i, d in enumerate(data) :

        try :
            image = get_image(d[6])
        except :
            print ("failure with {}".format(d[6]))
            traceback.print_exc()
            artist_name[i] = "FAILURE".encode('ascii', 'ignore')
            continue

        images[i] = image

        artist_name[i] = unicodedata.normalize('NFKC', d[0]).encode('ascii', 'ignore')
        born_died[i] = unicodedata.normalize('NFKC', d[1]).encode('ascii', 'ignore')
        title[i] = unicodedata.normalize('NFKC', d[2]).encode('ascii', 'ignore')
        date[i] = unicodedata.normalize('NFKC', d[3]).encode('ascii', 'ignore')
        technique[i] = unicodedata.normalize('NFKC', d[4]).encode('ascii', 'ignore')
        location[i] = unicodedata.normalize('NFKC', d[5]).encode('ascii', 'ignore')
        form[i] = unicodedata.normalize('NFKC', d[7]).encode('ascii', 'ignore')
        genre[i] = unicodedata.normalize('NFKC', d[8]).encode('ascii', 'ignore')
        school[i] = unicodedata.normalize('NFKC', d[9]).encode('ascii', 'ignore')
        timeframe[i] = unicodedata.normalize('NFKC', d[10]).encode('ascii', 'ignore')



    database.close()
        

def count_sections(catalog) :
    data = open(catalog).readlines()
    data = [d.split(';') for d in data]

    count = [0] * len(data[0])

    for l in data :
        for i, d in enumerate(l) :
            if count[i] < len(d) :
                count[i] = len(d)

    print(data[0])

    return count
    
##This creates a smaller forms database from the main database.h5 file
def create_form_database(database_file) :
    data = h5py.File(database_file, 'r')
    output = h5py.File('forms.h5', 'w')

    image = output.create_dataset("image", (1100, 256, 256, 3), dtype=np.uint8)
    form = output.create_dataset("form", (1100,), dtype=np.int8)

    count = {}
    label_index = 0

    first_empty = 0
    for i, f in enumerate(data['form']) :
        if (f == b'glassware') or (f == b'others') :
            continue

        if f not in count :
            count[f] = [0, label_index]
            label_index += 1

        if count[f][0] < 100 :
            image[first_empty] = data['image'][i]
            form[first_empty] = count[f][1]
            count[f][0] += 1
            first_empty += 1

    data.close()

    print(count)

    lines = [f.decode('UTF-8') + ' ' + str(count[f][1]) + ' ' + str(count[f][0]) + '\n' for f in count]
    open("form_labels.txt", 'w').writelines(lines)
 


def main() :
    create_form_database('database.h5')


main()

