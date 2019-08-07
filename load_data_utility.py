import pathlib
import random
import tensorflow as tf
import data_augmentation_utility as da_util


# Given a directory of data of the form: ~/data/labels/images.png

# data augmentation methods in data_augmentation_utility.py

def augment_image(image):
                       
#     image = da_util.rotate(image)

    image = da_util.horiz_flip(image)
    
    image = da_util.translate(image)
    image = da_util.crop(image)

    return image

# load an array of image paths
def load_image_paths(path):
    
    data_root = pathlib.Path(path)
    
    # create a list of every file and its label index
    all_image_paths = list(data_root.glob('*/*'))
    
    all_image_paths = [path for path in all_image_paths if path.name != ".DS_Store"]
    all_image_paths = [str(path) for path in all_image_paths if path.name != ".DS_Store"]
#     all_image_paths = [str(path) for path in all_image_paths]
    return all_image_paths

# randomly shuffle train and test values and split based on parameters
def split(image_paths, split=[0.6,0.2,0.2], seed=777):
    random.Random(seed).shuffle(image_paths)
    
    boundary1 = int(len(image_paths) * split[0])
    boundary2 = int(len(image_paths) * (split[0]+split[1]))
    
    train = image_paths[:boundary1]
    evaluate = image_paths[boundary1: boundary2]
    test = image_paths[boundary2:]
    
    return train, evaluate, test

# preprocessing functions
def preprocess_image(image):
    image = tf.image.decode_png(image, channels=1)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, shape=[512,512])
    image = image / 255.0
    
#     image = tf.div(
#        tf.subtract(
#           image, 
#           tf.reduce_min(image)
#        ), 
#        tf.subtract(
#           tf.reduce_max(image), 
#           tf.reduce_min(image)
#        )
#     )
    
    
    return image

def load_and_preprocess_image(path, augment=True):
    image = tf.read_file(path)
    image = preprocess_image(image)
    
    if (augment):
        image = augment_image(image[:,:,None])
    
    return image

def load(path, image_paths, training=True, augment=True, batch_size=64, shuffle=True, drop_remainder=False):
    with tf.device("/CPU:0"):
        # data root
        data_root = pathlib.Path(path)

        # return label names
        label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

        # assign index to label
        label_to_index = dict((name, index) for index,name in enumerate(label_names))

        # array of all labels corresponding to image_paths
        all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                            for path in image_paths]

        # make path dataset
        path_ds = tf.data.Dataset.from_tensor_slices(image_paths)

        # get image tensors by mapping function over the path dataset
        image_ds = path_ds.map(lambda path : load_and_preprocess_image(path, augment=augment))

        # create label dataset
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

        # zip together image and label dataset into dataset of tuples for
        # estimator input
        image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

        # Setting a shuffle buffer size as large as the dataset ensures that the data is
        # completely shuffled
        
        image_label_ds = image_label_ds.batch(batch_size, drop_remainder=drop_remainder)
        
        if (training):
            print("shuffling and repeating b/c training flag set")

            image_label_ds = image_label_ds.repeat()
        else:
            image_label_ds = image_label_ds.repeat(count=1)
        
        
        if (shuffle):
            image_label_ds = image_label_ds.shuffle(buffer_size = 5 * batch_size)
            
        # `prefetch` lets the dataset fetch batches, in the background while the model is training.
        image_label_ds = image_label_ds.prefetch(6)
       
        
        return image_label_ds