import tensorflow as tf
import random

def rotate(image):
    
    return tf.image.rot90(image, tf.random_uniform(shape=[], minval=0, maxval=4, 
                                               dtype=tf.int32))

def horiz_flip(image):
    
    image = tf.image.random_flip_left_right(image)
#     image = tf.image.random_flip_up_down(image)

    return image

def vert_flip(image):
    
#     image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    return image


def crop(image, range_start=0.8, range_end=1.0):
    
    image = tf.image.central_crop(image, central_fraction=random.uniform(range_start, range_end))
    image = tf.image.resize_images(image,tf.constant([512, 512]))
    return image

def translate(image,  x_max=75, y_max=75):
    
    return tf.contrib.image.translate(image, translations=[random.uniform(-1 * x_max, x_max), 
                                                           random.uniform(-1 * y_max, y_max)])
