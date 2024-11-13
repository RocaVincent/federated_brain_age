from tensorflow import function as tf_function, TensorSpec, pad as tf_pad, numpy_function
from tensorflow.random import uniform
from scipy import ndimage
import numpy as np

from ...constants import IMAGE_SHAPE



######### TRANSFORMATION FUNCTIONS ###############

@tf_function(input_signature=(
                TensorSpec(shape=IMAGE_SHAPE+[1], dtype='float32'),
                TensorSpec(shape=(3,), dtype='int32')
            ))
def shift_tensors(mri, shifts):
    paddings = ((shifts[0],0) if shifts[0]>=0 else (0,-shifts[0]),
               (shifts[1],0) if shifts[1]>=0 else (0,-shifts[1]),
               (shifts[2],0) if shifts[2]>=0 else (0,-shifts[2]),
               (0,0))
    mri = mri[paddings[0][1]:IMAGE_SHAPE[0]-paddings[0][0],
              paddings[1][1]:IMAGE_SHAPE[1]-paddings[1][0],
              paddings[2][1]:IMAGE_SHAPE[2]-paddings[2][0],:]
    return tf_pad(mri, paddings=paddings, mode="SYMMETRIC")


def rotate_data(mri, angle):
    return ndimage.rotate(mri, angle=angle, axes=(0,1), reshape=False, order=3, mode='nearest')


@tf_function(input_signature=(
                TensorSpec(shape=IMAGE_SHAPE+[1], dtype='float32'),
                TensorSpec(shape=(), dtype='float32'),
            ))
def rotate_tensors(mri, angle):
    """
    angle in degree
    """
    return numpy_function(rotate_data, inp=[mri, angle], Tout='float32', stateful=False)


############ FONCTION GLOBALE D'IN-LINE AUGMENTATION ##################
ANGLE_MAX = 10
SHIFT_MAX = 15,20,2

@tf_function(input_signature=(
                TensorSpec(shape=IMAGE_SHAPE+[1], dtype='float32'),
            ))
def augmentation(mri):
    rotation_angle = uniform(shape=(), minval=-ANGLE_MAX, maxval=ANGLE_MAX, dtype='float32')
    mri = rotate_tensors(mri, rotation_angle)
    shifts = (
        uniform((), minval=-SHIFT_MAX[0], maxval=SHIFT_MAX[0], dtype='int32'),
        uniform((), minval=-SHIFT_MAX[1], maxval=SHIFT_MAX[1], dtype='int32'),
        uniform((), minval=-SHIFT_MAX[2], maxval=SHIFT_MAX[2], dtype='int32'),
    )
    return shift_tensors(mri, shifts)