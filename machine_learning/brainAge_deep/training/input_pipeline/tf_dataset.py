import nibabel as nib
import numpy as np
from tensorflow import constant as tf_constant, TensorSpec
from tensorflow.data import Dataset, AUTOTUNE

from declearn.dataset.tensorflow import TensorflowDataset

from ...constants import IMAGE_SHAPE
from .data_augmentation import augmentation

def training_set_from_pathList(mri_paths, targets, batch_size, buffer_size=300):
    """
    mri_paths: a 1d array-like (n,)
    targets: a 1d array-like (n,)
    batch_size
    """
    def gen():
        for mri_path,target in zip(mri_paths,targets):
            mri = np.expand_dims(nib.load(mri_path).get_fdata(), axis=-1)
            yield tf_constant(mri, dtype='float32'), tf_constant(target, dtype='float32')

    dataset = Dataset.from_generator(gen, output_signature=(TensorSpec(shape=IMAGE_SHAPE+[1], dtype='float32'),
                                                            TensorSpec(shape=(), dtype='float32')))
    if not buffer_size: buffer_size = len(mri_paths)
    dataset = dataset.cache().shuffle(buffer_size, reshuffle_each_iteration=True)
    dataset = dataset.map(lambda mri,target: (augmentation(mri), target), num_parallel_calls=AUTOTUNE, deterministic=False)
    return dataset.batch(batch_size, drop_remainder=True, num_parallel_calls=AUTOTUNE, deterministic=False).prefetch(AUTOTUNE)

def declearn_set_from_pathList(mri_paths, targets):
    """
    mri_paths: a 1d array-like (n,)
    targets: a 1d array-like (n,)
    """
    def gen():
        for mri_path,target in zip(mri_paths,targets):
            mri = np.expand_dims(nib.load(mri_path).get_fdata(), axis=-1)
            yield tf_constant(mri, dtype='float32'), tf_constant(target, dtype='float32')
            
    dataset = Dataset.from_generator(gen, output_signature=(TensorSpec(shape=IMAGE_SHAPE+[1], dtype='float32'),
                                                            TensorSpec(shape=(), dtype='float32')))
    dataset = dataset.cache().map(lambda mri,target: (augmentation(mri), target), num_parallel_calls=AUTOTUNE, deterministic=False)
    return TensorflowDataset(dataset)
    