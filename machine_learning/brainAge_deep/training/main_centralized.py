mri_paths = None # TO DEFINE -> paths of the preprocessed MR images in Niftti format
ages = None # TO DEFINE -> corresponding ages
dest_dir = None # TO DEFINE -> path of the directory where the model will be saved along with the training losses

#### END INPUTS ######

BATCH_SIZE = 8
N_EPOCHS = 1000
INIT_LR = 0.001
END_LR = 0.0001

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

from tensorflow.config import list_physical_devices, set_logical_device_configuration, LogicalDeviceConfiguration
gpu = list_physical_devices('GPU')[0]
set_logical_device_configuration(gpu, [LogicalDeviceConfiguration(memory_limit=1000*9)])


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
import json
import numpy as np

from ..model_architecture import Regressor
from .input_pipeline.tf_dataset import training_set_from_pathList

dataset = training_set_from_pathList(mri_paths, ages, BATCH_SIZE)

steps_per_epoch = len(mri_paths)//BATCH_SIZE
optimizer = Adam(learning_rate=PolynomialDecay(INIT_LR, N_EPOCHS*steps_per_epoch, END_LR))

model = Regressor(declearn=False, last_bias_value=np.mean(ages))
model.compile(loss='mae', optimizer=optimizer, jit_compile=True)

# training and saving
history = model.fit(dataset, epochs=N_EPOCHS)

with open(dest_dir+'/stats.json', 'w') as f:
    json.dump(history.history, f)
    
model.save_weights(dest_dir+'/last_model.h5')