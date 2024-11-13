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


######### INPUT DATA (defines 'dataset' variable) #############
import pandas as pd
BATCH_SIZE = 8

df1 = pd.read_csv('/NAS/deathrow/vincent/declearn_test/data/stroke/new_exp/metadata.csv')
df1 = df1[df1.center=='CHU_Lille']
# df2 = pd.read_csv('/NAS/deathrow/vincent/declearn_test/data/stroke/new_exp/cv_split_v2/train1.csv')
# df = pd.concat((df1,df2))
df = df1


mri_paths = df.apply(lambda row: f"/NAS/coolio/vincent/data/cohorte_AVC/n4brains_regNorm/{row.sub_id}_{row.description}.nii.gz", axis=1).tolist()
ages = df.age.tolist()
dataset = training_set_from_pathList(mri_paths, ages, BATCH_SIZE)
########################################################

# SAVE DIR (defines 'DEST_DIR_PATH' variable)
DEST_DIR_PATH = '/NAS/deathrow/vincent/declearn_test/saved_models/new_models/deep_lille'
#############################################

# number of peochs and optimizer
N_EPOCHS = 1000
INIT_LR = 0.001
END_LR = 0.0001
steps_per_epoch = len(mri_paths)//BATCH_SIZE
optimizer = Adam(learning_rate=PolynomialDecay(INIT_LR, N_EPOCHS*steps_per_epoch, END_LR))


# model instantiation and compilation
model = Regressor(declearn=False, last_bias_value=np.mean(ages))
model.compile(loss='mae', optimizer=optimizer, jit_compile=True)

# training and saving
history = model.fit(dataset, epochs=N_EPOCHS)

with open(DEST_DIR_PATH+'/stats.json', 'w') as f:
    json.dump(history.history, f)
    
model.save_weights(DEST_DIR_PATH+'/last_model.h5')