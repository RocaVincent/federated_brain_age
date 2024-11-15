mri_paths = None # TO DEFINE -> paths of the preprocessed MR images in Niftti format
ids = None # TO DEFINE -> Dictionary identifying each individual for the output CSV, e.g. {'sub_id':<list>, 'session_id':<list>}
model_dir = None # TO DEFINE -> path of the directory of the trained model (corresponds to 'dest_dir' in *./training/main_centralized.py*)
dest_path = None # TO DEFINE -> path where the predictions will be saved in CSV format

##### END INPUTS ########

import pandas as pd
from .model_architecture import Regressor
import numpy as np
import nibabel as nib
from tensorflow import constant as tf_constant

model = Regressor(declearn=False)
model.load_weights(f"{model_dir}/last_model.h5")

ids['prediction'] = np.empty(len(mri_paths))
for i,mri_path in enumerate(mri_paths):
    print(f"Processing image {i+1}/{len(mri_paths)}", end='\r')
    mri = nib.load(mri_path).get_fdata()
    mri = np.expand_dims(mri, axis=(0,4))
    mri = tf_constant(mri, dtype='float32')
    ids['prediction'][i] = model(mri, training=False).numpy()[0,0]
    
pd.DataFrame(ids).to_csv(dest_path, index=False)