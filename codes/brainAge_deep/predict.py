"""
INPUTS : 
    - model (with weights loaded)
    - ids (dict)
    - mri_paths
    - csv_dest
"""
import pandas as pd
from declearn.utils import json_load
from .model_architecture import Regressor
model = Regressor(declearn=True)
# model.load_weights('/NAS/deathrow/vincent/declearn_test/saved_models/new_models/deep_lille/last_model.h5')
weights = json_load('/NAS/deathrow/vincent/declearn_test/saved_models/new_models/deep_declearn/model5/model_state_best.json')
model.set_weights(weights)
model = model._model

df = pd.read_csv('/NAS/deathrow/vincent/declearn_test/data/stroke/new_exp/cv_split_v2/val5.csv')
ids = df[['sub_id','description']].to_dict('list')
mri_paths = df.apply(lambda row: f"/NAS/coolio/vincent/data/cohorte_AVC/n4brains_regNorm/{row.sub_id}_{row.description}.nii.gz", axis=1).tolist()
csv_dest = '/NAS/deathrow/vincent/declearn_test/predictions/new_predictions/deep_declearn/val5.csv'

# END INPUTS

import pandas as pd
import numpy as np
import nibabel as nib
from tensorflow import constant as tf_constant

ids['prediction'] = np.empty(len(mri_paths))

for i,mri_path in enumerate(mri_paths):
    print(f"Processing image {i+1}/{len(mri_paths)}", end='\r')
    mri = nib.load(mri_path).get_fdata()
    mri = np.expand_dims(mri, axis=(0,4))
    mri = tf_constant(mri, dtype='float32')
    ids['prediction'][i] = model(mri, training=False).numpy()[0,0]
    
pd.DataFrame(ids).to_csv(csv_dest, index=False)