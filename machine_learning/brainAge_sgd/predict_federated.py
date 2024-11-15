features = None # TO DEFINE -> array-like of shape (n_individuals,n_features)
ids = None # TO DEFINE -> Dictionary identifying each individual for the output CSV, e.g. {'sub_id':<list>, 'session_id':<list>}
model_dir = None # TO DEFINE -> path of the directory of the trained model (corresponds to 'dest_dir' in *train_declearn.py*)
dest_path = None # TO DEFINE -> path where the predictions will be saved in CSV format

##### END INPUTS ########

import joblib
import pandas as pd
from sklearn.linear_model import SGDRegressor
from declearn.model.sklearn import SklearnSGDModel
from declearn.utils import json_load

preproc = joblib.load(f"{model_dir}/preproc_pipeline.joblib")
features = preproc.transform(features)

model = SklearnSGDModel(SGDRegressor())
weights = json_load(f"{model_dir}/model_state_best.json")
model.set_weights(weights)
ids['prediction'] = model._predict(features)
pd.DataFrame(ids).to_csv(dest_path, index=False)
