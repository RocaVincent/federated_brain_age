# INPUTS (ids, features, model_dir, dest_path)
import pandas as pd
import numpy as np
df = pd.read_csv('/NAS/deathrow/vincent/declearn_test/data/stroke/new_exp/cv_split_v2/val5.csv')
ids = df[['sub_id','description']].to_dict('list')
df.set_index(['sub_id','description'], inplace=True)
data = pd.read_csv('/NAS/deathrow/vincent/declearn_test/data/stroke/new_exp/synthseg_volumes.csv', index_col=['sub_id','description'])
cols = data.columns[1:]
#cols = np.loadtxt('/NAS/deathrow/vincent/declearn_test/data/stroke/new_exp/radiomics_list.txt', dtype=str)
data[cols] = data[cols].divide(data['total intracranial'], axis='index')
features = data.loc[df.index, cols].to_numpy()
model_dir = '/NAS/deathrow/vincent/declearn_test/saved_models/new_models/synthseg_simple_declearn_decay/model5'
dest_path = '/NAS/deathrow/vincent/declearn_test/predictions/new_predictions/synthseg_simple_declearn_decay/val5.csv'
#############

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
