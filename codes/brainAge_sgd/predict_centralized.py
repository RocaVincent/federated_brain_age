# INPUTS
import pandas as pd
import numpy as np
# df = pd.read_csv('/NAS/deathrow/vincent/declearn_test/data/stroke/new_exp/metadata.csv')
# df = df[df.center!='CHU_Lille']
df = pd.read_csv('/NAS/deathrow/vincent/declearn_test/data/stroke/new_exp/cv_split_v2/val1.csv')
ids = df[['sub_id','description']].to_dict('list')
df.set_index(['sub_id','description'], inplace=True)

data = pd.read_csv('/NAS/deathrow/vincent/declearn_test/data/stroke/new_exp/synthseg_volumes.csv', index_col=['sub_id','description'])
cols = data.columns[1:]
#cols = np.loadtxt('/NAS/deathrow/vincent/declearn_test/data/stroke/new_exp/radiomics_list.txt', dtype=str)
data[cols] = data[cols].divide(data['total intracranial'], axis='index')
features = data.loc[df.index,cols].to_numpy()

MODEL_PATH = '/NAS/deathrow/vincent/declearn_test/saved_models/new_models/synthseg_simple_centralized/model1.joblib'
DEST_PATH = '/NAS/deathrow/vincent/declearn_test/predictions/new_predictions/synthseg_simple_centralized/val1.csv'

# END INPUTS

import joblib
import pandas as pd
model = joblib.load(MODEL_PATH)
ids['prediction'] = model.predict(features)
pd.DataFrame(ids).to_csv(DEST_PATH, index=False)