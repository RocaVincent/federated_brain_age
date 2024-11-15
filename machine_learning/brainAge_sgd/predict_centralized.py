features = None # TO DEFINE -> array-like of shape (n_individuals,n_features)
ids = None # TO DEFINE -> Dictionary identifying each individual for the output CSV, e.g. {'sub_id':<list>, 'session_id':<list>}
model_path = None # TO DEFINE -> joblib path of the trained model
dest_path = None # TO DEFINE -> path where the predictions will be saved in CSV format

##### END INPUTS ########

import joblib
import pandas as pd
model = joblib.load(MODEL_PATH)
ids['prediction'] = model.predict(features)
pd.DataFrame(ids).to_csv(DEST_PATH, index=False)