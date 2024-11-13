# INPUTS (features, targets, sgd_lrate, alpha_vals, preproc_steps, dest_path)
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler

df = pd.read_csv('/NAS/deathrow/vincent/declearn_test/data/stroke/new_exp/metadata.csv', index_col=['sub_id','description'])
df = df[df.center=='CHU_Lille']
# df2 = pd.read_csv('/NAS/deathrow/vincent/declearn_test/data/stroke/new_exp/cv_split_v2/train1.csv', index_col=['sub_id','description'])
# df = pd.concat((df,df2))

data = pd.read_csv('/NAS/deathrow/vincent/declearn_test/data/stroke/new_exp/synthseg_volumes.csv', index_col=['sub_id','description'])
#cols = np.loadtxt('/NAS/deathrow/vincent/declearn_test/sdata/stroke/new_exp/radiomics_list.txt', dtype=str)
cols = data.columns[1:]
data[cols] = data[cols].divide(data['total intracranial'], axis='index')
features = data.loc[df.index,cols].to_numpy()
targets = df.age.to_numpy()

sgd_lrate = 0.5
alpha_vals = np.logspace(-20,0, num=10)
preproc_steps = [
    MinMaxScaler()#, PolynomialFeatures(degree=2, include_bias=False)
]

dest_path = '/NAS/deathrow/vincent/declearn_test/saved_models/new_models/synthseg_simple_lille.joblib'

# END INPUTS

from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
import joblib

reg = SGDRegressor(
  loss='epsilon_insensitive',
  epsilon=0,
  penalty='l2',
  fit_intercept=True,
  max_iter=1000,
  n_iter_no_change=1000,
  tol=None,
  eta0=sgd_lrate,
  learning_rate='invscaling'
)

model = Pipeline((
    ('preproc', make_pipeline(*preproc_steps)),
    ('reg', reg)
))

gSearch = GridSearchCV(
    estimator=model,
    param_grid={'reg__alpha': alpha_vals},
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    refit=True,
    cv=5,
    error_score='raise'
)

gSearch.fit(features, targets)
print(f"End training -> best alpha = {gSearch.best_params_['reg__alpha']}, corresponding MAE = {-gSearch.best_score_:.2f}")
joblib.dump(gSearch, dest_path)





