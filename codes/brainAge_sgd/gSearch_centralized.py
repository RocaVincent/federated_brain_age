# Définition données et modèles
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

reg = SGDRegressor(
  loss='epsilon_insensitive',
  epsilon=0,
  penalty='l2',
  fit_intercept=True,
  max_iter=1000,
  n_iter_no_change=1000,
  tol=None,
  eta0=0.004,
  learning_rate='invscaling'
)

model = Pipeline((
    ('minMax', MinMaxScaler()),
    #('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('reg', reg)
))

alpha_vals = np.logspace(-20,0, num=10)
#param_dicts = [{'reg__alpha':alpha} for alpha in alpha_vals]
param_dicts = {'reg__alpha':alpha_vals}

meta = pd.read_csv('/NAS/deathrow/vincent/declearn_test/data/stroke/new_exp/metadata.csv', index_col=['sub_id','description'])
meta = meta[meta.center=='CHU_Lille']
features = pd.read_csv('/NAS/deathrow/vincent/declearn_test/data/stroke/new_exp/radiomics_wm.csv', index_col=['sub_id','description'])
#cols = features.columns[1:]
cols = np.loadtxt('/NAS/deathrow/vincent/declearn_test/data/stroke/new_exp/radiomics_list.txt', dtype=str)
#features[cols] = features[cols].divide(features['total intracranial'], axis='index')
features = features.loc[meta.index,cols].to_numpy()
targets = meta.age.to_numpy()
N_JOBS=15

# fin paramétrage

gSearch = GridSearchCV(
    estimator=model,
    param_grid=param_dicts,
    scoring='neg_mean_absolute_error',
    n_jobs=N_JOBS,
    refit=False,
    cv=5,
    error_score='raise'
)

gSearch.fit(features, targets)
print(f"End of training, best params = {gSearch.best_params_}, corresponding MAE = {-gSearch.best_score_:.2f}")