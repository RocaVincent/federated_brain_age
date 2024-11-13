import numpy as np

# Définition données et modèles
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
import pandas as pd

reg = SGDRegressor(
  loss='epsilon_insensitive',
  epsilon=0,
  penalty='l2',
  fit_intercept=True,
  max_iter=1000,
  n_iter_no_change=1000,
  tol=None,
  eta0=0.07,
  learning_rate='invscaling'
)

model = Pipeline((
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('reg', reg),
))

#alpha_vals = np.logspace(-6,2,num=20)
alpha_vals = np.logspace(-20,0, num=10)
params_dicts = [{'reg__alpha':alpha} for alpha in alpha_vals]

splits = []
for i in range(1,6):
    train_meta = pd.read_csv(f"/NAS/deathrow/vincent/declearn_test/data/stroke/new_exp/crossVal_hpTuning/metadata/train{i}.csv", index_col=['sub_id','description'])
    train = pd.read_csv(f"/NAS/deathrow/vincent/declearn_test/data/stroke/new_exp/crossVal_hpTuning/synthseg_vols/train{i}.csv", index_col=['sub_id','description'])
    val_meta = pd.read_csv(f"/NAS/deathrow/vincent/declearn_test/data/stroke/new_exp/crossVal_hpTuning/metadata/val{i}.csv", index_col=['sub_id','description'])
    val = pd.read_csv(f"/NAS/deathrow/vincent/declearn_test/data/stroke/new_exp/crossVal_hpTuning/synthseg_vols/val{i}.csv", index_col=['sub_id','description'])
    train_features = train.loc[train_meta.index].to_numpy()
    train_targets = train_meta.age.to_numpy()
    val_features = val.loc[val_meta.index].to_numpy()
    val_targets = val_meta.age.to_numpy()
    splits.append((train_features,train_targets,val_features,val_targets))

N_PROCS = 15

# fin paramétrage

from multiprocessing import Pool
from sklearn.base import clone

def get_mae(params_dict):
    print(f"Début traitement {params_dict}")
    m = clone(model)
    m.set_params(**params_dict)
    maes_fold = np.empty(len(splits))
    for i,(train_features,train_targets,val_features,val_targets) in enumerate(splits):
        m.fit(train_features,train_targets)
        preds_val = m.predict(val_features)
        maes_fold[i] = abs(preds_val-val_targets).mean()
    return maes_fold.mean()
    
pool = Pool(N_PROCS)
maes_val = pool.map(get_mae, params_dicts)
print(maes_val)
idxMin = np.argmin(maes_val)
print(f"End of training, best params = {params_dicts[idxMin]}, corresponding MAE = {maes_val[idxMin]:.2f}")