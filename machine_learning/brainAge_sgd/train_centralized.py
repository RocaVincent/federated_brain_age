features = None # TO DEFINE -> array-like of shape (n_individuals,n_features)
targets = None # TO DEFINE -> array-like of shape (n_individuals,)
sgd_lrate = None # TO DEFINE -> initial SGD learning rate
preproc_steps = None # TO DEFINE -> array-like of sklearn transformers corresponding to the preprocessing
dest_path = None # TO DEFINE -> joblib path where the model will be saved in joblib format

##### END INPUTS ########

from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np

alpha_vals = np.logspace(-20,0, num=10)

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





