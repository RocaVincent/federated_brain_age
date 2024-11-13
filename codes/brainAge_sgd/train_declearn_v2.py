# INPUTS (features,targets,train_groups,sgd_alpha,sgd_initLR,preproc_steps,dest_dir)
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
df = pd.read_csv('/NAS/deathrow/vincent/declearn_test/data/stroke/new_exp/metadata.csv')
df = df[df.center=='CHU_Lille']
df2 = pd.read_csv('/NAS/deathrow/vincent/declearn_test/data/stroke/new_exp/cv_split_v2/train5.csv')
df = pd.concat((df,df2)).set_index(['sub_id','description'])

data = pd.read_csv('/NAS/deathrow/vincent/declearn_test/data/stroke/new_exp/synthseg_volumes.csv', index_col=['sub_id','description'])
cols = data.columns[1:]
#cols = np.loadtxt('/NAS/deathrow/vincent/declearn_test/data/stroke/new_exp/radiomics_list.txt', dtype=str)
data[cols] = data[cols].divide(data['total intracranial'], axis='index')
features = data.loc[df.index,cols].to_numpy()
targets = df.age.to_numpy()
train_groups = df.groupby('center').indices.items()

sgd_alpha = 1.2915496650148826e-09
sgd_initLR = 0.1
preproc_steps = [
    MinMaxScaler()#, PolynomialFeatures(degree=2, include_bias=False)
]
dest_dir = '/NAS/deathrow/vincent/declearn_test/saved_models/new_models/synthseg_simple_declearn_decay/model5'
#######

from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
import joblib
from declearn.model.sklearn import SklearnSGDModel
from declearn.optimizer import Optimizer
from declearn.optimizer.schedulers import LinearDecay
from declearn.dataset import InMemoryDataset
from declearn.utils import run_as_processes
from ..declearn_client_server.server import run_server
from ..declearn_client_server.client import run_client

BATCH_SIZE = 1
#POWER_T = 0.25 # The exponent for inverse scaling learning rate. see SGDRegressor doc
DROP_REMAINDER = False

sgd = SGDRegressor(
    loss='epsilon_insensitive',
    epsilon=0,
    penalty='l2',
    tol=None,
    alpha=sgd_alpha
)
sgd.intercept_ = 69.38261997405966
model = SklearnSGDModel(sgd)

# preproc data
preproc_pipeline = make_pipeline(*preproc_steps)
features = preproc_pipeline.fit_transform(features)
# end preproc


client_opt = Optimizer(
    lrate = LinearDecay(base=sgd_initLR, rate=0.0009, step_level=False)
)
server_opt = Optimizer(lrate=1.0)
server = (run_server, (dest_dir, len(train_groups), model, client_opt, server_opt, BATCH_SIZE, DROP_REMAINDER))

clients = []
for study_name,indices in train_groups:
    dataset = InMemoryDataset(data=features[indices], target=targets[indices])
    clients.append((run_client, (study_name, dataset, False)))
    
success, outp = run_as_processes(server, *clients)
if not success:
    exceptions = "\n".join(str(e) for e in outp if isinstance(e, RuntimeError))
    raise RuntimeError("Something went wrong during the demo. Exceptions caught:\n"+ exceptions)
    
joblib.dump(preproc_pipeline, dest_dir+'/preproc_pipeline.joblib')