features = None # TO DEFINE -> array-like of shape (n_individuals,n_features)
targets = None # TO DEFINE -> array-like of shape (n_individuals,)
train_groups = None # TO DEFINE -> array-like of array-like of indices defining the different clients
sgd_alpha = None # TO DEFINE -> weight of the SGD regularization
sgd_lrate = None # TO DEFINE -> initial SGD learning rate
preproc_steps = None # TO DEFINE -> array-like of sklearn transformers corresponding to the preprocessing
dest_dir = None # TO DEFINE ->  directory path where the losses and the trained model will be saved

##### END INPUTS ########

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
DROP_REMAINDER = False

sgd = SGDRegressor(
    loss='epsilon_insensitive',
    epsilon=0,
    penalty='l2',
    tol=None,
    alpha=sgd_alpha
)
sgd.intercept_ = 69.38261997405966 # corresponds to the mean age in the CHU_Lille dataset
model = SklearnSGDModel(sgd)

# preproc data
preproc_pipeline = make_pipeline(*preproc_steps)
features = preproc_pipeline.fit_transform(features)
# end preproc

client_opt = Optimizer(
    lrate = LinearDecay(base=sgd_lrate, rate=0.0009, step_level=False)
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