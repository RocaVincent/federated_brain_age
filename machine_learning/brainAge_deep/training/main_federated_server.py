from sys import argv
from declearn.optimizer import Optimizer
from declearn.optimizer.schedulers import LinearDecay
from ..model_architecture import Regressor
from ...declearn_client_server.server import run_server


# COMMAND LINE INPUTS
dest_folder = argv[1]
nbClients = int(argv[2])
#####################

model = Regressor(declearn=True, last_bias_value=69.38261997405966) # bias corresponds to Lille mean

client_opt = Optimizer(
    lrate = LinearDecay(base=0.0005, rate=0.0018, step_level=False)
)
server_opt = Optimizer(lrate=1.0)

BATCH_SIZE = 8
DROP_REMAINDER = True

run_server(
    dest_folder=dest_folder,
    nbClients=nbClients,
    model=model,
    client_opt=client_opt,
    server_opt=server_opt,
    batch_size=BATCH_SIZE,
    drop_remainder=DROP_REMAINDER,
    n_rounds=500,
    eval_frequency=10
)