from sklearn.linear_model import SGDRegressor
from declearn.model.sklearn import SklearnSGDModel
from declearn.main.config import FLOptimConfig, FLRunConfig
from declearn.communication import NetworkServerConfig
from declearn.main import FederatedServer
from declearn.test_utils import setup_server_argparse


def run_server(nbClients, certificate, private_key, protocol='websockets', host='localhost', port=8765):
    # Instancitation du modèle
    model = SGDRegressor(
        loss='squared_error',
        penalty='l2',
        alpha=5.78e-02, # fixé par GSCV, moyenne par étude
        tol=None,
    )
    model = SklearnSGDModel(model)
    
    # Configuration de l'optimisation
    aggregator = {
        "name": "averaging",
        "config": {"steps_weighted": True},
    }
    client_opt = {
        "lrate": 0.02,
        "modules": ["rmsprop"],
    }
    server_opt = {
        "lrate": 1.0,
        "modules": [("momentum", {"beta": 0.95})],
    }
    optim = FLOptimConfig.from_params(
        aggregator=aggregator,
        client_opt=client_opt,
        server_opt=server_opt,
    )
    
    # Paramètres réseau
    network = NetworkServerConfig(
        protocol='websockets',
        host='localhost',
        port=8765,
        certificate=certificate,
        private_key=private_key,
    )
    
    # Instanciation du serveur
    server = FederatedServer(
        model=model,
        netwk=network,
        optim=optim,
        checkpoint='results/server'
    )
    
    # Configuration de l'exécution
    run_cfg = FLRunConfig.from_params(
        rounds=1000,
        register={'min_clients': nbClients},
        training={"batch_size": 30, "drop_remainder": False},
        #evaluate={"batch_size": 50, "drop_remainder": False},
        #early_stop={"tolerance": 0.0, "patience": 5, "relative": False},
    )
    
    # Lancement du serveur
    server.run(run_cfg)
    
    
    
if __name__ == "__main__":
    # Parse command-line arguments.
    parser = setup_server_argparse(
        usage="Start a server to train a ridge regression model.",
        default_cert="server-cert.pem",
        default_pkey="server-pkey.pem",
    )
    parser.add_argument(
        "nbClients",
        type=int,
        help="number of clients",
        choices=range(1,11),
    )
    args = parser.parse_args()
    # Run the server routine.
    run_server(**args.__dict__)