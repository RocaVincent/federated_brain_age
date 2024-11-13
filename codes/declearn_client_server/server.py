from declearn.aggregator import AveragingAggregator
from declearn.communication import NetworkServerConfig
from declearn.main import FederatedServer
from declearn.main.config import FLOptimConfig, FLRunConfig


def run_server(dest_folder, nbClients, model, client_opt, server_opt, batch_size, drop_remainder, n_rounds=1000, eval_frequency=1, n_epochs=1,
               protocol='websockets', host='localhost', port=8765,
               certificate='/NAS/deathrow/vincent/declearn_test/declearn_certificates/server-cert.pem',
               private_key='/NAS/deathrow/vincent/declearn_test/declearn_certificates/server-pkey.pem'):
    aggregator = AveragingAggregator(steps_weighted=True)
    optim = FLOptimConfig(aggregator=aggregator, server_opt=server_opt, client_opt=client_opt)
    
    # Paramètres réseau
    network = NetworkServerConfig(
        protocol=protocol,
        host=host,
        port=port,
        certificate=certificate,
        private_key=private_key,
    )
    
    # Instanciation du serveur
    server = FederatedServer(
        model=model,
        netwk=network,
        optim=optim,
        checkpoint=dest_folder
    )
    
    # Configuration de l'exécution
    run_cfg = FLRunConfig.from_params(
        rounds=n_rounds,
        register={'min_clients': nbClients},
        training={'n_epoch':n_epochs, "batch_size": batch_size, "drop_remainder": drop_remainder},
        evaluate={'frequency':eval_frequency}
    )
    
    # Lancement du serveur
    server.run(run_cfg)