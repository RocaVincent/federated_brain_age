from declearn.communication import NetworkClientConfig
from declearn.main import FederatedClient


## ATTENTION : peut-être nécessaire d'ajouter les imports de dépendance client ici (e.g. declearn.model.tensorflow)

def run_client(name, dataset, verbose, protocol='websockets', server_uri='wss://localhost:8765',
               certificate='/NAS/deathrow/vincent/declearn_test/declearn_certificates/ca-cert.pem'):
    # paramètres réseau
    network = NetworkClientConfig(
        protocol=protocol,
        server_uri=server_uri,
        certificate=certificate,
        name=name
    )
    
    # instanciation du client
    client = FederatedClient(
        netwk=network,
        train_data=dataset,
        verbose=verbose
    )
    client.run()