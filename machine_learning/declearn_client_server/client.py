from declearn.communication import NetworkClientConfig
from declearn.main import FederatedClient


def run_client(name, dataset, verbose, protocol='websockets', server_uri='wss://localhost:8765',
               certificate='declearn_certificates/ca-cert.pem'):
    network = NetworkClientConfig(
        protocol=protocol,
        server_uri=server_uri,
        certificate=certificate,
        name=name
    )
    client = FederatedClient(
        netwk=network,
        train_data=dataset,
        verbose=verbose
    )
    client.run()