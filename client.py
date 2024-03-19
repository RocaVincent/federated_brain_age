from declearn.dataset import InMemoryDataset
from declearn.communication import NetworkClientConfig
from declearn.main import FederatedClient
from declearn.test_utils import setup_client_argparse

from random import randint
import pandas as pd

def run_client(name, features, targets, ca_cert, protocol='websockets', serv_uri='wss://localhost:8765', verbose=True):
    # Instanciation dataset d'entraînement
    ds_train = InMemoryDataset(data=features, target=targets, expose_classes=True)
    
    # paramètres réseau
    network = NetworkClientConfig(
        protocol=protocol,
        server_uri=serv_uri,
        certificate=ca_cert,
        name=name
    )
    
    # instanciation du client
    client = FederatedClient(
        netwk=network,
        train_data=ds_train,
        checkpoint=f'results/clients/{name}',
        verbose=verbose
    )
    client.run()
    

if __name__ == "__main__":
    # Parse command-line arguments.
    parser = setup_client_argparse(
        usage="Start a client for brain age",
        default_cert="ca-cert.pem",
    )
    parser.add_argument(
        "studyName",
        type=str,
        help="name of your client",
        choices=['CI2C', 'COBRE', 'CODE2', 'CPP', 'FBIRN', 'HCP', 'IXI', 'MCIC','NKI', 'NMORPH'],
    )
    args = parser.parse_args()
    # Run the client routine.
    
    df = pd.read_csv('../train.csv')
    df = df[df.study==args.studyName]
    features = [c for c in df.columns if c.split('_')[0] in ('thick','subVol','grayVol')]
    run_client(args.studyName, df[features].to_numpy(), df.age.to_numpy(), args.certificate, args.protocol, args.uri)