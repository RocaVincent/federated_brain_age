from declearn.test_utils import generate_ssl_certificates
from declearn.utils import run_as_processes

from tempfile import TemporaryDirectory

from client import run_client
from server import run_server

import pandas as pd

def run_demo():
    df = pd.read_csv('data/train_norm.csv')
    features = [c for c in df.columns if c.split('_')[0] in ('thick','subVol','grayVol')]
    groups = df.groupby('study').groups
    
    with TemporaryDirectory() as folder:
        ca_cert, sv_cert, sv_pkey = generate_ssl_certificates(folder)
        server = (run_server, (len(groups), sv_cert, sv_pkey))
        
        clients = [
            (run_client, (study_name, df.loc[indices,features], df.loc[indices,'age'], ca_cert))
            for study_name,indices in groups.items()
        ]
        success, outp = run_as_processes(server, *clients)
        if not success:
            exceptions = "\n".join(
                str(e) for e in outp if isinstance(e, RuntimeError)
            )
            raise RuntimeError(
                "Something went wrong during the demo. Exceptions caught:\n"
                + exceptions
            )
            
            
if __name__ == "__main__":
    run_demo()