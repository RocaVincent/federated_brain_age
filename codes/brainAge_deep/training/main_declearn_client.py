from sys import argv
import pandas as pd
from tensorflow.config import list_physical_devices, set_logical_device_configuration, LogicalDeviceConfiguration
from tensorflow.keras import mixed_precision
from declearn.model.tensorflow import TensorflowModel # required for declearn despite no usage in this script
from .input_pipeline.tf_dataset import declearn_set_from_pathList
from ...declearn_client_server.client import run_client

# COMMAND LINE INPUTS
center_name = argv[1]
#####################


# metadata
df_train = None
if center_name == 'CHU_Lille':
    df_train = pd.read_csv('/NAS/deathrow/vincent/declearn_test/data/stroke/new_exp/metadata.csv')
    df_train = df_train[df_train.center=='CHU_Lille']
else:
    df_train = pd.read_csv('/NAS/deathrow/vincent/declearn_test/data/stroke/new_exp/cv_split_v2/train5.csv')
    df_train = df_train[df_train.center==center_name]
assert len(df_train)>0
##########


# GPU mixed precision and memory
gpu = list_physical_devices('GPU')[0]
set_logical_device_configuration(gpu, [LogicalDeviceConfiguration(memory_limit=1000*9)])
mixed_precision.set_global_policy('mixed_float16')
#


# declearn dataset instantiation
mri_paths = df_train.apply(lambda row: f"/NAS/coolio/vincent/data/cohorte_AVC/n4brains_regNorm/{row.sub_id}_{row.description}.nii.gz", axis=1)
ages = df_train.age
dataset = declearn_set_from_pathList(
    mri_paths=mri_paths.tolist(),
    targets=ages.tolist()
)
###################


print(f"Starting client for center {center_name}, number of MR images = {len(mri_paths)}")
run_client(
    name=center_name,
    dataset=dataset,
    verbose=True
)

