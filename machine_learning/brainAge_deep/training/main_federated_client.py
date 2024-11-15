mri_paths = None # TO DEFINE -> paths of the preprocessed MR images in Niftti format
ages = None # TO DEFINE -> corresponding ages
center_name = None # TO DEFINE -> name of the client, used for the server traces

#### END INPUTS ######

from tensorflow.config import list_physical_devices, set_logical_device_configuration, LogicalDeviceConfiguration
from tensorflow.keras import mixed_precision
from declearn.model.tensorflow import TensorflowModel # required for declearn despite no usage in this script
from .input_pipeline.tf_dataset import declearn_set_from_pathList
from ...declearn_client_server.client import run_client

gpu = list_physical_devices('GPU')[0]
set_logical_device_configuration(gpu, [LogicalDeviceConfiguration(memory_limit=1000*9)])
mixed_precision.set_global_policy('mixed_float16')

dataset = declearn_set_from_pathList(
    mri_paths=mri_paths.tolist(),
    targets=ages.tolist()
)

print(f"Starting client for center {center_name}, number of MR images = {len(mri_paths)}")
run_client(
    name=center_name,
    dataset=dataset,
    verbose=True
)

