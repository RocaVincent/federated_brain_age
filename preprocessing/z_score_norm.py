import pandas as pd
from multiprocessing import Pool
import nibabel as nib

df = pd.read_csv('/NAS/deathrow/vincent/declearn_test/data/stroke/data.csv')
df = df[~df.flair_raw.isna()]
mri_paths = df.apply(lambda row: f"/NAS/coolio/vincent/data/cohorte_AVC/n4brains_reg/{row.sub_id}_{row.description}.nii.gz", axis=1)
mask_paths = df.apply(lambda row: f"/NAS/coolio/vincent/data/cohorte_AVC/brainMasks_reg/{row.sub_id}_{row.description}.nii.gz", axis=1)
dest_paths = df.apply(lambda row: f"/NAS/coolio/vincent/data/cohorte_AVC/n4brains_regNorm/{row.sub_id}_{row.description}.nii.gz", axis=1)
N_PROCS = 20


def process_entry(mri_path, mask_path, dest_path):
    print(f"Process {mri_path}")
    mri = nib.load(mri_path)
    data = mri.get_fdata()
    mask = nib.load(mask_path).get_fdata()>0
    brain_intensities = data[mask]
    mean,std = brain_intensities.mean(), brain_intensities.std()
    data = (data-mean)/std
    mri = nib.Nifti1Image(data, affine=mri.affine, header=mri.header)
    nib.save(mri, dest_path)
    

pool = Pool(N_PROCS)

def raise_(e): raise e
for args in zip(mri_paths, mask_paths, dest_paths):
    pool.apply_async(process_entry, args=args, error_callback=raise_)

pool.close()
pool.join()