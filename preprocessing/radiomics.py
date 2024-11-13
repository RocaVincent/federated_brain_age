import pandas as pd
from radiomics.featureextractor import RadiomicsFeatureExtractor
import nibabel as nib
import numpy as np
from uuid import uuid4
from os import remove
from multiprocessing import Pool
from time import time

params = {
    'setting':{
        'normalize':False,
        'binWidth': 5,
        'label': 1, # TO CHECK
        'correctMask': True,
        'force2D': True,
        'force2Ddimension': 0, # TO CHECK
        'padDistance': None, # car pas de resampling
        'preCrop': False, # car pas de resampling
        'resampledPixelSpacing': None, # on a déjà resampled
        'interpolator': None, # on a déjà resampled
        'weightingNorm': None,
        'geometryTolerance': 0.001
    },
    'imageType':{
        'Original':{},
        'LoG': {'sigma': [1.0, 2.0, 3.0]},
        'LBP2D':{
            'force2Ddimension':0,
            'lbp2DRadius':1,
            'lbp2DSamples':9,
            'lbp2DMethod':'uniform'
        },
        'Wavelet':{
            'binWidth':5,
            'level':3
        }
    },
    'featureClass':{
        'shape':None,
        'firstorder':[],
        'glcm':[
            'Autocorrelation', 'JointAverage', 'ClusterProminence', 'ClusterShade', 'ClusterTendency', 'Contrast', 'Correlation', 'DifferenceAverage',
            'DifferenceEntropy', 'DifferenceVariance', 'JointEnergy', 'JointEntropy', 'Imc1', 'Imc2', 'Idm', 'Idmn', 'Id', 'Idn', 'InverseVariance',
            'MaximumProbability', 'SumEntropy', 'SumSquares'
        ],
        'glrlm':None,
        'glszm':None,
        'gldm':None,
        'ngtdm':None
    }
}

df = pd.read_csv('/NAS/deathrow/vincent/declearn_test/data/stroke/data.csv')
df = df[~df.flair_raw.isna()]
mri_paths = df.apply(lambda row: f"/NAS/coolio/vincent/data/cohorte_AVC/n4brains_regNorm/{row.sub_id}_{row.description}.nii.gz", axis=1)
seg_paths = df.apply(lambda row: f"/NAS/coolio/vincent/data/cohorte_AVC/synthseg_reg/{row.sub_id}_{row.description}.nii.gz", axis=1)
id_dicts = df[['sub_id','description']].to_dict('records')
DEST_PATH = '/NAS/deathrow/vincent/declearn_test/data/stroke/radiomics_wm_v3.csv'
N_PROCS = 23
LABELS = 2,41

# fin paramétrage
t_start = time()
feature_extractor = RadiomicsFeatureExtractor(params)

def get_radiomics(mri_path, seg_path, id_dict):
    print(f'Traitement {id_dict}')
    
    mri = nib.load(seg_path)
    seg = np.where(np.isin(mri.get_fdata(), LABELS), 1, 0)
    mri = nib.Nifti1Image(seg, affine=mri.affine, header=mri.header)
    seg_path2 = uuid4().hex+'.nii'
    nib.save(mri, seg_path2)
    
    features = feature_extractor.execute(mri_path, seg_path2)
    id_dict.update(features)
    remove(seg_path2)
    return id_dict


pool = Pool(N_PROCS)
dicts = pool.starmap(get_radiomics, zip(mri_paths,seg_paths,id_dicts))
pool.close()

pd.DataFrame(dicts).to_csv(DEST_PATH, index=False)
print(f"Fin traitement, tps d'execution en secondes : {time()-t_start:.3f}")