import pandas as pd
import numpy as np
import pymatgen.core as pym
from sklearn.ensemble import BaggingRegressor
import catboost as ctb
import pickle

from matminer.featurizers.composition import ElementProperty


def predict(np_core):
    np_shell = 'C6H5Na3O7'
    small = 10
    medium = 40
    large = 100
    small_nshell = inference(np_core=np_core, emic_size=small)
    small_wshell = inference(np_core=np_core, np_shell=np_shell, emic_size=small)
    medium_nshell = inference(np_core=np_core, emic_size=medium)
    medium_wshell = inference(np_core=np_core, np_shell=np_shell, emic_size=medium)
    large_nshell = inference(np_core=np_core, emic_size=large)
    large_wshell = inference(np_core=np_core, np_shell=np_shell, emic_size=large)
    return small_nshell, small_wshell, medium_nshell, medium_wshell, large_nshell, large_wshell


def add_sample(np_core, emic_x_size, emic_y_size, emic_z_size, r1, r2):
    pass


def inference(np_core, np_shell=None, emic_size=0):

    r1_cols = ['Fe', 'O', 'c', 'h', 'na', 'o', 'g0+', 'MagpieData avg_dev MeltingT',
        'MagpieData avg_dev NsValence', 'MagpieData range NdValence',
        'MagpieData avg_dev NdValence', 'MagpieData avg_dev NfValence',
        'MagpieData range NValence', 'MagpieData mean NsUnfilled',
        'MagpieData avg_dev NpUnfilled', 'MagpieData maximum NdUnfilled',
        'MagpieData range NdUnfilled', 'MagpieData mean NdUnfilled',
        'MagpieData avg_dev NdUnfilled', 'MagpieData maximum NUnfilled',
        'MagpieData mean NUnfilled', 'MagpieData mean GSvolume_pa',
        'MagpieData avg_dev GSvolume_pa', 'MagpieData mean GSbandgap',
        'MagpieData avg_dev GSbandgap', 'MagpieData mean GSmagmom',
        'MagpieData avg_dev GSmagmom']
    
    r2_cols = ['emic_size', 'Fe', 'O', 'o', 'fe', 'MagpieData maximum Number',
        'MagpieData minimum MendeleevNumber', 'MagpieData maximum AtomicWeight',
        'MagpieData range AtomicWeight', 'MagpieData avg_dev MeltingT',
        'MagpieData avg_dev NdValence', 'MagpieData maximum NfValence',
        'MagpieData mean NsUnfilled', 'MagpieData avg_dev NpUnfilled',
        'MagpieData avg_dev NdUnfilled', 'MagpieData maximum GSvolume_pa',
        'MagpieData range GSvolume_pa', 'MagpieData mean GSvolume_pa',
        'MagpieData avg_dev GSvolume_pa', 'MagpieData maximum GSbandgap',
        'MagpieData maximum GSmagmom']
    
    data = pd.DataFrame(dict(pym.Composition(np_core).as_dict()),index=[0])

    composition = pym.Composition(np_core)
    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    feat = ep_feat.featurize(composition)
    feat_df = pd.DataFrame([feat], columns=ep_feat.feature_labels())
    
    if np_shell:
        data_shell = pd.DataFrame(dict(pym.Composition(np_shell).as_dict()),index=[0])
        data_shell.columns = data_shell.columns.str.lower()

    r1_list = []
    r2_list = [emic_size]
    for col in ['Fe', 'O']:
        
        if col in data.columns:
            r1_list.append(data[col][0])
            r2_list.append(data[col][0])
        else:
            r1_list.append(0)
            r2_list.append(0)

    for col in ['c', 'h', 'na', 'o', 'g0+']:
        if np_shell:
            if col in data_shell.columns:
                r1_list.append(data_shell[col][0])
            else:
                r1_list.append(0)
        else:
            r1_list.append(0)
    
    for col in ['o', 'fe']:
        if np_shell:
            if col in data_shell.columns:
                r2_list.append(data_shell[col][0])
            else:
                r2_list.append(0)
        else:
            r2_list.append(0)


    r1_list = r1_list + list(feat_df[['MagpieData avg_dev MeltingT','MagpieData avg_dev NsValence', 'MagpieData range NdValence',
        'MagpieData avg_dev NdValence', 'MagpieData avg_dev NfValence',
        'MagpieData range NValence', 'MagpieData mean NsUnfilled',
        'MagpieData avg_dev NpUnfilled', 'MagpieData maximum NdUnfilled',
        'MagpieData range NdUnfilled', 'MagpieData mean NdUnfilled',
        'MagpieData avg_dev NdUnfilled', 'MagpieData maximum NUnfilled',
        'MagpieData mean NUnfilled', 'MagpieData mean GSvolume_pa',
        'MagpieData avg_dev GSvolume_pa', 'MagpieData mean GSbandgap',
        'MagpieData avg_dev GSbandgap', 'MagpieData mean GSmagmom',
        'MagpieData avg_dev GSmagmom']].values[0])
    
    r2_list = r2_list + list(feat_df[['MagpieData maximum Number',
        'MagpieData minimum MendeleevNumber', 'MagpieData maximum AtomicWeight',
        'MagpieData range AtomicWeight', 'MagpieData avg_dev MeltingT',
        'MagpieData avg_dev NdValence', 'MagpieData maximum NfValence',
        'MagpieData mean NsUnfilled', 'MagpieData avg_dev NpUnfilled',
        'MagpieData avg_dev NdUnfilled', 'MagpieData maximum GSvolume_pa',
        'MagpieData range GSvolume_pa', 'MagpieData mean GSvolume_pa',
        'MagpieData avg_dev GSvolume_pa', 'MagpieData maximum GSbandgap',
        'MagpieData maximum GSmagmom']].values[0])

    r1_df = pd.DataFrame(columns=r1_cols)
    r1_row = pd.Series(r1_list, index=r1_cols)
    r1_df = r1_df.append(r1_row, ignore_index=True)

    r2_df = pd.DataFrame(columns=r2_cols)
    r2_row = pd.Series(r2_list, index=r2_cols)
    r2_df = r2_df.append(r2_row, ignore_index=True)

    bag = pickle.load(open("bag.best", 'rb'))
    r1 = bag.predict(r1_df)

    catboost = ctb.CatBoostRegressor()
    catboost.load_model("catboost.best")
    r2 = catboost.predict(r2_df)

    ratio = float(r2) / float(r1)
    return round(ratio, 3)
