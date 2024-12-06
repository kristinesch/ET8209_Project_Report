import pandas as pd
import numpy as np

def extract_results(x):
    return pd.DataFrame.from_dict(x.extract_values(), orient='index', columns=[str(x)])

def convert_to_array(df, stage = 2):
    arr=np.zeros((5,24))
    for o, t in df.index:
        arr[o][t-24*(stage-1)-1]+=df[o,t]
    return arr

def save_results(model, case):
    prod1_df = extract_results(model.p_1)
    prod2_df = extract_results(model.p_2)
    price_df = extract_results(model.rho)
    vol_df = extract_results(model.v)
    q_df = extract_results(model.q)

    prod1_df.to_pickle("prod1_df_"+case+".pkl")
    prod2_df.to_pickle("prod2_df_"+case+".pkl")
    price_df.to_pickle("price_df_"+case+".pkl")
    vol_df.to_pickle("vol_df_"+case+".pkl")
    q_df.to_pickle("q_df_"+case+".pkl")

def read_results(case):
    prod1_df = pd.read_pickle("prod1_df_" + case+".pkl")
    prod2_df = pd.read_pickle("prod2_df_" + case+".pkl")
    price_df = pd.read_pickle("price_df_" + case+".pkl")
    vol_df = pd.read_pickle("vol_df_" + case+".pkl")
    q_df = pd.read_pickle("q_df_" + case+".pkl")
    return prod1_df, prod2_df, price_df, vol_df, q_df


