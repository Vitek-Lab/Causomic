
import pandas as pd
import numpy as np
import statsmodels.api as sm

import torch

def calc_dpc(df: pd.DataFrame) -> float:

    """
    Calculate the detection probability curve (DPC) for a given dataset.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame containing the data in wide format (Proteins x Runs).
    
    Returns
    -------
    float
        The slope of the DPC.
    """

    vals = pd.DataFrame(
        {"missing": 1-df.isnull().mean(),
            "mean": df.mean()})

    X = sm.add_constant(vals["mean"])
    model = sm.GLM(vals["missing"], X, 
                family=sm.families.Binomial(link=sm.families.links.logit()))
    result = model.fit()
    
    return result.params["mean"]

def prep_data_for_model(root_nodes, 
                        descendent_nodes, 
                        input_data, 
                        input_missing):

    """
    Prepare input data to feed into model.

    Parameters
    ----------
    root_nodes : list
        A list of root nodes in the causal graph.
    descendent_nodes : list
        A list of descendent nodes in the causal graph.
    input_data : pd.DataFrame
        A pandas DataFrame containing the data in wide format (Proteins x Runs).
    input_missing : pd.DataFrame
        A pandas DataFrame containing missing info in wide format 
        (Proteins x Runs).
    
    Returns
    -------
    dict
        A dictionary containing the data to be fed into the model.
    """
    
    condition_data = dict()
    for node in root_nodes:
        if "latent" not in node:
            condition_data[f"obs_{node}"] = torch.tensor(np.nan_to_num(
                input_data.loc[:, node].values))
            condition_data[f"missing_{node}"] = torch.tensor(
                input_missing.loc[:, node].values)

    for node in descendent_nodes:
        condition_data[f"obs_{node}"] = torch.tensor(
            np.nan_to_num(input_data.loc[:, node].values))
        condition_data[f"missing_{node}"] = torch.tensor(
            input_missing.loc[:, node].values)

    return condition_data