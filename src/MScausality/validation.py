
import pandas as pd

from MScausality.causal_model.LVM import LVM
from MScausality.simulation.simulation import simulate_data
from MScausality.data_analysis.normalization import normalize

import pyro

from eliater.regression import summary_statistics
from y0.dsl import Variable
from sklearn.impute import KNNImputer

def validate_model(data,
                   bulk_graph, 
                   y0_graph_bulk, 
                   msscausality_graph,
                   coef, 
                   int1, 
                   int2, 
                   outcome,
                   priors=None):
    """
    Validate a model using the provided data.
    
    Parameters
    ----------
    
    Returns
    -------

    """

    gt_effect = gt_ate(bulk_graph, coef, int1, int2, outcome)
    eliator_effect = eliator_ate(y0_graph_bulk, data, int1, int2, outcome)
    mscausality_effect = mscausality_ate(msscausality_graph, 
                                         data, int1, int2, 
                                         outcome, priors)
    
    result_df = pd.DataFrame({
        "Ground_truth": [gt_effect],
        "Eliator": [eliator_effect],
        "MScausality": [mscausality_effect],
    })

    return result_df

def gt_ate(bulk_graph, coef, int1, int2, outcome):

    """
    Calculate the ground truth effect of two interventions.

    Parameters
    ----------
    bulk_graph : networkx.DiGraph
        The graph to use for the comparison.
    coef : dict
        The coefficients of the graph.
    int1 : dict
        The first intervention to compare.
    int2 : dict
        The second intervention to compare.
    outcome : str
        The outcome to measure the effect.
    """
    
    # Ground truth
    intervention_low = simulate_data(bulk_graph, coefficients=coef,
                                    intervention=int1, 
                                    add_feature_var=False, n=10000, seed=2)

    intervention_high = simulate_data(bulk_graph, coefficients=coef,
                                    intervention=int2, 
                                    add_feature_var=False, n=10000, seed=2)

    gt_ate = (intervention_high["Protein_data"][outcome].mean() \
          - intervention_low["Protein_data"][outcome].mean())
    
    return gt_ate

def eliator_ate(y0_graph_bulk, data, int1, int2, outcome):

    """
    Calculate the Eliator estimated effect of two interventions.

    """

    # Eliator prediction
    obs_data_eliator = data.copy()

    if data.isnull().values.any():
        imputer = KNNImputer(n_neighbors=3, keep_empty_features=True)
        # Impute missing values (the result is a NumPy array, so we need to convert it back to a DataFrame)
        obs_data_eliator = pd.DataFrame(imputer.fit_transform(obs_data_eliator), columns=data.columns)

    eliator_int_low = summary_statistics(
        y0_graph_bulk, obs_data_eliator,
        treatments={Variable(list(int1.keys())[0])},
        outcome=Variable(outcome),
        interventions={
            Variable(list(int1.keys())[0]): list(int1.values())[0]})

    eliator_int_high = summary_statistics(
        y0_graph_bulk, obs_data_eliator,
        treatments={Variable(list(int2.keys())[0])},
        outcome=Variable(outcome),
        interventions={
            Variable(list(int2.keys())[0]): list(int2.values())[0]})
    
    eliator_ate = eliator_int_high.mean - eliator_int_low.mean

    return eliator_ate

def mscausality_ate(msscausality_graph, data, int1, int2, outcome, priors):
    """
    Compare the results of two interventions using MScausality.

    Parameters
    ----------
    model : MScausality.causal_model.LVM
        The model to use for the comparison.
    int1 : dict
        The first intervention to compare.
    int2 : dict
        The second intervention to compare.
    outcome : str
        The outcome to measure the effect on.
    priors : dict
        Optional priors to use for the model.
    """
    
    # Basic Bayesian model
    pyro.clear_param_store()
    transformed_data = normalize(data, wide_format=True)
    input_data = transformed_data["df"]
    scale_metrics = transformed_data["adj_metrics"]

    # Full imp Bayesian model
    lvm = LVM(backend="numpyro", informative_priors=priors)
    lvm.fit(input_data, msscausality_graph)

    ## MScausality results
    lvm.intervention({list(int1.keys())[0]: (list(int1.values())[0] \
                                            - scale_metrics["mean"]) \
                                                / scale_metrics["std"]}, outcome)
    mscausality_int_low = lvm.intervention_samples
    lvm.intervention({list(int2.keys())[0]: (list(int2.values())[0] \
                                            - scale_metrics["mean"]) \
                                                / scale_metrics["std"]}, outcome)
    mscausality_int_high = lvm.intervention_samples

    mscausality_int_low = ((mscausality_int_low*scale_metrics["std"]) \
                        + scale_metrics["mean"])
    mscausality_int_high = ((mscausality_int_high*scale_metrics["std"]) \
                            + scale_metrics["mean"])
    mscausality_ate = mscausality_int_high.mean() - mscausality_int_low.mean()

    return mscausality_ate.item()


def main():

    from MScausality.simulation.simulation import simulate_data
    from MScausality.data_analysis.dataProcess import dataProcess
    from MScausality.simulation.example_graphs import signaling_network

    sn = signaling_network()

    data = simulate_data(
        sn["Networkx"], 
        coefficients=sn["Coefficients"], 
        mnar_missing_param=[-4, .3],
        add_feature_var=True, 
        n=30, 
        seed=200)
    data["Feature_data"]["Obs_Intensity"] = data["Feature_data"]["Intensity"]

    summarized_data = dataProcess(
        data["Feature_data"], 
        normalization=False, 
        feature_selection="All",
        summarization_method="TMP",
        MBimpute=False,
        sim_data=True)
    
    summarized_data = summarized_data.loc[:, [
        i for i in summarized_data.columns if i not in ["IGF", "EGF"]]]

    result = validate_model(
        summarized_data,
        sn["Networkx"], sn["y0"], sn["MScausality"],
        sn["Coefficients"], {"Ras": 5}, {"Ras": 7}, "Erk")
    
    print(result)


if __name__ == "__main__":
    main()