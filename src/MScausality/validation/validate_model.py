
import numpy as np
import pandas as pd

def validate_model(model, data, source, target, group1, group2, parse_gene=False, gene_map=None):
    """
    Validate a model using the provided data.
    
    Parameters
    ----------
    model : MScausality.causal_model.LVM
        The model to validate. Must be trained using SVI.
    data : MSstats protein level results
        The data to use for validation.
    source : String protein (gene) name
        The protein to intervene on.
    target : String protein (gene) name
        The protein to measure the effect on.
    
    Returns
    -------
    dict
        A dictionary containing the results of the validation.
    """

    if parse_gene:
        data['Protein'] = data['Protein'].apply(lambda x: x.split("_")[0])

    if gene_map is not None:
        data = pd.merge(data, gene_map, how='left', left_on='Protein', right_on='From')
        data['Protein'] = data['To']

    # Determine what intervention to make
    # figure out how many standard deviations the fold change is
    p1_data = data.loc[data["Protein"] == source]
    true_perturb_std = determine_effect(p1_data, group1, group2)

    p2_data = data.loc[data["Protein"] == target]
    true_effect_std = determine_effect(p2_data, group1, group2)

    intervention_effect = model.input_data.loc[:, source].std() * true_perturb_std
    true_ace = model.input_data.loc[:, target].std() * true_effect_std

    # Make the intervention
    model.intervention(source, target, intervention_effect)

    # Compare to target
    pred_ace = model.intervention_samples.mean() - model.posterior_samples.mean()

    # Calculate the error
    error = np.abs(pred_ace - true_ace)

    return error


def determine_effect(data, g1, g2):

    mu1 = data.loc[data["GROUP"] == g1, "LogIntensities"].mean()
    mu2 = data.loc[data["GROUP"] == g2, "LogIntensities"].mean()
    mu = mu2 - mu1

    std1 = data.loc[data["GROUP"] == g1, "LogIntensities"].std()
    std2 = data.loc[data["GROUP"] == g2, "LogIntensities"].std()
    std = np.sqrt(std1**2 + std2**2)

    return mu / std