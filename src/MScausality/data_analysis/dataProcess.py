
# from rpy2.robjects.packages import importr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


def format_sim_data(data):
      
    data = data.loc[:, ["Protein", "Feature", "Replicate", "Obs_Intensity"]]
    data = data.rename(columns={"Protein": "ProteinName", 
                                "Feature": "PeptideSequence",
                                "Replicate": "BioReplicate", 
                                "Obs_Intensity": "Intensity"})

    data.loc[:, "PrecursorCharge"] = 2
    data.loc[:, "FragmentIon"] = np.nan
    data.loc[:, "ProductCharge"] = np.nan
    data.loc[:, "IsotopeLabelType"] = "L"
    data.loc[:, "Condition"] = "Obs"
    data.loc[:, "Run"] = data.loc[:, "BioReplicate"]
    data.loc[:, "Fraction"] = 1
    data.loc[:, "PeptideSequence"]  = data.loc[:, "ProteinName"].astype(str) +\
        "_" + data.loc[:, "PeptideSequence"].astype(str)
    data.loc[:, "Run"] = data.loc[:, "Run"].astype(str) + "_" + \
      data.loc[:, "Condition"].astype(str)
    
    data.loc[:, "Feature"] = data.loc[:, "PeptideSequence"].astype(str) \
      + "_" + \
      data.loc[:, "PrecursorCharge"].astype(str) + "_" + \
      data.loc[:, "FragmentIon"].astype(str) + "_" + \
      data.loc[:, "ProductCharge"].astype(str) + "_"

    return data

def normalize_median(data):

    # Calculate ABUNDANCE_RUN (equivalent of ABUNDANCE_RUN in R)
    abundance_run = data.groupby(['Run', 'Fraction'])['Intensity'].median(
                  ).reset_index().rename(columns={'Intensity': 'ABUNDANCE_RUN'})

    # Calculate ABUNDANCE_FRACTION (equivalent of ABUNDANCE_FRACTION in R)
    abundance_fraction = abundance_run.groupby('Fraction'
                  )['ABUNDANCE_RUN'].median().reset_index().rename(
                      columns={'ABUNDANCE_RUN': 'ABUNDANCE_FRACTION'})

    # Adjust ABUNDANCE
    data = pd.merge(
        pd.merge(data, abundance_run, on=['Run', 'Fraction'], how='left'),
        abundance_fraction, on='Fraction', how='left')
    
    data['Intensity'] = data['Intensity'] - \
      data['ABUNDANCE_RUN'] + \
      data['ABUNDANCE_FRACTION']

    # Drop the temporary columns used for calculations
    data = data.drop(columns=['ABUNDANCE_RUN', 'ABUNDANCE_FRACTION'])

    # Log message (can use logging instead of print for production code)
    print("Normalization based on median: OK")
    
    return data

def topn_feature_selection(data, n):
      
    proteins = data["ProteinName"].unique()
    con_list = list()

    for i in range(len(proteins)):
         temp_data = data.loc[data["ProteinName"] == proteins[i]]
         
         # Calculate top features by highest mean intensity
         top_features = temp_data.groupby('Feature')['Intensity'].mean()
         top_features = top_features.sort_values(ascending=False).head(n)
         temp_data = temp_data[temp_data['Feature'].isin(top_features.index)]
         
         con_list.append(temp_data)
    
    data = pd.concat(con_list, ignore_index=True)
    return data

def imputation(data):
    """
    Imputation by linear model
    """
    if data["Intensity"].isna().mean() != 1:

        keep_runs = data['Intensity'].isna().groupby(
            data['Run']).mean()[data['Intensity'].isna().groupby(
            data['Run']).mean() != 1].index.values

        keep_feat = data['Intensity'].isna().groupby(
            data['Feature']).mean()[data['Intensity'].isna().groupby(
            data['Feature']).mean() != 1].index.values
        
        keep_data = data[(data["Run"].isin(keep_runs) & 
                       data["Feature"].isin(keep_feat))]
        na_data = data[~(data["Run"].isin(keep_runs) & 
                       data["Feature"].isin(keep_feat))]

        # Encoding 'Run' and 'Feature' as numerical values
        run_dummies = pd.get_dummies(keep_data['Run'])
        feature_dummies = pd.get_dummies(keep_data['Feature'])

        model_data = pd.concat([run_dummies, feature_dummies, 
                                keep_data["Intensity"]], axis=1)

        train_data = model_data[model_data['Intensity'].notna()]
        test_data = model_data[model_data['Intensity'].isna()]

        X_train = train_data[[i for i in train_data.columns if i != 'Intensity']]
        y_train = train_data['Intensity']

        # Prepare X for testing
        X_test = test_data[[i for i in test_data.columns if i != 'Intensity']]

        # Fit the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict missing values
        predicted_values = model.predict(X_test)

        # Fill missing values with the predictions
        model_data.loc[model_data['Intensity'].isna(), 
                    'Intensity'] = predicted_values

        keep_data = pd.concat([
             keep_data[[i for i in keep_data.columns if i != 'Intensity']], 
             model_data["Intensity"]], axis=1)
        data = pd.concat([keep_data, na_data], ignore_index=True)
        data.loc[:, "Intensity"] = np.where(data.loc[:, "Intensity"] > 40, 
                                            np.nan, 
                                            data.loc[:, "Intensity"])

    return data

def tukey_median_polish(data, eps = 0.01, maxiter=10, trace_iter=True, na_rm=True):

      z = copy.copy(data)
      nr = data.shape[0]
      nc = data.shape[1]
      t = 0
      oldsum = 0

      r = np.array([0 for _ in range(nr)])
      c = np.array([0 for _ in range(nc)])

      for _ in range(maxiter):
            rdelta = list()
            if na_rm:
                  for i in range(nr):
                        rdelta.append(np.nanmedian(z[i, :]))
            else:
                  for i in range(nr):
                        rdelta.append(np.median(z[i, :]))
            rdelta = np.array(rdelta)

            z = z - np.repeat(rdelta, nc, axis=0).reshape(nr, nc)
            r = r + rdelta
            if na_rm:
                  delta = np.nanmedian(c)
            else:
                  delta = np.median(c)
            c = c - delta
            t = t + delta

            cdelta = list()
            if na_rm:
                  for i in range(nc):
                        cdelta.append(np.nanmedian(z[:, i]))
            else:
                  for i in range(nc):
                        cdelta.append(np.median(z[:, i]))
            cdelta = np.array(cdelta)

            z = z - np.repeat(cdelta, nr, axis=0).reshape(nr, nc, order='F')
            c = c + cdelta

            if na_rm:
                  delta = np.nanmedian(r)
            else:
                  delta = np.median(r)

            r = r - delta
            t = t + delta

            if na_rm:
                  newsum = np.nansum(abs(z))
            else:
                  newsum = np.sum(abs(z))

            converged = (newsum == 0) | (abs(newsum - oldsum) < eps * newsum)
            if converged:
                  break
            oldsum = newsum
            # if trace_iter:
            #     print("{0}: {1}\n".format(str(iter), str(newsum)))

            ## TODO Add in converged info
            # if (converged) {
            # if (trace.iter)
            #   cat("Final: ", newsum, "\n", sep = "")
            # }
            # else warning(sprintf(ngettext(maxiter, "medpolish() did not converge in %d iteration",
            # "medpolish() did not converge in %d iterations"), maxiter),
            # domain = NA)

      ans = {"overall": t, "row": r, "col": c, "residuals": z}
      return ans

def summarize_data(data, summarization_method, MBimpute):
    """
    Summarize data by Tukey median polish
    """

    summarized_data = pd.DataFrame()
    proteins = data["ProteinName"].unique()

    for i in range(len(proteins)):
        protein = proteins[i]
        protein_data = data[data["ProteinName"] == protein]
        if MBimpute:
            protein_data = imputation(protein_data)

        protein_data = protein_data.pivot(index='Feature', 
                                          columns='RUN', 
                                          values='Intensity')

        if summarization_method == "TMP":
            protein_data = protein_data.to_numpy()
            tmp_data = tukey_median_polish(protein_data)
            tmp_data = tmp_data["overall"] + tmp_data["col"]
        elif summarization_method == "median":
            tmp_data = protein_data.median(axis=0, skipna=True).values
        elif summarization_method == "mean":
            tmp_data = protein_data.mean(axis=0, skipna=True).values
        
        summarized_data.loc[:, protein] = tmp_data
    return summarized_data

def dataProcess(data, 
                normalization="equalizeMedians", 
                feature_selection="All",
                n_features=3,
                summarization_method="TMP",
                MBimpute=True,
                sim_data=False):

    """
    Implementation of MSstats dataProcess function in Python.

    :param data:
    :param normalization:
    :param MBimpute:
    :return:
    """
    # TODO: Implement dataProcess function in Python
    #       - TopN feature selection
    #       - Imputation by accelerated failure time model

    if sim_data:
        data = format_sim_data(data)
    else:
        data["Intensity"] = np.log2(data["Intensity"])
        data.loc[:, "Feature"] = data.loc[:, "PeptideSequence"].astype(str) \
            + "_" + \
            data.loc[:, "PrecursorCharge"].astype(str) + "_" + \
            data.loc[:, "FragmentIon"].astype(str) + "_" + \
            data.loc[:, "ProductCharge"].astype(str) + "_"

    data.loc[:, "RUN"] = pd.factorize(data.loc[:, "Run"])[0]

    if normalization == "equalizeMedians":
        data = normalize_median(data)

    if feature_selection == "TopN":
        data = topn_feature_selection(data, n_features)

    summarized_data = summarize_data(data, summarization_method, MBimpute)

    return summarized_data


def main():

    from MScausality.simulation.example_graphs import signaling_network
    from MScausality.simulation.simulation import simulate_data

    # Test dataProcess function
    fd = signaling_network(add_independent_nodes=False)
    simulated_fd_data = simulate_data(fd['Networkx'], 
                                    coefficients=fd['Coefficients'], 
                                    mnar_missing_param=[-5, .4],
                                    add_feature_var=True, n=25, seed=3)
    fd_data = dataProcess(simulated_fd_data["Feature_data"], 
                          normalization=False, 
                          summarization_method="TMP", 
                          MBimpute=True, sim_data=True)

    fig, ax = plt.subplots()
    ax.scatter(fd_data["SOS"], fd_data["Ras"])
    plt.show()

if __name__ == "__main__":
    main()