
# from rpy2.robjects.packages import importr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy

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

def tukey_median_polish(data, eps = 0.01, maxiter=10, trace_iter=True, na_rm=True):

      z = copy.copy(data)
      nr = data.shape[0]
      nc = data.shape[1]
      t = 0
      oldsum = 0

      r = np.array([0 for _ in range(nr)])
      c = np.array([0 for _ in range(nc)])

      for iter in range(maxiter):
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

def summarize_data(data):
    """
    Summarize data by Tukey median polish
    """
    data.loc[:, "Feature"] = data.loc[:, "PeptideSequence"].astype(str) + "_" + \
      data.loc[:, "PrecursorCharge"].astype(str) + "_" + \
      data.loc[:, "FragmentIon"].astype(str) + "_" + \
      data.loc[:, "ProductCharge"].astype(str) + "_"

    summarized_data = pd.DataFrame()
    proteins = data["ProteinName"].unique()

    for i in range(len(proteins)):
        protein = proteins[i]
        protein_data = data[data["ProteinName"] == protein]
        protein_data = protein_data.pivot(index='Feature', 
                                          columns='Run', 
                                          values='Intensity')
      #   protein_data = protein_data.fillna(0)
        protein_data = protein_data.to_numpy()
        tmp_data = tukey_median_polish(protein_data)
        tmp_data = tmp_data["overall"] + tmp_data["col"]
        
        summarized_data.loc[:, protein] = tmp_data
    return summarized_data

def dataProcess(data, normalization="equalizeMedians", MBimpute=True,
                sim_data=False):

    """
    Implementation of MSstats dataProcess function in Python.

    Currently unavailable. To convert process simulated data first save the feature level data to a csv and then run
    the following code in R:

    library(MSstats)
    library(tidyverse)

    data = read.csv("sim_feature_level_data.csv")
    data = data %>% select(Protein, Feature, Replicate, Obs_Intensity)
    data = data %>% rename(ProteinName=Protein, PeptideSequence=Feature,
                           BioReplicate=Replicate, Intensity=Obs_Intensity)

    data$PrecursorCharge = 2
    data$FragmentIon = NA
    data$ProductCharge= NA
    data$IsotopeLabelType = "L"
    data$Condition = "Obs"
    data$Run = data$BioReplicate
    data$Fraction = 1
    data$PeptideSequence  = paste(data$ProteinName, data$PeptideSequence , sep="_")
    data$Run = paste(data$Run, data$Condition, sep="_")
    data$Intensity = 2**data$Intensity

    processed_data = dataProcess(data, normalization=FALSE, MBimpute=TRUE)
    processed_data$ProteinLevelData %>% select(Protein, originalRUN, LogIntensities) %>% pivot_wider(
          names_from = Protein,
          values_from = c(LogIntensities)
          ) %>%
          write.csv("protein_data.csv", row.names=FALSE)

    :param data:
    :param normalization:
    :param MBimpute:
    :return:
    """
    # TODO: Implement dataProcess function in Python
    #       - Normalization by equalizing medians across runs
    #       - Imputation by accelerated failure time model
    #       - Summarization by tukey median polish

    if sim_data:
        data = format_sim_data(data)
    else:
        data["Intensity"] = np.log2(data["Intensity"])

    if normalization == "equalizeMedians":
        data = normalize_median(data)
    
    print("TODO: imputation")
    data = summarize_data(data)

    return data


def main():
    # Test dataProcess function
    data = pd.read_csv("data/methods_paper_data/tf_sim/simple_regression_feature_data.csv")
    test = dataProcess(data, sim_data=True)
    print(test)

    fig, ax = plt.subplots()
    ax.scatter(test["STAT3"], test["MYC"])
    plt.show()

if __name__ == "__main__":
    main()
