
from rpy2.robjects.packages import importr

def dataProcess(data, normalization="equalizeMedians", MBimpute=True):

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

    processed_data = dataProcess(data, normalization="equalizeMedians", MBimpute=TRUE)
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

    # base = importr('base')
    # base.source("http://www.bioconductor.org/biocLite.R")
    # biocinstaller = importr("BiocInstaller")
    # biocinstaller.biocLite("MSstats")
    #
    # # load the installed package "MSstats"
    # seqlogo = importr("MSstats")

    pass