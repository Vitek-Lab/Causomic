import pandas as pd
import numpy as np

def normalize(df: pd.DataFrame,
              wide_format = False) -> pd.DataFrame:

    if wide_format:
        long_array = df.to_numpy().flatten()
        mean = np.nanmean(long_array)
        std = np.nanstd(long_array)
        df = df.applymap(lambda x: (x - mean) / std)
    else:
        mean = np.nanmean(df.loc[:, "Intensities"])
        std = np.nanstd(df.loc[:, "Intensities"])
        df.loc[:, "LogIntensities"] = (df.loc[:, "LogIntensities"] - mean) / std
    
    return {"df": df, "adj_metrics": {"mean": mean, "std": std}}

def main():
    import networkx as nx
    from MScausality.simulation.simulation import simulate_data

    graph = nx.DiGraph()
    graph.add_edge("STAT3", "MYC")

    sr_coef = {'STAT3': {'intercept': 15, "error": 1},
            'MYC': {'intercept': 2, "error": .25, 'STAT3': 1.}}
    
    obs_data = pd.DataFrame(simulate_data(graph, coefficients=sr_coef, 
                        add_feature_var=False, n=1000, seed=2)
                        ["Protein_data"])
    transformed_data = normalize(obs_data, wide_format=True)
if __name__ == "__main__":
    main()