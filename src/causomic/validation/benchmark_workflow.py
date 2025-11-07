# Create a benchmark workflow to validate the LVM implementation
# - Select drug for testing (troglitazone?)
# - Define start and end nodes (small network for quick testing)
# - Extract network from INDRA (test different sizes of evidence/confounders)
# - Reconcile with data (test with different weights on priors and AIC/BIC)
# - Add conditional independences fixing nodes
# - Fit LVM and perform intervention
# - Compare with differential analysis experimental results (define scoring metric)

# - Add TF

import argparse
import logging
import os
import pickle
import time
from collections import Counter
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd
import pyro
from indra.databases import uniprot_client
from indra_cogex.client import Neo4jClient
from pgmpy.estimators import ExpertKnowledge
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from y0.graph import NxMixedGraph

from causomic.causal_model.LVM import LVM
from causomic.graph_construction.prior_data_reconciliation import (
    AICGaussIndraPriors,
    BICGaussIndraPriors,
    SparseHillClimb,
    run_bootstrap,
)
from causomic.network import (
    estimate_posterior_dag,
    extract_indra_prior,
    repair_confounding,
)


def uniprot_to_hgnc_name(uniprot_mnemonic):
    """Get an HGNC ID from a UniProt mnemonic."""
    uniprot_id = uniprot_client.get_id_from_mnemonic(uniprot_mnemonic)
    if uniprot_id:
        return uniprot_client.get_gene_name(uniprot_id)
    else:
        return None


def reconcile_network(
    prior_network, data, prior_weight=1, criterion="bic", n_bootstrap=100, edge_probability=0.9
):

    # Prep data
    # Check for missing values in the data & impute if necessary
    if data.isnull().values.any():
        knn_imputer = KNNImputer(n_neighbors=5)

        input_data_imputed = pd.DataFrame(
            knn_imputer.fit_transform(data), index=data.index, columns=data.columns
        )
    else:
        print("No missing values in the data.")
    input_data_imputed.columns = input_data_imputed.columns.str.replace("-", "")

    # Define blacklist
    nodes = pd.unique(prior_network[["source_symbol", "target_symbol"]].values.ravel())
    nodes = np.array([node.replace("-", "") for node in nodes])

    all_possible_edges = [
        (u.replace("-", ""), v.replace("-", "")) for u in nodes for v in nodes if u != v
    ]
    obs_edges = [
        (
            prior_network.loc[i, "source_symbol"].replace("-", ""),
            prior_network.loc[i, "target_symbol"].replace("-", ""),
        )
        for i in range(len(prior_network))
    ]
    forbidden_edges = [edge for edge in all_possible_edges if edge not in obs_edges]

    expert_knowledge = ExpertKnowledge(forbidden_edges=forbidden_edges)

    if criterion == "aic":
        scoring_function = AICGaussIndraPriors
    elif criterion == "bic":
        scoring_function = BICGaussIndraPriors

    model_input = (
        input_data_imputed.loc[:, nodes],
        prior_network,
        prior_weight,
        scoring_function,
        SparseHillClimb,
        expert_knowledge,
    )

    bootstrap_dags = run_bootstrap(*model_input, n_bootstrap)

    edge_counts = Counter()
    for dag in bootstrap_dags:
        if dag is not None:
            # Update the edge counts
            edge_counts.update(list(dag.edges()))
    # Extract best edges
    n_dags = len(bootstrap_dags)

    edge_probabilities = pd.DataFrame(
        [(edge, count / n_dags) for edge, count in edge_counts.items()],
        columns=["Edge", "Probability"],
    )

    final_dag = edge_probabilities.loc[edge_probabilities["Probability"] > edge_probability, :]
    final_dag[["source", "target"]] = pd.DataFrame(
        final_dag["Edge"].tolist(), index=final_dag.index
    )
    final_dag = final_dag.drop("Edge", axis=1)

    final_dag = final_dag.reset_index(drop=True)

    return final_dag


def fit_causal_model(dag, data):

    # all_nodes = set(dag["source_symbol"]).union(set(dag["target_symbol"]))

    # final_graph = nx.DiGraph()
    # for i in range(len(dag)):

    #     final_graph.add_edge(dag.loc[i, "source_symbol"],
    #                          dag.loc[i, "target_symbol"])

    # # all_nodes=["BAX", "CYP2E1"]
    # obs_nodes = all_nodes

    # attrs = {node: (True if node not in obs_nodes and
    #                 node != "\\n" else False) for node in all_nodes}

    # nx.set_node_attributes(final_graph, attrs, name="hidden")
    # # Use y0 to build ADMG
    # dag = NxMixedGraph()
    # dag = dag.from_latent_variable_dag(final_graph, "hidden")

    # Appears to be computationally expensive
    pyro.clear_param_store()

    # Full imp Bayesian model
    lvm = LVM(backend="pyro", num_steps=2000, patience=50, initial_lr=0.01, verbose=True)

    data.columns = data.columns.str.replace("-", "")
    lvm.fit(data, dag)

    return lvm


def driz_report(y, yhat, p=1.0, tau=None):
    """
    Compute Direction-Aware Relative Improvement over Zero (DRIZ) with
    companion reporting statistics.

    Parameters
    ----------
    y, yhat : array-like
        Ground truth and predictions (same shape).
    p : float >= 1
        Error exponent (1 = linear, 2 = quadratic).
    tau : float or None
        Stabilizer; if None, set to 0.1 * IQR(|y|).

    Returns
    -------
    dict
        {
          "driz_mean": float,
          "driz_median": float,
          "sign_agreement": float,
          "median_rel_error": float,
          "mae": float,
          "rmse": float,
          "driz_per_sample": np.ndarray
        }
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)

    if tau is None:
        ay = np.abs(y)
        if ay.size > 0:
            q75, q25 = np.percentile(ay[~np.isnan(ay)], [75, 25])
            tau = 0.1 * (q75 - q25)
        if not np.isfinite(tau) or tau <= 0:
            tau = np.finfo(float).eps

    num = np.abs(y - yhat)
    den = np.abs(y) + tau
    ratio = np.clip(num / den, 0.0, 1e6)

    s = np.sign(y * yhat)  # +1 correct side, -1 wrong side, 0 if either is 0
    # driz_vals = dir_factor * (1.0 - np.power(ratio, p))
    # driz_vals = np.clip(driz_vals, -1.0, 1.0)  # safety bound
    driz_vals = 0.5 * s - 1.0 * np.log(ratio)

    # Companion metrics
    sign_agree = np.mean(np.sign(y) == np.sign(yhat))
    rel_error = np.median(num / (np.abs(y) + tau))
    mae = np.mean(num)
    rmse = np.sqrt(np.mean(num**2))

    return {
        "driz_mean": np.nanmean(driz_vals),
        "driz_median": np.nanmedian(driz_vals),
        "sign_agreement": sign_agree,
        "median_rel_error": rel_error,
        "mae": mae,
        "rmse": rmse,
        "driz_per_sample": driz_vals,
    }


def validate_model(model, intervention, target, ground_truth):

    try:
        target = [str(t).replace("-", "") for t in target]
    except:
        pass

    # Perform intervention
    model.intervention(intervention, target)

    intervention_samples = model.intervention_samples - model.posterior_samples

    # Calculate the mean of intervention_samples for each target
    mean_intervention = intervention_samples.mean()

    # Align ground_truth and mean_intervention by index (target names)
    common_targets = set(mean_intervention.index) & set(ground_truth.keys())
    mean_intervention = mean_intervention.loc[list(common_targets)]
    gt_values = pd.Series({k: ground_truth[k] for k in common_targets})

    # Align ground truth and predictions into a single DataFrame keyed by target
    comparison_df = pd.concat(
        [
            gt_values.rename("ground_truth"),
            mean_intervention.rename("mean_intervention"),
        ],
        axis=1,
    )

    # Drop rows where either value is missing (no basis for comparison)
    comparison_df = comparison_df.dropna(how="any").sort_index()

    # persist aligned comparison for later inspection
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_folder = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
    os.makedirs(data_folder, exist_ok=True)
    fname = f"validation_comparison_{timestamp}.csv"
    save_path = os.path.join(data_folder, fname)
    try:
        comparison_df.to_csv(save_path, index=True)
        print(f"Saved comparison_df to {save_path}")
    except Exception as e:
        print(f"Failed to save comparison_df: {e}")

    # Reassign aligned series so the existing return call uses the merged data
    gt_values = comparison_df["ground_truth"]
    mean_intervention = comparison_df["mean_intervention"]

    return driz_report(gt_values.values, mean_intervention.values)


def plot_network(dag):

    import matplotlib.pyplot as plt

    # Create directed graph from posterior_network
    G = nx.DiGraph()
    for _, row in dag.iterrows():
        source, target = row["Edge"]
        prob = row["Probability"]
        G.add_edge(source, target, weight=prob)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="lightblue", arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
    plt.title("Posterior Network")
    plt.show()


class SimpleLinearInterventionModel:
    def __init__(self, models, X_cols, X_filled, y_cols):
        self.models = models
        self.X_cols = X_cols
        self.X_filled = X_filled
        self.y_cols = y_cols
        # initialize sample DataFrames
        self.posterior_samples = pd.DataFrame(
            index=self.X_filled.index, columns=self.y_cols, dtype=float
        )
        self.intervention_samples = dict()
        # compute baseline predictions
        for y in self.y_cols:
            m = self.models.get(y)
            if m is None:
                self.posterior_samples[y] = np.nan
            else:
                self.posterior_samples[y] = m.predict(self.X_filled[self.X_cols].values)

    def intervention(self, intervention_dict, targets):
        # Build intervened X: copy baseline filled X and set intervened features
        X_int = np.array(list(intervention_dict.values()), dtype=float).reshape(1, -1)

        # Compute intervention predictions for requested targets (restrict to available y_cols)
        # targets = [t for t in targets if t in self.y_cols]
        for y in self.y_cols:
            m = self.models.get(y)
            if m is None:
                self.intervention_samples[y] = np.nan
            else:
                int_pred = m.predict(X_int)
                zero_pred = m.predict(np.array([[0] * len(self.X_cols)]).reshape(1, -1))
                self.intervention_samples[y] = int_pred - zero_pred


def compare_to_lm(input_data, drug_targets, dili_targets, intervention):

    # Prepare data copy and sanitize column names
    scm = input_data.copy()
    scm.columns = scm.columns.str.replace("-", "")

    # Determine available features and targets in the supplied data
    X_cols = [c for c in drug_targets if c in scm.columns]
    y_cols = [c for c in dili_targets if c in scm.columns]

    if len(X_cols) == 0 or len(y_cols) == 0:
        raise ValueError("No overlap between drug_targets/dili_targets and input_data columns.")

    # Simple imputation for predictors (column means) to allow prediction on all rows
    # KMeans-based imputation: cluster rows and fill missing X values with cluster centroids

    X = scm[X_cols].copy()
    # If there are no missing values, just use X
    if not X.isnull().values.any():
        X_filled = X
    else:
        # Initial fill for clustering (column means)
        init_X = X.fillna(X.mean())

        # Choose number of clusters based on data size (bounded)
        n_samples = init_X.shape[0]
        n_clusters = min(10, max(2, int(max(2, np.ceil(n_samples / 10)))))

        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        kmeans.fit(init_X.values)
        centroids = pd.DataFrame(
            kmeans.cluster_centers_, columns=init_X.columns, index=range(n_clusters)
        )

        X_filled = X.copy()
        col_means = init_X.mean()

        # For each row with missing values, assign to nearest centroid using only observed features
        for idx, row in X.iterrows():
            if row.isnull().any():
                obs_cols = row.index[~row.isnull()]
                if len(obs_cols) == 0:
                    # If a row has no observed features, fill with global column means
                    X_filled.loc[idx, :] = col_means
                    continue

                # Compute distances to centroids using only observed columns
                diffs = centroids[obs_cols].values - row[obs_cols].values
                dists = np.linalg.norm(diffs, axis=1)
                nearest = int(np.argmin(dists))
                # Fill missing entries from the centroid
                for c in row.index[row.isnull()].tolist():
                    X_filled.at[idx, c] = centroids.at[nearest, c]

        # As a final safeguard, fill any remaining NaNs (if any) with column means
        X_filled = X_filled.fillna(col_means)

    # Fit a separate linear model for each dili target
    models = {}
    for y in y_cols:
        # Build training set: drop rows with NaN in y or any X
        mask = scm[y].notna()
        mask &= scm[X_cols].notna().all(axis=1)
        if mask.sum() < 2:
            # Not enough data to fit; skip and store None
            models[y] = None
            continue
        lr = LinearRegression()
        lr.fit(scm.loc[mask, X_cols].values, scm.loc[mask, y].values)
        models[y] = lr

    # instantiate the model wrapper
    model = SimpleLinearInterventionModel(models, X_cols, X_filled, y_cols)
    model.intervention(intervention, dili_targets)

    return model.intervention_samples


def run_benchmark(
    one_step_evidence=3,
    two_step_evidence=3,
    three_step_evidence=6,
    confounder_evidence=10,
    prior_weight=1,
    criterion="bic",
    n_bootstrap=100,
    edge_probability=0.9,
    add_high_corr_edges_to_priors=False,
    corr_threshold=0.95,
    api_url=None,
    password=None,
    log_file=None,
):
    # Track start time
    start_time = time.time()

    # Setup logging
    logger = logging.getLogger("benchmark_logger")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    else:
        logger.addHandler(logging.StreamHandler())

    logger.info("Benchmark workflow started.")
    logger.info(f"Start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(
        f"Parameters: one_step_evidence={one_step_evidence}, two_step_evidence={two_step_evidence}, "
        f"three_step_evidence={three_step_evidence}, confounder_evidence={confounder_evidence}, "
        f"prior_weight={prior_weight}, criterion={criterion}, n_bootstrap={n_bootstrap}, "
        f"edge_probability={edge_probability}, add_high_corr_edges_to_priors={add_high_corr_edges_to_priors}, "
        f"corr_threshold={corr_threshold}"
    )

    # INDRA client
    client = Neo4jClient(
        url=api_url or os.getenv("API_URL"), auth=("neo4j", password or os.getenv("PASSWORD"))
    )

    trog_targets = ["SERPINE1", "CYP3A4", "CTNNB1", "MAPK1"]

    dili_targets = [
        "ABCC2",
        "ALB",
        "CAT",
        "CYP2C19",
        "CYP2C9",
        "CYP2E1",
        "ENO1",
        "GPT",
        "GSR",
        "GSTM1",
        "GSTT1",
        "HLA-A",
        "HMOX1",
        "HPD",
        "KNG1",
        "MTHFR",
        "NAT2",
        "SOD1",
    ]

    # INDRA client
    client = Neo4jClient(url=os.getenv("API_URL"), auth=("neo4j", os.getenv("PASSWORD")))

    # Load data & prep for model
    data_folder = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")

    input_data_path = os.path.join(data_folder, "model_input.csv")
    input_data = pd.read_csv(input_data_path)

    input_data_graph, input_data_scm = train_test_split(input_data, test_size=0.5, random_state=42)

    logger.info("Extracting network from INDRA...")
    step_start = time.time()
    indra_prior = extract_indra_prior(
        trog_targets,
        dili_targets,
        input_data_graph.columns,
        client,
        one_step_evidence=one_step_evidence,
        two_step_evidence=two_step_evidence,
        three_step_evidence=three_step_evidence,
        confounder_evidence=confounder_evidence,
    )
    step_end = time.time()
    logger.info(f"Network extraction completed in {step_end - step_start:.2f} seconds")

    logger.info("Reconciling network with data...")
    step_start = time.time()

    if criterion == "aic":
        scoring_function = AICGaussIndraPriors
    elif criterion == "bic":
        scoring_function = BICGaussIndraPriors

    # Filter input_data_graph to only include columns (nodes) present in indra_prior
    indra_nodes = pd.unique(indra_prior[["source_symbol", "target_symbol"]].values.ravel())
    indra_nodes = [node.replace("-", "") for node in indra_nodes]
    input_data_graph = input_data_graph.loc[
        :, input_data_graph.columns.str.replace("-", "").isin(indra_nodes)
    ]

    posterior_network = estimate_posterior_dag(
        input_data_graph,
        indra_prior,
        prior_weight,
        scoring_function,
        SparseHillClimb,
        n_bootstrap,
        add_high_corr_edges_to_priors,
        corr_threshold,
        edge_probability,
    )
    step_end = time.time()
    logger.info(f"Network reconciliation completed in {step_end - step_start:.2f} seconds")

    logger.info("Repairing confounding...")
    step_start = time.time()
    posterior_network = repair_confounding(
        input_data_graph, posterior_network, client, max_conditional=2
    )
    step_end = time.time()
    logger.info(f"Confounding repair completed in {step_end - step_start:.2f} seconds")

    logger.info("Fitting causal model...")
    step_start = time.time()
    model = fit_causal_model(posterior_network, input_data_scm)
    step_end = time.time()
    logger.info(f"Causal model fitting completed in {step_end - step_start:.2f} seconds")

    logger.info("Validating model...")
    step_start = time.time()
    gt_df = pd.read_csv(os.path.join(data_folder, "model.csv"))
    gt_df["Protein"] = gt_df["Protein"].apply(lambda x: uniprot_to_hgnc_name(x))

    gene_intervention = gt_df.loc[
        (gt_df["Protein"].isin(trog_targets)) & (gt_df["Label"] == "troglitazone_200 - DMSO_0"),
        ["Protein", "log2FC"],
    ]
    gene_intervention = dict(zip(gene_intervention["Protein"], gene_intervention["log2FC"]))

    outcome = gt_df.loc[
        (gt_df["Protein"].isin(dili_targets)) & (gt_df["Label"] == "troglitazone_200 - DMSO_0"),
        ["Protein", "log2FC"],
    ]
    outcome = dict(zip(outcome["Protein"], outcome["log2FC"]))

    # Save fitted model to a pickle for later inspection
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"fitted_model_{timestamp}.pkl"
    model_path = os.path.join(data_folder, model_filename)
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved fitted model to {model_path}")
    except Exception as e:
        logger.exception(f"Failed to save fitted model: {e}")
    validation_summary = validate_model(model, gene_intervention, dili_targets, outcome)

    gene_intervention = gt_df.loc[
        (gt_df["Protein"].isin(trog_targets)) & (gt_df["Label"] == "troglitazone_200 - DMSO_0"),
        ["Protein", "log2FC"],
    ]
    gene_intervention = dict(zip(gene_intervention["Protein"], gene_intervention["log2FC"]))

    step_end = time.time()
    logger.info(f"Model validation completed in {step_end - step_start:.2f} seconds")

    logger.info("Validation summary:")
    for k, v in validation_summary.items():
        if k != "driz_per_sample":
            logger.info(f"{k}: {v}")

    logger.info("LM summary:")
    lm_comp = compare_to_lm(input_data_scm, trog_targets, dili_targets, gene_intervention)
    gt_values = pd.Series({k: outcome[k] for k in lm_comp.keys()})

    # Align ground truth and predictions into a single DataFrame keyed by target
    comparison_df = pd.concat(
        [
            gt_values.rename("ground_truth"),
            pd.DataFrame(lm_comp).T.rename(columns={0: "mean_intervention"}),
        ],
        axis=1,
    )

    # Drop rows where either value is missing (no basis for comparison)
    comparison_df = comparison_df.dropna(how="any").sort_index()

    # # persist aligned comparison for later inspection
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # data_folder = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')
    # os.makedirs(data_folder, exist_ok=True)
    # fname = f"validation_lm_model.csv"
    # save_path = os.path.join(data_folder, fname)
    # try:
    #     comparison_df.to_csv(save_path, index=True)
    #     print(f"Saved comparison_df to {save_path}")
    # except Exception as e:
    #     print(f"Failed to save comparison_df: {e}")

    # Reassign aligned series so the existing return call uses the merged data
    gt_values = comparison_df["ground_truth"]
    lm_comp = comparison_df["mean_intervention"]

    lm_report = driz_report(gt_values.values, lm_comp.values)

    for k, v in lm_report.items():
        if k != "driz_per_sample":
            logger.info(f"{k}: {v}")

    # Calculate and log total execution time
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info(f"End time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(
        f"Total execution time: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f} (HH:MM:SS.ss)"
    )
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    logger.info("Benchmark workflow completed.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run benchmark workflow for LVM validation.")
    parser.add_argument("--one_step_evidence", type=int, default=1)
    parser.add_argument("--two_step_evidence", type=int, default=2)
    parser.add_argument("--three_step_evidence", type=int, default=10)
    parser.add_argument("--confounder_evidence", type=int, default=10000)
    parser.add_argument("--prior_weight", type=float, default=5)
    parser.add_argument("--criterion", type=str, choices=["aic", "bic"], default="bic")
    parser.add_argument("--n_bootstrap", type=int, default=100)
    parser.add_argument("--edge_probability", type=float, default=0.9)
    parser.add_argument("--add_high_corr_edges_to_priors", action="store_true")
    parser.add_argument("--corr_threshold", type=float, default=0.9)
    parser.add_argument("--api_url", type=str, default=None)
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument("--log_file", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    data_folder = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")

    log_file = args.log_file or os.path.join(
        data_folder, f"benchmark_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    run_benchmark(
        one_step_evidence=args.one_step_evidence,
        two_step_evidence=args.two_step_evidence,
        three_step_evidence=args.three_step_evidence,
        confounder_evidence=args.confounder_evidence,
        prior_weight=args.prior_weight,
        criterion=args.criterion,
        n_bootstrap=args.n_bootstrap,
        edge_probability=args.edge_probability,
        add_high_corr_edges_to_priors=False,
        api_url=args.api_url,
        password=args.password,
        log_file=log_file,
    )


# def main():

#     print("Starting benchmark workflow...")
#     # Calculated externally
#     trog_targets = ['SERPINE1', 'CYP3A4', 'CTNNB1', 'MAPK1']

#     dili_targets = ['ABCC2', 'ALB', 'CAT', 'CYP2C19', 'CYP2C9',
#                      'CYP2E1', 'ENO1', 'GPT', 'GSR', 'GSTM1',
#                      'GSTT1', 'HLA-A', 'HMOX1', 'HPD', 'KNG1',
#                      'MTHFR', 'NAT2', 'SOD1']

#     # INDRA client
#     client = Neo4jClient(
#         url=os.getenv("API_URL"),
#         auth=("neo4j", os.getenv("PASSWORD"))
#     )

#     # Load data & prep for model
#     data_folder = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')
#     input_data_path = os.path.join(data_folder, 'model_input.csv')
#     input_data = pd.read_csv(input_data_path)

#     # Randomly split input_data into two equal parts
#     input_data_graph, input_data_scm = train_test_split(
#         input_data, test_size=0.5, random_state=42)

#     print("Extracting network from INDRA...")
#     indra_prior = extract_network(
#         trog_targets, dili_targets, input_data_graph.columns, client,
#         one_step_evidence=3, two_step_evidence=3,
#         three_step_evidence=6, confounder_evidence=10)

#     print("Reconciling network with data...")
#     # Reconcile network with data
#     posterior_network = reconcile_network(
#         indra_prior, input_data_graph, prior_weight=1,
#         criterion='bic', n_bootstrap=100, edge_probability=.9)

#     print("Fitting causal model...")
#     model = fit_causal_model(posterior_network, input_data_scm)

#     print("Validating model...")
#     gt_df = pd.read_csv(os.path.join(data_folder, "model.csv"))
#     gt_df['Protein'] = gt_df['Protein'].apply(lambda x: uniprot_to_hgnc_name(x))

#     # Prepare data for intervention
#     intervention = gt_df.loc[(gt_df["Protein"].isin(trog_targets)) &
#                              (gt_df["Label"] == "troglitazone_200 - DMSO_0"),
#                              ["Protein", "log2FC"]]
#     intervention = dict(zip(intervention["Protein"], intervention["log2FC"]))

#     outcome = gt_df.loc[(gt_df["Protein"].isin(dili_targets)) &
#                         (gt_df["Label"] == "troglitazone_200 - DMSO_0"),
#                         ["Protein", "log2FC"]]
#     outcome = dict(zip(outcome["Protein"], outcome["log2FC"]))

#     validation_summary = validate_model(model, intervention, dili_targets, outcome)

#     print("Validation summary:")
#     for k, v in validation_summary.items():
#         if k != "driz_per_sample":
#             print(f"{k}: {v}")

if __name__ == "__main__":
    main()
