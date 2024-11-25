
import pyro
from pyro.nn import PyroModule
import pyro.distributions as pyro_dist
import pyro.poutine as poutine

from chirho.interventional.handlers import do

import numpyro
import numpyro.distributions as numpyro_dist
from jax import numpy as jnp

class ProteomicPerturbationModel(PyroModule):
    def __init__(self, n_obs, root_nodes, downstream_nodes):
        
        super().__init__()
        self.n_obs = n_obs
        self.root_nodes = root_nodes
        self.downstream_nodes = downstream_nodes

    def forward(self, data, priors, dpc_slope=1):

        # Define objects that will store the coefficients
        downstream_coef_dict_mean = dict()
        downstream_coef_dict_scale = dict()
        root_coef_dict_mean = dict()
        root_coef_dict_scale = dict()

        # Initial priors for coefficients
        for node_name, items in self.downstream_nodes.items():

            downstream_coef_dict_mean[f"{node_name}_int"] = pyro.sample(
                f"{node_name}_int", pyro_dist.Normal(
                    priors[node_name][f"{node_name}_int"], 
                    priors[node_name][f"{node_name}_scale"])
                    )
            
            for item in items:

                downstream_coef_dict_mean[f"{node_name}_{item}_coef"
                                          ] = pyro.sample(
                    f"{node_name}_{item}_coef", 
                    pyro_dist.Normal(
                        priors[node_name][f"{node_name}_{item}_coef"], 
                        priors[node_name][f"{node_name}_{item}_scale"])
                    )

            downstream_coef_dict_scale[f"{node_name}_scale"] = pyro.sample(
                f"{node_name}_scale", pyro_dist.Exponential(1))

        for node_name in self.root_nodes:
            root_coef_dict_mean[f"{node_name}_int"] = pyro.sample(
                f"{node_name}_int", pyro_dist.Normal(
                    priors[node_name][f"{node_name}_int"], 
                    priors[node_name][f"{node_name}_scale"])
                    )

            root_coef_dict_scale[f"{node_name}_scale"] = pyro.sample(
                f"{node_name}_scale", 
                pyro_dist.Exponential(1)
                )

        # Loop through the data
        downstream_distributions = dict()

        with pyro.plate("observations", self.n_obs):
            
            # Start with root nodes (sampled from normal)
            for node_name in self.root_nodes:

                mean = root_coef_dict_mean[f"{node_name}_int"]
                scale = root_coef_dict_scale[f"{node_name}_scale"]

                # Impute missing values where needed
                with poutine.mask(mask=data[f"missing_{node_name}"].bool()):
                    
                    # missing_values = pyro.sample(f"imp_{node_name}", 
                    #     dist.Normal(
                    #         (mean - (1 - (1 / (1 + torch.exp(-2*(mean+.5)))))*scale), 
                    #         scale)
                    #     ).detach()

                    missing_values = pyro.sample(f"imp_{node_name}", 
                        pyro_dist.Normal(mean, scale)
                        ).detach()

                # # If data passed in, condition on observed data
                if f"obs_{node_name}" in data:

                    # Add in missing data
                    observed = data[f"obs_{node_name}"]#.detach()
                    observed[data[f"missing_{node_name}"].bool()
                                ] = missing_values[
                        data[f"missing_{node_name}"].bool()]
                    
                    root_sample = pyro.sample(
                        f"{node_name}",
                        pyro_dist.Normal(mean, scale),
                        obs=observed
                        )
                    # root_sample = pyro.sample(
                    #     f"{node_name}",
                    #     dist.Normal(
                    #         root_coef_dict_mean[f"{node_name}_int"], 
                    #         root_coef_dict_scale[f"{node_name}_scale"]
                    #         ),
                    #     obs=data[f"obs_{node_name}"]
                    # )
                # If not data passed in, just sample
                else:
                    root_sample = pyro.sample(
                        f"{node_name}",
                        pyro_dist.Normal(
                            root_coef_dict_mean[f"{node_name}_int"], 
                            root_coef_dict_scale[f"{node_name}_scale"]
                            )
                        )

                downstream_distributions[node_name] = root_sample

            # Linear regression for each downstream node
            for node_name, items in self.downstream_nodes.items():

                # calculate mean as sum of upstream nodes and coefficients
                mean = downstream_coef_dict_mean[f"{node_name}_int"]
                for item in items:
                    coef = downstream_coef_dict_mean[f"{node_name}_{item}_coef"]
                    mean = mean + coef*downstream_distributions[item]

                # Define scale
                scale = downstream_coef_dict_scale[f"{node_name}_scale"]

                # # Impute missing values where needed
                with poutine.mask(mask=data[f"missing_{node_name}"].bool()):
                    # missing_values = pyro.sample(f"imp_{node_name}", 
                    #                                 dist.Normal(
                    #                                     mean - \
                    #                                     (1 - (1 / (1 + torch.exp(-2*(mean+.5)))))*scale,
                    #                                     scale
                    #                                     )
                    #                             ).detach()
                    missing_values = pyro.sample(
                        f"imp_{node_name}", 
                        pyro_dist.Normal(mean, scale)
                            ).detach()

                if f"obs_{node_name}" in data:

                    # Add in missing data, detach for pyro gradient
                    observed = data[f"obs_{node_name}"]#.detach_()
                    observed[data[f"missing_{node_name}"].bool()
                                ] = missing_values[
                                    data[f"missing_{node_name}"].bool()]
                    downstream_sample = pyro.sample(
                                f"{node_name}",
                                pyro_dist.Normal(mean, scale),
                                obs=observed)
                    # downstream_sample = pyro.sample(
                    #             f"{node_name}",
                    #             dist.Normal(mean, scale),
                    #             obs=data[f"obs_{node_name}"])
                
                else:
                    downstream_sample = pyro.sample(
                        f"{node_name}",
                        pyro_dist.Normal(mean, scale))

                downstream_distributions[node_name] = downstream_sample

        return downstream_distributions
    
class ProteomicPerturbationCATE(pyro.nn.PyroModule):
    def __init__(self, model: ProteomicPerturbationModel):
        super().__init__()
        self.model = model

    def forward(self, intervention, condition_data, priors):#, obs_data, missing, root_nodes, downstream_nodes, intervention, intervention_node

        # with do(actions={intervention_node: (torch.tensor(intervention).float())}):#, \
            # condition(data=condition_data):#, MultiWorldCounterfactual(), 
        with do(actions=intervention):
            return self.model(data=condition_data, priors=priors)
        # with MultiWorldCounterfactual(), do(actions=intervention):
        #     return self.model(data=condition_data)

def NumpyroProteomicPerturbationModel(data, 
                                      missing,
                                      priors,
                                      root_nodes, 
                                      downstream_nodes): #TODO: add priors

    root_coef_dict_mean = dict()
    root_coef_dict_scale = dict()

    downstream_coef_dict_mean = dict()
    downstream_coef_dict_scale = dict()

    for node_name in root_nodes:

        root_coef_dict_mean[node_name] = numpyro.sample(
            f"{node_name}_int", 
            numpyro_dist.Normal(
                priors[node_name][f"{node_name}_int"], 
                priors[node_name][f"{node_name}_int_scale"])
                )
        root_coef_dict_scale[
            node_name] = numpyro.sample(
                f"{node_name}_scale", 
                numpyro_dist.Exponential(.1)
                )

    for node_name, items in downstream_nodes.items():

        downstream_coef_dict_mean[
            f"{node_name}_intercept"] = numpyro.sample(
                f"{node_name}_intercept",
                numpyro_dist.Normal(
                    priors[node_name][f"{node_name}_int"], 
                    priors[node_name][f"{node_name}_int_scale"])
                )

        for item in items:

            downstream_coef_dict_mean[
                f"{node_name}_{item}_coef"] = numpyro.sample(
                    f"{node_name}_{item}_coef",
                    numpyro_dist.Normal(
                        priors[node_name][f"{node_name}_{item}_coef"], 
                        priors[node_name][f"{node_name}_{item}_coef_scale"])
                        )

        downstream_coef_dict_scale[
            f"{node_name}_scale"] = numpyro.sample(
                f"{node_name}_scale", 
                numpyro_dist.Exponential(1.)
                )


    # Dictionary to store the Pyro root distribution objects
    downstream_distributions = dict()

    # Create Pyro Normal distributions for each node
    for node_name in root_nodes:
        if data is not None:
            # Create a Normal distribution object
            if "latent" in node_name:
                root_sample = numpyro.sample(f"{node_name}",
                            numpyro_dist.Normal(root_coef_dict_mean[node_name],
                                        root_coef_dict_scale[node_name]
                                        ).expand(
                                [data[list(data.keys())[0]].shape[0]])
                )
            else:

                imp = numpyro.sample(
                    f"imp_{node_name}", numpyro_dist.Normal(
                        root_coef_dict_mean[node_name],
                        root_coef_dict_scale[node_name]).expand(
                        [sum(missing[node_name] == 1)]
                    ).mask(False)
                )

                observed = jnp.asarray(
                    data[node_name]).at[missing[node_name] == 1].set(imp)

                root_sample = numpyro.sample(f"{node_name}",
                                             numpyro_dist.Normal(
                                                 root_coef_dict_mean[node_name],
                                                 root_coef_dict_scale[node_name]
                                                 ).expand(
                                        [data[list(data.keys())[0]].shape[0]]),
                                        obs=observed)
        else:
            root_sample = numpyro.sample(f"{node_name}",
                                    numpyro_dist.Normal(
                                        root_coef_dict_mean[node_name],
                                        root_coef_dict_scale[node_name]
                                        ))
            
        # Store the distribution in the dictionary
        downstream_distributions[node_name] = root_sample

    # Create pyro linear regression obj for each downstream node
    for node_name, items in downstream_nodes.items():

        # calculate mean as sum of upstream items
        mean = downstream_coef_dict_mean[f"{node_name}_intercept"]
        for item in items:
            coef = downstream_coef_dict_mean[f"{node_name}_{item}_coef"]
            mean = mean + coef*downstream_distributions[item]

        # Define scale
        scale = downstream_coef_dict_scale[f"{node_name}_scale"]

        if data is not None:

            imp = numpyro.sample(
                f"imp_{node_name}", numpyro_dist.Normal(
                    mean[missing[node_name] == 1],
                    scale).mask(False)
            )

            observed = jnp.asarray(
                data[node_name]).at[missing[node_name] == 1].set(imp)

            # Create a Normal distribution object
            downstream_sample = numpyro.sample(f"{node_name}",
                                            numpyro_dist.Normal(mean, scale),
                                            obs=observed)
        else:
            downstream_sample = numpyro.sample(f"{node_name}",
                                            numpyro_dist.Normal(mean, scale))
            
        # Store the distribution in the dictionary
        downstream_distributions[node_name] = downstream_sample

    return downstream_distributions

