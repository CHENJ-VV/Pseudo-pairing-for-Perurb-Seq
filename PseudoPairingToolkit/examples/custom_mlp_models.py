"""Example model factories for direct Python API use."""
from pseudopair.evaluation.models import ForwardPerturbationMLP, InversePerturbationMLP


def forward_model(n_genes, n_perturbations, config):
    return ForwardPerturbationMLP(
        n_genes=n_genes,
        n_perturbations=n_perturbations,
        pert_emb_dim=512,
        hidden_dim=2048,
        latent_dim=1024,
        dropout=0.10,
    )


def inverse_model(n_genes, n_perturbations, config):
    return InversePerturbationMLP(
        n_genes=n_genes,
        n_perturbations=n_perturbations,
        hidden_dim=2048,
        latent_dim=1024,
        dropout=0.10,
    )
