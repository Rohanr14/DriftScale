"""Trace loading and synthetic workload generation."""

from driftscale.traces.azure_loader import (
    AZURE_CHECKPOINT_IDS,
    AzureCheckpointRegimes,
    AzureTraceColumns,
    ChronologicalSplit,
    VmSelectionStrategy,
    azure_checkpoint_paths,
    chronological_split,
    load_azure_checkpoint_regimes,
    load_azure_cpu_matrix,
    load_dense_azure_cpu_matrix,
    load_persistent_vm_matrix,
    scan_checkpoint_dense_vms,
    scan_persistent_dense_vms,
)
from driftscale.traces.preprocess import (
    DemandMappingConfig,
    DemandMappingVariant,
    build_preprocessed_trace,
    compute_demand_signal,
)
from driftscale.traces.synthetic import SyntheticEpisode, generate_synthetic_episode

__all__ = [
    "AzureTraceColumns",
    "AzureCheckpointRegimes",
    "AZURE_CHECKPOINT_IDS",
    "ChronologicalSplit",
    "DemandMappingConfig",
    "DemandMappingVariant",
    "SyntheticEpisode",
    "VmSelectionStrategy",
    "build_preprocessed_trace",
    "compute_demand_signal",
    "generate_synthetic_episode",
    "azure_checkpoint_paths",
    "chronological_split",
    "load_azure_checkpoint_regimes",
    "load_azure_cpu_matrix",
    "load_dense_azure_cpu_matrix",
    "load_persistent_vm_matrix",
    "scan_checkpoint_dense_vms",
    "scan_persistent_dense_vms",
]
