from .layers import PositionalEncoder

__all__ = [
    "PositionalEncoder",
    "MultiScaleEMA",
    "MSTNFModel",
    "CMSTNFModel",
    "MSSCNFModel",
]


def __getattr__(name):
    """Lazy imports to avoid circular dependencies with src.fields/."""
    if name == "MultiScaleEMA":
        from src.encoders.multi_scale_ema import MultiScaleEMA
        return MultiScaleEMA
    if name == "MSTNFModel":
        from .model_mstnf import MSTNFModel
        return MSTNFModel
    if name == "CMSTNFModel":
        from .model_cmstnf import CMSTNFModel
        return CMSTNFModel
    if name == "MSSCNFModel":
        from .model_ms_scnf import MSSCNFModel
        return MSSCNFModel
    if name == "FlowMatchPointCloudModel":
        from .model_flowmatch import FlowMatchPointCloudModel
        return FlowMatchPointCloudModel
    if name == "SpatialSequenceModel":
        from .model_spatial_sequence import SpatialSequenceModel
        return SpatialSequenceModel
    if name == "PCSpatialSequenceModel":
        from .model_pc_spatial import PCSpatialSequenceModel
        return PCSpatialSequenceModel
    if name == "StateTransitionSpatialModel":
        from .model_state_transition import StateTransitionSpatialModel
        return StateTransitionSpatialModel
    if name == "GTObservedTransitionModel":
        from .model_gt_transition import GTObservedTransitionModel
        return GTObservedTransitionModel
    if name == "OpenLoopTransitionModel":
        from .model_open_loop_transition import OpenLoopTransitionModel
        return OpenLoopTransitionModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
