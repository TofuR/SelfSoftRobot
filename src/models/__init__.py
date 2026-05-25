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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
