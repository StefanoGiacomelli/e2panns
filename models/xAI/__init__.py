"""XAI Framework for Emergency Vehicle Recognition Models."""

__version__ = "1.0.0"
__author__ = "Stefano Giacomelli"

from .core.base_explainer import BaseExplainer
from .core.cnn_explainer import CNNExplainer
from .core.transformer_explainer import TransformerExplainer
from .core.clap_explainer import CLAPExplainer

__all__ = [
    "BaseExplainer",
    "CNNExplainer",
    "TransformerExplainer",
    "CLAPExplainer",
]
