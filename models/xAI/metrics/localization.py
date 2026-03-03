"""
Localization and Consistency Metrics
====================================
Sparsity, Peak-to-Mean, Cross-Model Agreement, Temporal Overlap.
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from typing import List, Dict, Tuple


class SparsityMetric:
    """
    Measure sparsity of saliency maps using Gini coefficient.
    Higher value = more concentrated/sparse explanation.
    """
    
    @staticmethod
    def compute(saliency_map: np.ndarray) -> float:
        """
        Compute Gini coefficient as sparsity measure.
        
        Args:
            saliency_map: Saliency values
            
        Returns:
            gini: Gini coefficient (0-1)
        """
        # Flatten and sort
        values = saliency_map.flatten()
        values = np.sort(values)
        n = len(values)
        
        if n == 0 or values.sum() == 0:
            return 0.0
        
        # Gini coefficient
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n
        
        return float(gini)
    
    @staticmethod
    def compute_hoyer(saliency_map: np.ndarray) -> float:
        """
        Hoyer sparsity measure (alternative to Gini).
        
        Args:
            saliency_map: Saliency values
            
        Returns:
            hoyer: Hoyer sparsity (0-1)
        """
        values = saliency_map.flatten()
        n = len(values)
        
        if n == 0:
            return 0.0
        
        l1_norm = np.sum(np.abs(values))
        l2_norm = np.sqrt(np.sum(values ** 2))
        
        if l2_norm == 0:
            return 0.0
        
        hoyer = (np.sqrt(n) - (l1_norm / l2_norm)) / (np.sqrt(n) - 1)
        
        return float(hoyer)


class PeakToMeanMetric:
    """
    Ratio of maximum saliency to mean saliency.
    Higher value = more concentrated attention.
    """
    
    @staticmethod
    def compute(saliency_map: np.ndarray) -> float:
        """
        Compute peak-to-mean ratio.
        
        Args:
            saliency_map: Saliency values
            
        Returns:
            ratio: Peak-to-mean ratio
        """
        values = saliency_map.flatten()
        
        if len(values) == 0:
            return 0.0
        
        peak = np.max(values)
        mean = np.mean(values)
        
        if mean > 0:
            ratio = peak / mean
        else:
            ratio = 0.0
        
        return float(ratio)
    
    @staticmethod
    def compute_top_k_concentration(saliency_map: np.ndarray, k_percent: float = 0.1) -> float:
        """
        Measure concentration: sum of top-k% values / total sum.
        
        Args:
            saliency_map: Saliency values
            k_percent: Percentage of top values to consider
            
        Returns:
            concentration: Concentration ratio (0-1)
        """
        values = saliency_map.flatten()
        
        if len(values) == 0 or values.sum() == 0:
            return 0.0
        
        k = max(1, int(len(values) * k_percent))
        top_k_values = np.sort(values)[-k:]
        
        concentration = top_k_values.sum() / values.sum()
        
        return float(concentration)


class CrossModelAgreement:
    """
    Measure agreement between saliency maps from different models.
    """
    
    @staticmethod
    def compute_correlation(
        saliency_1: np.ndarray,
        saliency_2: np.ndarray,
        method: str = "pearson"
    ) -> float:
        """
        Compute correlation between two saliency maps.
        
        Args:
            saliency_1: First saliency map
            saliency_2: Second saliency map
            method: 'pearson' or 'spearman'
            
        Returns:
            correlation: Correlation coefficient
        """
        # Flatten and align shapes
        s1 = saliency_1.flatten()
        s2 = saliency_2.flatten()
        
        # Resize if needed
        if len(s1) != len(s2):
            # Interpolate shorter to match longer
            if len(s1) < len(s2):
                s1 = np.interp(
                    np.linspace(0, len(s1) - 1, len(s2)),
                    np.arange(len(s1)),
                    s1
                )
            else:
                s2 = np.interp(
                    np.linspace(0, len(s2) - 1, len(s1)),
                    np.arange(len(s2)),
                    s2
                )
        
        if method == "pearson":
            try:
                corr, _ = pearsonr(s1, s2)
            except:
                # Handle constant arrays
                corr = 0.0
        elif method == "spearman":
            try:
                corr, _ = spearmanr(s1, s2)
            except:
                # Handle constant arrays
                corr = 0.0
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        return float(corr)
    
    @staticmethod
    def compute_pairwise_agreement(
        saliency_maps: Dict[str, np.ndarray],
        method: str = "pearson"
    ) -> Dict[Tuple[str, str], float]:
        """
        Compute pairwise agreement between multiple saliency maps.
        
        Args:
            saliency_maps: Dictionary of {model_name: saliency_map}
            method: Correlation method
            
        Returns:
            Dictionary of {(model1, model2): correlation}
        """
        results = {}
        models = list(saliency_maps.keys())
        
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                corr = CrossModelAgreement.compute_correlation(
                    saliency_maps[model1],
                    saliency_maps[model2],
                    method=method
                )
                results[(model1, model2)] = corr
        
        return results
    
    @staticmethod
    def compute_consensus_score(
        saliency_maps: Dict[str, np.ndarray],
        threshold: float = 0.7
    ) -> float:
        """
        Compute overall consensus score (average pairwise correlation).
        
        Args:
            saliency_maps: Dictionary of saliency maps
            threshold: Minimum correlation for agreement
            
        Returns:
            consensus_score: Average correlation across all pairs
        """
        pairwise = CrossModelAgreement.compute_pairwise_agreement(saliency_maps)
        
        if len(pairwise) == 0:
            return 0.0
        
        avg_correlation = np.mean(list(pairwise.values()))
        
        return float(avg_correlation)
