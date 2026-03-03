"""
Main XAI Analysis Script
========================
Orchestrates XAI analysis across all models and samples.
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import models
from models.lightning_models import load_model

# Import XAI components (relative imports)
from .core.cnn_explainer import CNNExplainer
from .core.transformer_explainer import TransformerExplainer
from .core.clap_explainer import CLAPExplainer
from .methods.gradients import GuidedBackprop
from .methods.cam import ScoreCAM
from .methods.spectral import FilterbankAnalyzer, SpectrogramExtractor
from .metrics.sensitivity import DeletionMetric, AverageDropMetric
from .metrics.localization import SparsityMetric, PeakToMeanMetric, CrossModelAgreement
from .visualization.plots import SaliencyPlotter, SpectrogramPlotter, ComparisonPlotter
from .visualization.comparison import ModelComparisonVisualizer


class XAIAnalysisPipeline:
    """Main pipeline for XAI comparative analysis."""
    
    def __init__(self, config_path: str):
        """
        Initialize pipeline from config file.
        
        Args:
            config_path: Path to YAML config file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = self.config['processing']['device']
        self.output_dir = Path(self.config['visualization']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for results
        self.models = {}
        self.explainers = {}
        self.results = {}
        
        print("=" * 70)
        print("XAI COMPARATIVE ANALYSIS PIPELINE")
        print("=" * 70)
    
    def load_models(self):
        """Load all models specified in config."""
        print("\n[1/6] Loading models...")
        
        for model_name, model_config in self.config['models'].items():
            print(f"  Loading {model_config['name']}...")
            
            # Load base model
            if model_name == 'epanns':
                from models.epanns.model import EPANNs
                model = EPANNs(sample_rate=model_config['sample_rate'])
                # Load pretrained weights first
                pretrained_path = 'models/epanns/checkpoint_closeto_.44.pt'
                model.load_pretrained(pretrained_path)
                
            elif model_name == 'ced':
                from models.ced.model import CEDBase
                model = CEDBase()
                # Load pretrained weights first
                pretrained_path = 'models/ced/audiotransformer_base_mAP_4999.pt'
                model.load_pretrained(pretrained_path)
                
            elif model_name == 'clap':
                from models.clap.model import CLAP
                model = CLAP(sample_rate=model_config['sample_rate'])
                # Load pretrained weights first (this initializes self.model)
                pretrained_path = 'models/clap/630k-audioset-fusion-best.pt'
                model.load_pretrained(pretrained_path)
                
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Now load finetuned checkpoint if it's a Lightning checkpoint
            checkpoint_path = model_config['checkpoint']
            if checkpoint_path.endswith('.ckpt'):
                print(f"    Loading finetuned weights from {checkpoint_path}...")
                ckpt = torch.load(checkpoint_path, map_location='cpu')
                state_dict = ckpt['state_dict']
                
                # Remove 'model.' prefix from Lightning checkpoint
                state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                
                # Load finetuned weights
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if len(missing) > 0:
                    print(f"    Warning: Missing keys: {len(missing)}")
                if len(unexpected) > 0:
                    print(f"    Warning: Unexpected keys: {len(unexpected)}")
            
            model.to(self.device)
            model.eval()
            
            self.models[model_name] = model
            
            # Create explainer
            if model_name == 'epanns':
                explainer = CNNExplainer(
                    model, model_name,
                    sample_rate=model_config['sample_rate'],
                    device=self.device
                )
            elif model_name == 'ced':
                explainer = TransformerExplainer(
                    model, model_name,
                    sample_rate=model_config['sample_rate'],
                    device=self.device
                )
            elif model_name == 'clap':
                explainer = CLAPExplainer(
                    model, model_name,
                    sample_rate=model_config['sample_rate'],
                    device=self.device
                )
            
            self.explainers[model_name] = explainer
            print(f"    ✓ {model_config['name']} loaded successfully")
        
        print(f"  Total models loaded: {len(self.models)}")
    
    def analyze_sample(self, sample_path: str, sample_info: Dict):
        """
        Run complete XAI analysis on a single sample.
        
        Args:
            sample_path: Path to audio file
            sample_info: Sample metadata
            
        Returns:
            Dictionary with all results
        """
        sample_name = Path(sample_path).stem
        sample_type = sample_info['type']
        
        print(f"\n  Analyzing {sample_name} ({sample_type})...")
        
        results = {
            'sample_name': sample_name,
            'sample_type': sample_type,
            'models': {}
        }
        
        for model_name, explainer in self.explainers.items():
            print(f"    [{model_name}] Processing...")
            
            model_results = {
                'spectrograms': {},
                'saliency_maps': {},
                'metrics': {}
            }
            
            # Load audio
            waveform = explainer.load_audio(sample_path)
            
            # Get prediction
            prediction = explainer.get_prediction(waveform)
            model_results['prediction'] = prediction
            
            # Extract spectrograms
            spec_extractor = SpectrogramExtractor(explainer.model, model_name)
            specs = spec_extractor.extract(waveform, device=self.device)
            model_results['spectrograms'] = specs
            
            # Get primary spectrogram for visualization
            if 'logmel' in specs:
                primary_spec = specs['logmel']
            elif 'mel_spectrogram' in specs:
                primary_spec = specs['mel_spectrogram']
            else:
                primary_spec = list(specs.values())[0]
            
            spec_shape = primary_spec.shape  # (T, F)
            print(f"      [{model_name}] Spectrogram shape: {spec_shape} (T={spec_shape[0]}, F={spec_shape[1]})")
            
            # --- Guided Backprop ---
            if self.config['methods']['guided_backprop']['enabled']:
                # Check if multi-layer analysis is enabled
                multi_layer = self.config['methods']['guided_backprop'].get('multi_layer_analysis', False)
                target_layers_list = self.config['methods']['guided_backprop']['target_layers'].get(model_name, [])
                
                if multi_layer and len(target_layers_list) > 1:
                    # Multi-layer analysis - compute saliency for each layer
                    multi_layer_saliency = {}
                    
                    for target_layer in target_layers_list:
                        try:
                            
                            # For CED (Transformer), use attention maps instead of gradients
                            # because gradient-based methods don't work well with transformers
                            if model_name == 'ced':
                                # Extract attention maps and convert to saliency-like visualization
                                if 'blocks.' in target_layer:
                                    # For transformer blocks, use attention maps
                                    layer_idx = int(target_layer.split('.')[-1])
                                    
                                    # Get attention maps (already computed)
                                    attn_maps = explainer.get_attention_weights(waveform, average_heads=True)
                                    
                                    if layer_idx < len(attn_maps):
                                        # attn_maps[layer_idx] is (seq_len, seq_len)
                                        # Take mean attention from all tokens as importance
                                        attn = attn_maps[layer_idx]
                                        if isinstance(attn, np.ndarray):
                                            attn_importance = attn.mean(axis=0)  # (seq_len,)
                                        else:
                                            attn_importance = attn.mean(dim=0).numpy()
                                        
                                        # Remove CLS token (first token)
                                        attn_importance = attn_importance[1:]  # Now (num_patches,)
                                        
                                        # Map to spectrogram shape (T, F)
                                        # Replicate across frequency dimension
                                        from scipy.ndimage import zoom
                                        # Resize temporally to match spec_shape[0]
                                        temporal_sal = zoom(attn_importance, spec_shape[0] / len(attn_importance), order=1)
                                        # Replicate across frequency
                                        gb_sal = np.tile(temporal_sal.reshape(-1, 1), (1, spec_shape[1]))
                                        
                                        multi_layer_saliency[target_layer] = gb_sal
                                    else:
                                        print(f"        Warning: Layer index {layer_idx} out of range")
                                else:
                                    # For front_end, use gradient-based (should work on mel spectrogram)
                                    from models.xAI.methods.gradients import VanillaBackprop
                                    gb = VanillaBackprop(explainer.model, target_layer=target_layer)
                                    _, gb_sal = gb.generate(waveform, explainer.target_class, self.device)
                                    
                                    # Check if all zeros - if so, create uniform importance
                                    if gb_sal.max() < 1e-6:
                                        print(f"        Warning: Gradients are zero, using uniform importance")
                                        gb_sal = np.ones(spec_shape) * 0.5
                                    
                                    
                                    if gb_sal.shape == spec_shape:
                                        multi_layer_saliency[target_layer] = gb_sal
                                    else:
                                        from scipy.ndimage import zoom
                                        zoom_factors = (spec_shape[0] / gb_sal.shape[0], 
                                                       spec_shape[1] / gb_sal.shape[1])
                                        gb_sal_resized = zoom(gb_sal, zoom_factors, order=1)
                                        multi_layer_saliency[target_layer] = gb_sal_resized
                            else:
                                # Use guided backprop for CNNs
                                gb = GuidedBackprop(explainer.model, target_layer=target_layer)
                                _, gb_sal = gb.generate(waveform, explainer.target_class, self.device)
                                
                                
                                # Verify shape matches spectrogram
                                if gb_sal.shape == spec_shape:
                                    multi_layer_saliency[target_layer] = gb_sal
                                else:
                                    # Try to resize
                                    from scipy.ndimage import zoom
                                    zoom_factors = (spec_shape[0] / gb_sal.shape[0], 
                                                   spec_shape[1] / gb_sal.shape[1])
                                    gb_sal_resized = zoom(gb_sal, zoom_factors, order=1)
                                    multi_layer_saliency[target_layer] = gb_sal_resized
                        except Exception as e:
                            print(f"        Warning: Layer {target_layer} failed: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    model_results['saliency_maps']['guided_backprop_multi_layer'] = multi_layer_saliency
                    # Use last layer as primary
                    if len(multi_layer_saliency) > 0:
                        model_results['saliency_maps']['guided_backprop'] = list(multi_layer_saliency.values())[-1]
                    print(f"      ✓ Generated {len(multi_layer_saliency)} layer saliency maps")
                    
                else:
                    # Single layer analysis (default) - use front_end/logmel layer
                    if model_name == 'epanns':
                        target_layer = 'model.logmel_extractor'
                    elif model_name == 'ced':
                        target_layer = 'front_end'  # MelSpectrogram layer
                    elif model_name == 'clap':
                        target_layer = 'model.audio_branch.logmel_extractor'
                    else:
                        target_layer = None
                    
                    # For CED, use attention-based importance for primary saliency
                    if model_name == 'ced':
                        try:
                            attn_maps = explainer.get_attention_weights(waveform, average_heads=True)
                            if len(attn_maps) > 0:
                                # Use last layer attention
                                attn = attn_maps[-1]
                                if isinstance(attn, np.ndarray):
                                    attn_importance = attn.mean(axis=0)
                                else:
                                    attn_importance = attn.mean(dim=0).numpy()
                                
                                attn_importance = attn_importance[1:]  # Remove CLS
                                
                                from scipy.ndimage import zoom
                                temporal_sal = zoom(attn_importance, spec_shape[0] / len(attn_importance), order=1)
                                gb_saliency = np.tile(temporal_sal.reshape(-1, 1), (1, spec_shape[1]))
                                
                            else:
                                print(f"      Warning: No attention maps, using uniform saliency")
                                gb_saliency = np.ones(spec_shape) * 0.5
                        except Exception as e:
                            print(f"      Warning: Attention extraction failed: {e}, using gradient")
                            from models.xAI.methods.gradients import VanillaBackprop
                            gb = VanillaBackprop(explainer.model, target_layer=target_layer)
                            _, gb_saliency = gb.generate(waveform, explainer.target_class, self.device)
                    else:
                        # Use guided backprop for CNNs
                        gb = GuidedBackprop(explainer.model, target_layer=target_layer)
                        _, gb_saliency = gb.generate(waveform, explainer.target_class, self.device)
                    
                    # Verify we got the right shape
                    expected_shape = spec_shape
                    if gb_saliency.shape != expected_shape:
                        # Resize if needed
                        if gb_saliency.ndim == 2:
                            # Interpolate to match
                            from scipy.ndimage import zoom
                            zoom_factors = (expected_shape[0] / gb_saliency.shape[0], 
                                           expected_shape[1] / gb_saliency.shape[1])
                            gb_saliency = zoom(gb_saliency, zoom_factors, order=1)
                    
                    model_results['saliency_maps']['guided_backprop'] = gb_saliency
            
            # --- Score-CAM ---
            if self.config['methods']['score_cam']['enabled']:
                target_layers = self.config['methods']['score_cam']['target_layers'].get(model_name, [])
                if target_layers:
                    score_cam = None
                    try:
                        score_cam = ScoreCAM(explainer.model, target_layers[0])
                        scam_result = score_cam.generate(
                            waveform, explainer.target_class,
                            spec_shape=spec_shape, device=self.device
                        )
                        # ScoreCAM returns (cam, normalized_cam)
                        if isinstance(scam_result, tuple) and len(scam_result) == 2:
                            scam_saliency = scam_result[1]  # Use normalized version
                        else:
                            scam_saliency = scam_result
                        model_results['saliency_maps']['score_cam'] = scam_saliency
                    except Exception as e:
                        print(f"      Warning: Score-CAM failed: {e}")
                    finally:
                        # Clean up hooks immediately to avoid interference
                        if score_cam is not None:
                            del score_cam
            
            # --- Attention Maps (CED only) ---
            if model_name == 'ced' and self.config['methods'].get('attention_maps', {}).get('enabled', False):
                try:
                    attention_maps = explainer.get_attention_weights(waveform, average_heads=True)
                    model_results['attention_maps'] = attention_maps
                except Exception as e:
                    print(f"      Warning: Attention extraction failed: {e}")
            
            # --- Attention Rollout (CED only) ---
            # Use the already extracted attention maps to compute rollout
            if model_name == 'ced' and self.config['methods'].get('attention_rollout', {}).get('enabled', False):
                if 'attention_maps' in model_results and len(model_results['attention_maps']) > 0:
                    try:
                        # Compute rollout from extracted attention maps
                        attention_matrices = model_results['attention_maps']
                        
                        # Start with identity
                        result = torch.eye(attention_matrices[0].shape[0])
                        
                        for attn in attention_matrices:
                            if isinstance(attn, np.ndarray):
                                attn = torch.from_numpy(attn).float()
                            
                            # Add residual connection
                            attn = attn + torch.eye(attn.shape[0])
                            
                            # Renormalize
                            attn = attn / attn.sum(dim=-1, keepdim=True)
                            
                            # Multiply with previous result
                            result = torch.matmul(attn, result)
                        
                        # Get attention from [CLS] token or average
                        rollout = result[0, 1:].numpy()  # Exclude [CLS] token itself
                        model_results['attention_rollout'] = rollout
                        print(f"      Computed attention rollout from {len(attention_matrices)} layers")
                    except Exception as e:
                        print(f"      Warning: Attention rollout computation failed: {e}")
                else:
                    print(f"      Warning: No attention maps available for rollout")
            
            # Compute metrics
            if model_results['saliency_maps']:
                # Get primary saliency (skip dict values like multi_layer)
                primary_saliency = None
                for key, val in model_results['saliency_maps'].items():
                    if isinstance(val, np.ndarray):
                        primary_saliency = val
                        break
                
                if primary_saliency is None:
                    print(f"      Warning: No numpy array saliency found for metrics")
                else:
                    # Sparsity
                    if self.config['metrics']['sparsity']['enabled']:
                        model_results['metrics']['sparsity'] = SparsityMetric.compute(primary_saliency)
                    
                    # Peak-to-Mean
                    if self.config['metrics']['peak_to_mean']['enabled']:
                        model_results['metrics']['peak_to_mean'] = PeakToMeanMetric.compute(primary_saliency)
                    
                    # Deletion
                    if self.config['metrics']['deletion']['enabled']:
                        deletion_metric = DeletionMetric(
                            explainer.model, explainer.target_class,
                            num_steps=self.config['metrics']['deletion']['steps']
                        )
                        
                        def forward_func(w):
                            return explainer.model(w)
                        
                        _, del_auc = deletion_metric.compute(waveform, primary_saliency, forward_func, self.device)
                        model_results['metrics']['deletion_auc'] = del_auc
                    
                    # Average Drop
                    if self.config['metrics']['average_drop']['enabled']:
                        ad_metric = AverageDropMetric(
                            explainer.model, explainer.target_class,
                            top_k_percent=self.config['metrics']['average_drop']['top_k_percent']
                        )
                        
                        ad = ad_metric.compute(waveform, primary_saliency, forward_func, self.device)
                        model_results['metrics']['average_drop'] = ad
            
            results['models'][model_name] = model_results
            print(f"      ✓ Completed")
        
        # Cross-model metrics
        if self.config['metrics']['cross_model_agreement']['enabled']:
            saliency_maps = {
                m: r['saliency_maps']['guided_backprop']
                for m, r in results['models'].items()
                if 'guided_backprop' in r['saliency_maps']
            }
            
            if len(saliency_maps) > 1:
                consensus = CrossModelAgreement.compute_consensus_score(
                    saliency_maps,
                    threshold=self.config['metrics'].get('cross_model_agreement', {}).get('threshold', 0.7)
                )
                results['cross_model_consensus'] = consensus
        
        return results
    
    def run_analysis(self):
        """Run complete XAI analysis pipeline."""
        print("\n[2/6] Running XAI analysis on samples...")
        
        samples = self.config['samples']['auto_selected']
        
        for sample_info in samples:
            sample_path = sample_info['path']
            sample_results = self.analyze_sample(sample_path, sample_info)
            
            sample_name = sample_results['sample_name']
            self.results[sample_name] = sample_results
        
        print(f"  ✓ Analyzed {len(self.results)} samples")
    
    def filterbank_analysis(self):
        """Perform filterbank analysis for applicable models."""
        print("\n[3/6] Performing filterbank analysis...")
        
        if not self.config['methods']['filterbank_analysis']['enabled']:
            print("  Skipped (disabled in config)")
            return
        
        models_to_analyze = self.config['methods']['filterbank_analysis']['models']
        
        # Collect filterbank data from all models
        all_filterbank_data = {}
        
        for model_name in models_to_analyze:
            if model_name not in self.models:
                continue
            
            print(f"  Analyzing {model_name} filterbank...")
            
            model_config = self.config['models'][model_name]
            
            try:
                
                # Use model-specific parameters matching the actual model architecture
                if model_name == 'epanns':
                    # EPANNs constants from models/epanns/model.py
                    n_fft = 1024
                    n_mels = 64
                    fmin = 50
                    fmax = 14000
                elif model_name == 'ced':
                    # CED constants from models/ced/model.py
                    n_fft = 512
                    n_mels = 64
                    fmin = 0
                    fmax = 8000
                elif model_name == 'clap':
                    # CLAP constants from models/clap/model.py
                    n_fft = 1024
                    n_mels = 64
                    fmin = 50
                    fmax = 14000
                else:
                    # Default fallback
                    n_fft = 1024
                    n_mels = 64
                    fmin = 50
                    fmax = model_config['sample_rate'] // 2
                
                
                fb_analyzer = FilterbankAnalyzer(
                    self.models[model_name],
                    sr=model_config['sample_rate'],
                    n_fft=n_fft,
                    n_mels=n_mels,
                    fmin=fmin,
                    fmax=fmax
                )
                
                comparison = fb_analyzer.compare_filterbanks()
                all_filterbank_data[model_name] = comparison
                
                print(f"    ✓ Extracted filterbank data")
                
            except Exception as e:
                print(f"    ✗ Failed: {e}")
        
        # Create unified comparison plot
        if len(all_filterbank_data) > 0:
            comp_visualizer = ModelComparisonVisualizer(
                dpi=self.config['visualization']['dpi'],
                output_format=self.config['visualization']['output_format']
            )
            
            fig_path = self.output_dir / f'filterbanks_comparison.{self.config["visualization"]["output_format"]}'
            comp_visualizer.plot_unified_filterbank_comparison(
                all_filterbank_data,
                title='Mel Filterbank Comparison: EPANNs vs CED vs CLAP',
                save_path=str(fig_path)
            )
            
            print(f"  ✓ Filterbank comparison plot saved: {fig_path.name}")
        
        print("  ✓ Filterbank analysis complete")
    
    def generate_visualizations(self):
        """Generate all visualizations."""
        print("\n[4/6] Generating visualizations...")
        
        saliency_plotter = SaliencyPlotter(
            dpi=self.config['visualization']['dpi'],
            output_format=self.config['visualization']['output_format']
        )
        
        comp_visualizer = ModelComparisonVisualizer(
            dpi=self.config['visualization']['dpi'],
            output_format=self.config['visualization']['output_format']
        )
        
        for sample_name, sample_results in self.results.items():
            sample_type = sample_results['sample_type']
            
            print(f"  Visualizing {sample_name}...")
            
            # Collect data for visualization
            spectrograms = {}
            saliency_maps = {}
            
            for model_name, model_results in sample_results['models'].items():
                # Get primary spectrogram
                specs = model_results['spectrograms']
                if 'logmel' in specs:
                    spectrograms[model_name] = specs['logmel']
                elif 'mel_spectrogram' in specs:
                    spectrograms[model_name] = specs['mel_spectrogram']
                
                # Get primary saliency
                if 'guided_backprop' in model_results['saliency_maps']:
                    saliency_maps[model_name] = model_results['saliency_maps']['guided_backprop']
            
            # Get colormap from config
            saliency_cmap = self.config['visualization']['colormap']['saliency']
            
            # Side-by-side comparison - DISABLED (not needed)
            # if self.config['visualization']['plots']['side_by_side_comparison']:
            #     fig_path = self.output_dir / f'{sample_name}_side_by_side.{self.config["visualization"]["output_format"]}'
            #     comp_visualizer.plot_side_by_side(
            #         spectrograms,
            #         saliency_maps,
            #         sample_name=f'{sample_name} ({sample_type})',
            #         save_path=str(fig_path),
            #         saliency_cmap=saliency_cmap
            #     )
            
            # Time series comparison - DISABLED (not clear)
            # if self.config['visualization']['plots']['time_series']:
            #     fig_path = self.output_dir / f'{sample_name}_time_series.{self.config["visualization"]["output_format"]}'
            #     comp_visualizer.plot_time_series_comparison(
            #         saliency_maps,
            #         sample_name=f'{sample_name} ({sample_type})',
            #         save_path=str(fig_path)
            #     )
            
            # Multi-layer saliency visualization (per model)
            for model_name, model_results in sample_results['models'].items():
                # For TN samples: skip multi-layer plots (they will only get attention maps below)
                if sample_type == 'TN':
                    continue
                
                if 'guided_backprop_multi_layer' in model_results['saliency_maps']:
                    multi_layer_sal = model_results['saliency_maps']['guided_backprop_multi_layer']
                    
                    if len(multi_layer_sal) > 1:
                        # Get spectrogram for this model
                        spec = spectrograms.get(model_name)
                        fig_path = self.output_dir / f'{sample_name}_{model_name}_multi_layer.{self.config["visualization"]["output_format"]}'
                        saliency_plotter.plot_multi_layer_saliency(
                            multi_layer_sal,
                            spectrogram=spec,
                            title=f'{model_name.upper()} Multi-Layer Saliency: {sample_name} ({sample_type})',
                            save_path=str(fig_path),
                            cmap=saliency_cmap
                        )
                        print(f"    ✓ Multi-layer plot: {fig_path.name}")
            
            # Attention maps visualization (CED)
            # Generate for both TP and TN samples
            for model_name, model_results in sample_results['models'].items():
                if 'attention_maps' in model_results:
                    attention_maps = model_results['attention_maps']
                    if len(attention_maps) > 0:
                        # Convert to numpy if needed
                        attention_np = [a.numpy() if hasattr(a, 'numpy') else a for a in attention_maps]
                        fig_path = self.output_dir / f'{sample_name}_{model_name}_attention_maps.{self.config["visualization"]["output_format"]}'
                        saliency_plotter.plot_attention_maps(
                            attention_np,
                            title=f'{model_name.upper()} Attention Maps: {sample_name} ({sample_type})',
                            save_path=str(fig_path)
                        )
                        print(f"    ✓ Attention maps: {fig_path.name}")
        
        print("  ✓ Visualizations complete")
    

    def run(self):
        """Execute complete pipeline."""
        try:
            self.load_models()
            self.run_analysis()
            self.filterbank_analysis()
            self.generate_visualizations()
            
            print("\n[5/5] Pipeline complete!")
            print("=" * 70)
            print(f"Results saved to: {self.output_dir}")
            print("=" * 70)
            
        except Exception as e:
            print(f"\n✗ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='XAI Comparative Analysis Pipeline')
    parser.add_argument('--config', type=str, default='models/xAI/config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    pipeline = XAIAnalysisPipeline(args.config)
    pipeline.run()


if __name__ == '__main__':
    main()
