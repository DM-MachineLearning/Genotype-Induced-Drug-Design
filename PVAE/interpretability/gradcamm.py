"""
Interpretability utilities for 1D convolutional models (MSECNNVAE)

Provides:
- GradCAM1D: compute class-discriminative activation maps for 1D conv layers
- InputGradientAttribution: compute per-gene importance using input gradients
- IntegratedGradients: compute per-gene importance using integrated gradients
- helpers to visualize/save CAM overlays on sequence inputs

Usage:
  from Genotype_Induced_Drug_Design.PVAE.interpretability.gradcamm import InputGradientAttribution
  attr = InputGradientAttribution(model)
  importance = attr.attribute(input_dna_tensor, input_gene_tensor, target_class=1)
  # importance shape: (batch, num_genes) - direct per-gene scores

"""

import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class GradCAM1D:
	"""Simple Grad-CAM for 1D convolutional models.

	Designed to work with models similar to `MSECNNVAE` that accept two
	input channels stacked as (batch, 2, length) in their conv encoder.

	Parameters
	- model: torch.nn.Module. Model should expose a classifier output via
	  `forward_with_classifier` that returns (recon_dna, recon_gene, mu, logvar, class_logits).
	- target_module: optional module or module name to target. If None,
	  GradCAM will use the last Conv1d found inside `model.encoder_conv`.
	"""

	def __init__(self, model: torch.nn.Module, target_module: t.Union[str, torch.nn.Module, None] = None):
		self.model = model
		self.model.eval()
		self.device = next(model.parameters()).device

		# storage for forward activations and backward gradients
		self.activations = None
		self.gradients = None

		# resolve target module
		if target_module is None:
			target_module = self._find_last_conv1d()
		elif isinstance(target_module, str):
			target_module = self._get_module_by_name(target_module)

		if target_module is None:
			raise ValueError("Could not resolve target_module for GradCAM")

		self.target_module = target_module
		# register hooks
		self._register_hooks()

	def _find_last_conv1d(self):
		# Try to find last nn.Conv1d under model.encoder_conv
		enc = getattr(self.model, "encoder_conv", None)
		if enc is None:
			# fallback: search entire model
			modules = list(self.model.modules())
		else:
			modules = list(enc.modules())

		last = None
		for m in modules:
			import torch.nn as nn

			if isinstance(m, nn.Conv1d):
				last = m
		return last

	def _get_module_by_name(self, name: str):
		for n, m in self.model.named_modules():
			if n == name:
				return m
		# not found
		return None

	def _forward_hook(self, module, input, output):
		# store activations (detach)
		self.activations = output.detach()

	def _backward_hook(self, module, grad_input, grad_output):
		# grad_output is a tuple; take first element
		self.gradients = grad_output[0].detach()

	def _register_hooks(self):
		# remove existing hooks if any
		try:
			if hasattr(self, "_fwd_handle"):
				self._fwd_handle.remove()
			if hasattr(self, "_bwd_handle"):
				self._bwd_handle.remove()
		except Exception:
			pass

		self._fwd_handle = self.target_module.register_forward_hook(self._forward_hook)
		self._bwd_handle = self.target_module.register_backward_hook(self._backward_hook)

	def _clear(self):
		self.activations = None
		self.gradients = None

	def generate_cam(
		self,
		x_dna: torch.Tensor,
		x_gene: torch.Tensor,
		target_class: t.Optional[int] = None,
		use_cuda: bool = True,
	) -> np.ndarray:
		"""Generate CAM maps for the batch of inputs.

		Inputs
		- x_dna, x_gene: tensors of shape (batch, length). They will be stacked into (batch,2,length).
		- target_class: if None, uses predicted class for each sample.

		Returns
		- cams: numpy array shape (batch, input_length) with values in [0,1]
		"""
		self._clear()

		device = self.device if torch.cuda.is_available() and use_cuda else torch.device("cpu")
		self.model.to(device)
		x_dna = x_dna.to(device).float()
		x_gene = x_gene.to(device).float()

		# forward through model and get logits
		# prefer forward_with_classifier if present
		if hasattr(self.model, "forward_with_classifier"):
			recon_dna, recon_gene, mu, logvar, class_logits = self.model.forward_with_classifier(x_dna, x_gene)
		else:
			# fallback: call forward and expect logits as last element
			out = self.model(x_dna, x_gene)
			# try to find logits
			if isinstance(out, tuple) and len(out) >= 5:
				class_logits = out[4]
			elif isinstance(out, tuple) and len(out) >= 1:
				class_logits = out[-1]
			else:
				raise RuntimeError("Model does not return class logits; provide a model with `forward_with_classifier`")

		# determine target scores
		# if target_class is None, use predicted class per sample
		probs = F.softmax(class_logits, dim=1)
		preds = torch.argmax(probs, dim=1)

		cams = []
		batch_size = x_dna.shape[0]

		for i in range(batch_size):
			self.model.zero_grad()
			# choose index
			idx = target_class if target_class is not None else preds[i].item()

			# score for target class for sample i
			score = class_logits[i, idx]
			score.backward(retain_graph=True)

			if self.activations is None or self.gradients is None:
				raise RuntimeError("Hooks did not capture activations/gradients. Check target module selection.")

			# activations: (batch, channels, spatial)
			act = self.activations[i:i+1]  # (1,C,L')
			grad = self.gradients[i:i+1]   # (1,C,L')

			# global average pooling of gradients over spatial dim -> weights
			weights = torch.mean(grad, dim=2, keepdim=True)  # (1,C,1)

			# weighted combination
			cam = torch.sum(weights * act, dim=1, keepdim=True)  # (1,1,L')
			cam = F.relu(cam)

			# normalize
			cam = cam - cam.min()
			if cam.max() > 0:
				cam = cam / (cam.max() + 1e-8)

			# upsample to model input length
			input_length = x_dna.shape[1]
			cam_upsampled = F.interpolate(cam, size=input_length, mode="linear", align_corners=False)
			cam_np = cam_upsampled.squeeze().cpu().numpy()
			cams.append(cam_np)

			# clear gradients for next sample
			self.model.zero_grad()
			self._clear()

		cams = np.stack(cams, axis=0)
		return cams


def visualize_cam(
	cam: np.ndarray,
	input_seq: t.Optional[np.ndarray] = None,
	out_path: t.Optional[str] = None,
	cmap: str = "viridis",
	figsize: t.Tuple[int, int] = (12, 3),
	title: t.Optional[str] = None,
):
	"""Plot a CAM (1D heatmap). If `input_seq` is provided it is plotted below the heatmap.

	- cam: 1D numpy array (length,) or (batch, length). If batch, only first element plotted.
	- input_seq: optional 1D array of same length to plot as a line below the heatmap.
	- out_path: if provided, saves the figure to this path.
	"""
	if cam.ndim == 2:
		cam = cam[0]

	length = cam.shape[0]
	fig, ax = plt.subplots(2 if input_seq is not None else 1, 1, figsize=figsize, gridspec_kw={"height_ratios": [1, 0.6]} if input_seq is not None else None)

	if input_seq is not None:
		ax_heat, ax_seq = ax[0], ax[1]
	else:
		ax_heat = ax

	im = ax_heat.imshow(cam[np.newaxis, :], aspect="auto", cmap=cmap, extent=[0, length, 0, 1])
	ax_heat.set_yticks([])
	ax_heat.set_xlabel("Position")
	if title:
		ax_heat.set_title(title)
	fig.colorbar(im, ax=ax_heat, orientation="vertical", fraction=0.02)

	if input_seq is not None:
		ax_seq.plot(np.arange(length), input_seq, color="k", linewidth=0.8)
		ax_seq.set_xlabel("Position")
		ax_seq.set_ylabel("Input value")

	plt.tight_layout()
	if out_path:
		plt.savefig(out_path, dpi=150)
		plt.close(fig)
	else:
		plt.show()


def run_gradcam_example(model, sample_dna, sample_gene, target_class=None, out_path=None):
	"""Convenience function to run Grad-CAM on a single sample and optionally save visualization.

	- sample_dna, sample_gene: 1D tensors (length,) or torch tensors shaped (1, length)
	"""
	cammer = GradCAM1D(model)
	# ensure batch dim
	if sample_dna.ndim == 1:
		sample_dna = sample_dna.unsqueeze(0)
	if sample_gene.ndim == 1:
		sample_gene = sample_gene.unsqueeze(0)

	cams = cammer.generate_cam(sample_dna, sample_gene, target_class=target_class)
	# use dna channel values as background if available
	inp = sample_dna.squeeze(0).cpu().numpy()
	if out_path:
		visualize_cam(cams, input_seq=inp, out_path=out_path)
	else:
		visualize_cam(cams, input_seq=inp)
	return cams


class InputGradientAttribution:
    """
    Compute per-gene importance using Input × Gradient method.
    
    This directly computes gradients of the classifier output w.r.t. each input
    gene position, giving true per-gene importance without spatial compression.
    
    The importance is: |input * gradient| (element-wise)
    We combine both channels (DNA methylation + gene expression) by summing.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device
    
    def attribute(
        self,
        x_dna: torch.Tensor,
        x_gene: torch.Tensor,
        target_class: t.Optional[int] = None,
        combine_channels: bool = True,
    ) -> np.ndarray:
        """
        Compute input gradient attribution for each gene position.
        
        Args:
            x_dna: (batch, length) DNA methylation values
            x_gene: (batch, length) Gene expression values
            target_class: If None, uses predicted class
            combine_channels: If True, sum attributions from both channels
            
        Returns:
            attributions: (batch, length) importance scores per gene
                         or (batch, 2, length) if combine_channels=False
        """
        self.model.eval()
        device = self.device
        
        x_dna = x_dna.to(device).float().requires_grad_(True)
        x_gene = x_gene.to(device).float().requires_grad_(True)
        
        # Forward pass
        if hasattr(self.model, "forward_with_classifier"):
            _, _, _, _, class_logits = self.model.forward_with_classifier(x_dna, x_gene)
        else:
            out = self.model(x_dna, x_gene)
            class_logits = out[-1] if isinstance(out, tuple) else out
        
        batch_size = x_dna.shape[0]
        
        # Get predictions if no target specified
        if target_class is None:
            preds = torch.argmax(class_logits, dim=1)
        
        all_attributions = []
        
        for i in range(batch_size):
            self.model.zero_grad()
            if x_dna.grad is not None:
                x_dna.grad.zero_()
            if x_gene.grad is not None:
                x_gene.grad.zero_()
            
            # Target class for this sample
            idx = target_class if target_class is not None else preds[i].item()
            
            # Get score and backprop
            score = class_logits[i, idx]
            score.backward(retain_graph=True)
            
            # Get gradients
            grad_dna = x_dna.grad[i].detach()   # (length,)
            grad_gene = x_gene.grad[i].detach() # (length,)
            
            # Input × Gradient (absolute value for importance magnitude)
            attr_dna = torch.abs(x_dna[i].detach() * grad_dna)
            attr_gene = torch.abs(x_gene[i].detach() * grad_gene)
            
            if combine_channels:
                # Sum both channels
                attr = attr_dna + attr_gene
                all_attributions.append(attr.cpu().numpy())
            else:
                attr = torch.stack([attr_dna, attr_gene], dim=0)
                all_attributions.append(attr.cpu().numpy())
        
        return np.stack(all_attributions, axis=0)


class IntegratedGradients:
    """
    Compute per-gene importance using Integrated Gradients.
    
    More accurate than simple Input×Gradient by integrating gradients
    along a path from a baseline (zeros) to the actual input.
    
    Reference: Sundararajan et al., "Axiomatic Attribution for Deep Networks"
    """
    
    def __init__(self, model: nn.Module, n_steps: int = 50):
        self.model = model
        self.device = next(model.parameters()).device
        self.n_steps = n_steps
    
    def attribute(
        self,
        x_dna: torch.Tensor,
        x_gene: torch.Tensor,
        target_class: t.Optional[int] = None,
        baseline_dna: t.Optional[torch.Tensor] = None,
        baseline_gene: t.Optional[torch.Tensor] = None,
        combine_channels: bool = True,
    ) -> np.ndarray:
        """
        Compute integrated gradient attribution for each gene position.
        
        Args:
            x_dna: (batch, length) DNA methylation values
            x_gene: (batch, length) Gene expression values
            target_class: If None, uses predicted class
            baseline_dna: Baseline for DNA channel (default: zeros)
            baseline_gene: Baseline for gene channel (default: zeros)
            combine_channels: If True, sum attributions from both channels
            
        Returns:
            attributions: (batch, length) importance scores per gene
        """
        self.model.eval()
        device = self.device
        
        x_dna = x_dna.to(device).float()
        x_gene = x_gene.to(device).float()
        
        batch_size, length = x_dna.shape
        
        # Default baselines are zeros
        if baseline_dna is None:
            baseline_dna = torch.zeros_like(x_dna)
        else:
            baseline_dna = baseline_dna.to(device).float()
        if baseline_gene is None:
            baseline_gene = torch.zeros_like(x_gene)
        else:
            baseline_gene = baseline_gene.to(device).float()
        
        # Get predictions for target class
        with torch.no_grad():
            if hasattr(self.model, "forward_with_classifier"):
                _, _, _, _, class_logits = self.model.forward_with_classifier(x_dna, x_gene)
            else:
                out = self.model(x_dna, x_gene)
                class_logits = out[-1] if isinstance(out, tuple) else out
            preds = torch.argmax(class_logits, dim=1)
        
        all_attributions = []
        
        for i in range(batch_size):
            # Target class
            idx = target_class if target_class is not None else preds[i].item()
            
            # Input and baseline for this sample
            inp_dna = x_dna[i:i+1]      # (1, length)
            inp_gene = x_gene[i:i+1]
            base_dna = baseline_dna[i:i+1]
            base_gene = baseline_gene[i:i+1]
            
            # Compute scaled inputs along the path
            # alpha goes from 0 to 1
            alphas = torch.linspace(0, 1, self.n_steps, device=device).view(-1, 1)
            
            # Interpolated inputs: (n_steps, length)
            scaled_dna = base_dna + alphas * (inp_dna - base_dna)
            scaled_gene = base_gene + alphas * (inp_gene - base_gene)
            
            scaled_dna.requires_grad_(True)
            scaled_gene.requires_grad_(True)
            
            # Forward pass on all steps
            if hasattr(self.model, "forward_with_classifier"):
                _, _, _, _, logits = self.model.forward_with_classifier(scaled_dna, scaled_gene)
            else:
                out = self.model(scaled_dna, scaled_gene)
                logits = out[-1] if isinstance(out, tuple) else out
            
            # Get scores for target class
            scores = logits[:, idx]  # (n_steps,)
            
            # Compute gradients
            grads_dna = torch.autograd.grad(
                scores.sum(), scaled_dna, retain_graph=True
            )[0]  # (n_steps, length)
            grads_gene = torch.autograd.grad(
                scores.sum(), scaled_gene
            )[0]  # (n_steps, length)
            
            # Average gradients along the path
            avg_grad_dna = grads_dna.mean(dim=0)   # (length,)
            avg_grad_gene = grads_gene.mean(dim=0)
            
            # Integrated gradients = (input - baseline) * avg_gradient
            ig_dna = (inp_dna.squeeze() - base_dna.squeeze()) * avg_grad_dna
            ig_gene = (inp_gene.squeeze() - base_gene.squeeze()) * avg_grad_gene
            
            # Take absolute value for importance magnitude
            ig_dna = torch.abs(ig_dna)
            ig_gene = torch.abs(ig_gene)
            
            if combine_channels:
                attr = ig_dna + ig_gene
                all_attributions.append(attr.detach().cpu().numpy())
            else:
                attr = torch.stack([ig_dna, ig_gene], dim=0)
                all_attributions.append(attr.detach().cpu().numpy())
        
        return np.stack(all_attributions, axis=0)


