import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import pickle

class DeepInsightTransformer:
    def __init__(self, resolution=224, perplexity=30, random_state=42):
        """
        args:
            resolution: 224x224 is standard for CNNs (ResNet/VGG) and 
                        gives space for 15k genes without too much overlap.
            perplexity: 30 is standard, controls local vs global clustering.
        """
        self.resolution = resolution
        self.perplexity = perplexity
        self.random_state = random_state
        self.pixel_coords = None

    def fit(self, X):
        """
        Step 1: Map 15,703 genes to 2D coordinates.
        Note: This runs ONCE.
        """
        # X shape is (Samples, Genes) -> (8472, 15703)
        # We transpose to (15703, 8472) to cluster the GENES
        X_genes = X.T 
        print(f"Fitting t-SNE on {X_genes.shape[0]} genes... this may take a few minutes.")
        
        tsne = TSNE(n_components=2, 
                    perplexity=self.perplexity, 
                    metric='cosine', 
                    n_jobs=-1, 
                    random_state=self.random_state)
        embedding = tsne.fit_transform(X_genes)
        
        # Scale to image coordinates [0, resolution]
        scaler = MinMaxScaler(feature_range=(0, self.resolution - 1))
        self.pixel_coords = scaler.fit_transform(embedding).astype(int)
        
        print("Feature mapping complete. Gene locations fixed.")
        return self

    def transform(self, X):
        """
        Step 2: Convert samples to images.
        """
        n_samples = X.shape[0]
        images = np.zeros((n_samples, self.resolution, self.resolution, 1))
        
        # Normalize expression values to [0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_norm = scaler.fit_transform(X)
        
        print(f"Transforming {n_samples} samples into images...")
        for i in range(n_samples):
            if i % 1000 == 0 and i > 0:
                print(f"Processed {i} samples...")
                
            sample_img = np.zeros((self.resolution, self.resolution))
            count_map = np.zeros((self.resolution, self.resolution))
            
            sample_vals = X_norm[i]
            
            # Vectorized mapping
            np.add.at(sample_img, (self.pixel_coords[:, 0], self.pixel_coords[:, 1]), sample_vals)
            np.add.at(count_map, (self.pixel_coords[:, 0], self.pixel_coords[:, 1]), 1)
            
            # Average overlaps
            mask = count_map > 0
            sample_img[mask] /= count_map[mask]
            
            images[i, :, :, 0] = sample_img
            
        return images

# ==========================================
# 1. LOAD YOUR DATA
# ==========================================
# Uncomment this block to use your actual file:

with open('/home/dmlab/Devendra/data/preprocessed_datasets/gene_expression_tensor_tcga.pkl', 'rb') as f:
    # Assuming pkl is a tuple/dict with data and labels
    X= pickle.load(f)
    # Shape (8472, 15703)
with open('/home/dmlab/Devendra/data/preprocessed_datasets/cancer_tags_tensor_tcga.pkl','rb') as g:          # Shape (8472,) with 0 or 1 for cancer types
    y= pickle.load(g)
         

# --- DUMMY DATA GENERATOR (Matches your shape) ---
# Use this to test the pipeline before loading your real 2GB+ file

# Inject fake biological signals so we can see a difference
# Cancer Type A (Label 0): High expression in gene cluster 1

# -----------------------------------------------

# ==========================================
# 2. RUN DEEPINSIGHT
# ==========================================

# Initialize with 224x224 resolution for better clarity
transformer = DeepInsightTransformer(resolution=224, perplexity=30)

# Fit (Find gene locations)
transformer.fit(X)

# Transform (Create images)
# For visualization, we only transform the first 100 to save time in testing
# In production, use: image_data = transformer.transform(X)
image_data = transformer.transform(X[:200]) 

print(f"Output Tensor Shape: {image_data.shape}")

# ==========================================
# 3. VISUALIZE CANCER TYPES (Corrected)
# ==========================================

# 1. Extract valid integer indices
# If y is One-Hot Encoded (8472, 28):
# Cancer Type A is where column 0 has a 1.
# Cancer Type B is where column 1 has a 1.
try:
    # Find first sample index where column 0 is active
    indices_a = np.where(y[:200, 0] == 1)[0] 
    # Find first sample index where column 1 is active
    indices_b = np.where(y[:200, 1] == 1)[0]

    if len(indices_a) == 0 or len(indices_b) == 0:
        raise ValueError("Could not find samples for both cancer types in the first 200 rows.")

    idx_a = indices_a[0] # Get the first integer index
    idx_b = indices_b[0] # Get the first integer index
    
    print(f"Plotting Sample Index A: {idx_a}, Sample Index B: {idx_b}")

except IndexError:
    print("Error: Check your 'y' shape. Is it One-Hot Encoded or Label Encoded?")
    # Fallback if y is simple Label Encoded (8472,)
    idx_a = np.where(y[:200] == 0)[0][0]
    idx_b = np.where(y[:200] == 1)[0][0]

# 2. Create the figure
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Get the 2D images (Squeeze the channel dimension for imshow)
img_a = image_data[idx_a, :, :, 0]
img_b = image_data[idx_b, :, :, 0]

# Plot Cancer A
im0 = axes[0].imshow(img_a, cmap='inferno')
axes[0].set_title(f"Cancer Type A (Index {idx_a})")
axes[0].axis('off')
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# Plot Cancer B
im1 = axes[1].imshow(img_b, cmap='inferno')
axes[1].set_title(f"Cancer Type B (Index {idx_b})")
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# Plot Difference (Type A - Type B)
diff = img_a - img_b
v_lim = np.max(np.abs(diff)) * 0.8  
im2 = axes[2].imshow(diff, cmap='coolwarm', vmin=-v_lim, vmax=v_lim)
axes[2].set_title("Differential Expression (A vs B)")
axes[2].axis('off')
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

# 3. Save
plt.tight_layout()
plt.savefig('cancer_comparison_plot.png', dpi=300)
plt.close()

print("Plot successfully saved as 'cancer_comparison_plot.png'")
