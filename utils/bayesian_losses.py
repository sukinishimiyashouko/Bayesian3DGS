#
# Bayesian Losses for 3D Gaussian Splatting
#
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
from cuda_config import devices


class BayesianLossCalculator:
    def __init__(self, dataset, lambda_kl=0.01, lambda_geo=0.1, lambda_sem=0.1, p0=0.5):
        """
        Initialize the Bayesian loss calculator with hyperparameters.
        
        Parameters:
        - lambda_kl: Weight for KL divergence term
        - lambda_geo: Weight for geometric consistency term
        - lambda_sem: Weight for semantic consistency term
        - p0: Prior probability for Gaussian survival
        """
        self.lambda_kl = lambda_kl
        self.lambda_geo = lambda_geo
        self.lambda_sem = lambda_sem
        self.p0 = p0

        # Load CLIP model for semantic feature extraction
        if dataset.use_semantic:
            try:
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=devices.train_device())
                self.clip_model.eval()
            except:
                print("Warning: CLIP model could not be loaded. Semantic loss will be skipped.")
                self.clip_model = None

            # Define image transformations for feature extraction
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711]),
            ])

    def kl_divergence(self, survival_prob):
        """
        Compute KL divergence between variational distribution q(Z) and prior p(Z).
        
        Parameters:
        - survival_prob: Tensor of survival probabilities (phi_i) for each Gaussian
        
        Returns:
        - kl_loss: KL divergence loss
        """
        # Avoid numerical instability with small epsilon
        eps = 1e-10
        phi = torch.clamp(survival_prob, eps, 1 - eps)

        # KL divergence: KL(q||p) = E_q[log q/p]
        kl_loss = phi * torch.log(phi / self.p0) + (1 - phi) * torch.log((1 - phi) / (1 - self.p0))
        return kl_loss.mean()

    def geometric_consistency_loss(self, viewpoint_camera, neighboring_camera, gaussian_model, pipe, bg_color):
        """
        Compute geometric consistency loss between two views.
        
        Parameters:
        - viewpoint_camera: Main camera
        - neighboring_camera: Neighboring camera (for consistency check)
        - gaussian_model: Gaussian model
        - pipe: Pipeline settings
        - bg_color: Background color
        
        Returns:
        - geo_loss: Geometric consistency loss
        """
        # Render both views using expectation rendering
        render_result1 = render(viewpoint_camera, gaussian_model, pipe, bg_color, bayesian_rendering=True)
        render_result2 = render(neighboring_camera, gaussian_model, pipe, bg_color, bayesian_rendering=True)

        # Extract rendered images
        img1 = render_result1["render"]
        img2 = render_result2["render"]

        # Compute consistency loss in feature space (using simple L2 for now)
        # In practice, you might want to use a pre-trained feature extractor
        geo_loss = F.mse_loss(img1, img2)

        return geo_loss

    def semantic_consistency_loss(self, rendered_image, target_image):
        """
        Compute semantic consistency loss using CLIP features.
        
        Parameters:
        - rendered_image: Rendered image from the model
        - target_image: Ground truth image
        
        Returns:
        - sem_loss: Semantic consistency loss
        """
        if self.clip_model is None:
            return torch.tensor(0.0, device=devices.train_device())

        # Prepare images for CLIP
        # Resize to 224x224 and normalize
        rendered_img_resized = F.interpolate(rendered_image.unsqueeze(0), size=(224, 224), mode='bilinear',
                                             align_corners=False)
        target_img_resized = F.interpolate(target_image.unsqueeze(0), size=(224, 224), mode='bilinear',
                                           align_corners=False)

        # Normalize using CLIP's normalization parameters
        rendered_normalized = self.image_transform(rendered_img_resized.squeeze(0))
        target_normalized = self.image_transform(target_img_resized.squeeze(0))

        # Extract CLIP features
        with torch.no_grad():
            rendered_features = self.clip_model.encode_image(rendered_normalized.unsqueeze(0))
            target_features = self.clip_model.encode_image(target_normalized.unsqueeze(0))

        # Normalize features
        rendered_features = F.normalize(rendered_features, dim=-1)
        target_features = F.normalize(target_features, dim=-1)

        # Compute cosine similarity (1 - similarity as loss)
        cos_sim = F.cosine_similarity(rendered_features, target_features)
        sem_loss = 1 - cos_sim.mean()

        return sem_loss

    def compute_total_loss(self, render_result, target_image, gaussian_model,
                           viewpoint_camera=None, neighboring_camera=None, pipe=None, bg_color=None):
        """
        Compute the total loss for Bayesian 3DGS.
        
        Parameters:
        - render_result: Result from the render function
        - target_image: Ground truth image
        - gaussian_model: Gaussian model
        - viewpoint_camera: Main camera (for geometric loss)
        - neighboring_camera: Neighboring camera (for geometric loss)
        - pipe: Pipeline settings
        - bg_color: Background color
        
        Returns:
        - total_loss: Total loss combining all components
        - loss_components: Dictionary of individual loss components
        """
        # Reconstruction loss (L2)
        recon_loss = F.mse_loss(render_result["render"], target_image)

        # KL divergence loss
        kl_loss = 0.0
        if hasattr(gaussian_model, '_survival_logit') and gaussian_model._survival_logit.shape[0] > 0:
            survival_prob = gaussian_model.survival_prob_activation(gaussian_model._survival_logit)
            kl_loss = self.kl_divergence(survival_prob)

        # Geometric consistency loss
        geo_loss = 0.0
        if viewpoint_camera is not None and neighboring_camera is not None and pipe is not None:
            geo_loss = self.geometric_consistency_loss(viewpoint_camera, neighboring_camera,
                                                       gaussian_model, pipe, bg_color)

        # Semantic consistency loss
        sem_loss = 0.0
        if self.clip_model is not None:
            sem_loss = self.semantic_consistency_loss(render_result["render"], target_image)

        # Compute total loss
        total_loss = recon_loss + self.lambda_kl * kl_loss + \
                     self.lambda_geo * geo_loss + self.lambda_sem * sem_loss

        # Store individual components for logging
        loss_components = {
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
            "geo_loss": geo_loss.item() if isinstance(geo_loss, torch.Tensor) else geo_loss,
            "sem_loss": sem_loss.item() if isinstance(sem_loss, torch.Tensor) else sem_loss
        }

        return total_loss, loss_components


# Import needed for geometric consistency loss
from gaussian_renderer import render


# Standalone functions for easy integration with existing training code
def kl_divergence_loss(survival_prob, prior_p=0.5):
    """
    Compute KL divergence between variational distribution q(Z) and prior p(Z).
    
    Parameters:
    - survival_prob: Tensor of survival probabilities (phi_i) for each Gaussian
    - prior_p: Prior probability for Gaussian survival
    
    Returns:
    - kl_loss: KL divergence loss
    """
    # Avoid numerical instability with small epsilon
    eps = 1e-10
    phi = torch.clamp(survival_prob, eps, 1 - eps)

    # KL divergence: KL(q||p) = E_q[log q/p]
    # phi * torch.log(phi / prior_p):生存部分的KL贡献（点被保留时的差异）
    # (1 - phi) * torch.log((1 - phi) / (1 - prior_p)):移除部分的KL贡献（点被剪枝时的差异）
    kl_loss = phi * torch.log(phi / prior_p) + (1 - phi) * torch.log((1 - phi) / (1 - prior_p))
    return kl_loss.mean()


def geometric_consistency_loss(image1, image2):
    """
    Compute geometric consistency loss between two rendered images.
    
    Parameters:
    - image1: First rendered image
    - image2: Second rendered image
    
    Returns:
    - geo_loss: Geometric consistency loss
    """
    # Using MSE loss for simplicity
    return F.mse_loss(image1, image2)


def semantic_loss(rendered_image, target_image):
    """
    Simple semantic loss based on image similarity.
    In a real implementation, you might want to use a more sophisticated approach
    with feature extraction.
    
    Parameters:
    - rendered_image: Rendered image from the model
    - target_image: Ground truth image
    
    Returns:
    - sem_loss: Semantic loss
    """
    # Using L1 loss as a simple semantic proxy
    return F.l1_loss(rendered_image, target_image)


# Create a default instance of the BayesianLossCalculator for convenience
# default_bayesian_calculator = BayesianLossCalculator()
