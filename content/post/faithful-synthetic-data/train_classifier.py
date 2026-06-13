"""One-class neural network training in marimo notebook."""

import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def imports():
    """Import required libraries."""
    from __future__ import annotations # noqa:F404

    import logging
    from pathlib import Path
    from typing import Tuple

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split
    from tqdm.auto import tqdm
    from datasets import load_dataset
    from torchvision import transforms

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return (
        DataLoader,
        Path,
        TensorDataset,
        Tuple,
        load_dataset,
        logger,
        nn,
        np,
        random_split,
        torch,
        tqdm,
        transforms,
    )


@app.cell
def device_utils(logger, torch):
    """Device detection utilities."""

    def get_device() -> torch.device:
        """Get the best available device (MPS for Apple Silicon, CUDA, or CPU).

        Returns:
            torch.device: The device to use for training.
        """
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        logger.info(f"Using device: {device}")
        return device
    return (get_device,)


@app.cell
def model_definition(nn, torch):
    """Define the one-class neural network architecture."""

    class OneClassNet(nn.Module):
        """Simple feedforward neural network for one-class classification.

        Attributes:
            input_dim: Dimension of input features.
            hidden_dim: Dimension of hidden layers.
            rep_dim: Dimension of representation (output) layer.
            num_layers: Number of hidden layers.
            dropout_prob: Dropout probability.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 128,
            rep_dim: int = 32,
            num_layers: int = 3,
            dropout_prob: float = 0.2,
        ):
            """Initialize the one-class network.

            Args:
                input_dim: Dimension of input features.
                hidden_dim: Dimension of hidden layers. Defaults to 128.
                rep_dim: Dimension of representation layer. Defaults to 32.
                num_layers: Number of hidden layers. Defaults to 3.
                dropout_prob: Dropout probability. Defaults to 0.2.
            """
            super().__init__()

            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.rep_dim = rep_dim

            # Build network layers
            layers = []

            # Input layer
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_prob),
                ]
            )

            # Hidden layers
            for _ in range(num_layers - 1):
                layers.extend(
                    [
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout_prob),
                    ]
                )

            # Output layer (representation)
            layers.append(nn.Linear(hidden_dim, rep_dim))

            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through the network.

            Args:
                x: Input tensor of shape (batch_size, input_dim).

            Returns:
                Output tensor of shape (batch_size, rep_dim).
            """
            return self.network(x)
    return (OneClassNet,)


@app.cell
def loss_functions(torch):
    """Define one-class loss functions."""

    def soft_boundary_loss(
        outputs: torch.Tensor,
        center: torch.Tensor,
        radius: float,
        nu: float = 0.1,
    ) -> torch.Tensor:
        """Compute soft-boundary one-class loss.

        This loss function is inspired by one-class SVMs and learns a hypersphere
        around the data with radius R and center c.

        Args:
            outputs: Network outputs of shape (batch_size, rep_dim).
            center: Center of the hypersphere of shape (rep_dim,).
            radius: Radius of the hypersphere.
            nu: Fraction of outliers/tolerance. Defaults to 0.1.

        Returns:
            Scalar loss value.
        """
        # Compute squared distances to center
        dist = torch.sum((outputs - center) ** 2, dim=1)

        # Compute scores (how much points violate the boundary)
        scores = dist - radius**2

        # Penalize violations
        loss = radius**2 + (1 / nu) * torch.mean(torch.relu(scores))

        return loss

    def one_class_loss(
        outputs: torch.Tensor,
        center: torch.Tensor,
    ) -> torch.Tensor:
        """Compute simple one-class loss (mean distance to center).

        Args:
            outputs: Network outputs of shape (batch_size, rep_dim).
            center: Center of the hypersphere of shape (rep_dim).

        Returns:
            Scalar loss value.
        """
        dist = torch.sum((outputs - center) ** 2, dim=1)
        return torch.mean(dist)

    def get_radius(
        dist: torch.Tensor,
        nu: float,
    ) -> float:
        """Compute optimal radius via (1-nu)-quantile of distances.

        Efficiently computes quantile directly on tensor without CPU transfer.

        Args:
            dist: Tensor of squared distances.
            nu: Quantile parameter (fraction of outliers).

        Returns:
            Optimal radius value.
        """
        # Use torch.quantile to stay on device (MPS/CUDA/CPU)
        return float(torch.quantile(torch.sqrt(dist), 1 - nu).item())
    return get_radius, one_class_loss, soft_boundary_loss


@app.cell
def training_utils(DataLoader, Tuple, nn, torch, tqdm):
    """Training and evaluation utilities."""

    def initialize_center(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        eps: float = 0.1,
    ) -> torch.Tensor:
        """Initialize hypersphere center as mean of network outputs.

        Args:
            model: The neural network model.
            dataloader: DataLoader with training data.
            device: Device to use for computation.
            eps: Small constant to avoid centers at 0. Defaults to 0.1.

        Returns:
            Center tensor of shape (rep_dim,).
        """
        model.eval()
        centers = []

        with torch.no_grad():
            for batch in dataloader:
                inputs_loc = batch[0].to(device)
                outputs = model(inputs_loc)
                centers.append(outputs)

        center = torch.mean(torch.cat(centers, dim=0), dim=0)

        # Avoid centers at exactly 0
        center[(torch.abs(center) < eps) & (center < 0)] = -eps
        center[(torch.abs(center) < eps) & (center >= 0)] = eps

        return center

    def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer,
        center: torch.Tensor,
        radius: float,
        nu: float,
        device: torch.device,
        loss_fn,
        loss_type: str = "SoftBoundary",
    ) -> Tuple[float, float]:
        """Train for one epoch.

        Args:
            model: The neural network model.
            dataloader: DataLoader with training data.
            optimizer: Optimizer for training.
            center: Center of the hypersphere.
            radius: Radius of the hypersphere.
            nu: Outlier fraction parameter.
            device: Device to use for computation.
            loss_fn: Loss function (soft_boundary_loss or one_class_loss).
            loss_type: Type of loss. Defaults to "SoftBoundary".

        Returns:
            Tuple of (average loss, updated radius).
        """
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Training", leave=False):
            inputs = batch[0].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            if loss_type == "SoftBoundary":
                loss = loss_fn(outputs, center, radius, nu)
            else:
                loss = loss_fn(outputs, center)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss, radius

    def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        center: torch.Tensor,
        radius: float,
        nu: float,
        device: torch.device,
        loss_fn,
        loss_type: str = "SoftBoundary",
    ) -> float:
        """Evaluate the model on validation data.

        Args:
            model: The neural network model.
            dataloader: DataLoader with validation data.
            center: Center of the hypersphere.
            radius: Radius of the hypersphere.
            nu: Outlier fraction parameter.
            device: Device to use for computation.
            loss_fn: Loss function.
            loss_type: Type of loss. Defaults to "SoftBoundary".

        Returns:
            Average validation loss.
        """
        model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs_fwd = batch[0].to(device)
                outputs = model(inputs_fwd)

                if loss_type == "SoftBoundary":
                    loss = loss_fn(outputs, center, radius, nu)
                else:
                    loss = loss_fn(outputs, center)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches
    return evaluate, initialize_center, train_epoch


@app.cell
def save_checkpoint(Path, torch):
    """Checkpoint saving utility."""

    def save_model_checkpoint(
        model,
        center: torch.Tensor,
        radius: float,
        history: dict,
        save_path: str = "one_class_model.pth",
    ) -> None:
        """Save model checkpoint to disk.

        Args:
            model: The trained model.
            center: Hypersphere center.
            radius: Hypersphere radius.
            history: Training history dict.
            save_path: Path to save checkpoint. Defaults to "one_class_model.pth".
        """
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "center": center.cpu(),
            "radius": radius,
            "history": history,
            "model_config": {
                "input_dim": model.input_dim,
                "hidden_dim": model.hidden_dim,
                "rep_dim": model.rep_dim,
            },
        }

        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, save_path)
        print(f"✅ Model checkpoint saved to: {save_path}")
    return (save_model_checkpoint,)


@app.cell
def load_data(load_dataset, logger):
    """Load the CIFAKE dataset."""
    logger.info("Loading CIFAKE dataset...")
    dataset_hf = load_dataset("dragonintelligence/CIFAKE-image-dataset")

    real_img = dataset_hf["train"]["image"]  # Label 0 is Real
    fake_img = dataset_hf["train"]["image"]  # Label 1 is Fake

    print(f"Dataset: {dataset_hf}")
    print(f"Real image shape: {real_img}")
    print(f"Fake image shape: {fake_img}")
    return (dataset_hf,)


@app.cell
def prepare_training_data(dataset_hf, np, transforms):
    """Prepare training data efficiently using HF Dataset mapping with transforms."""
    n_samples_train = 60_000
    input_dim = 32 * 32 * 1  # 32x32x1 (Grayscale)

    # Define transforms: Grayscale + ToTensor + Normalize
    # ToTensor converts [0, 255] -> [0.0, 1.0]
    # Normalize((0.5,), (0.5,)) converts [0.0, 1.0] -> [-1.0, 1.0]
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Filter for REAL images (label = 1)
    ds_real = dataset_hf['train'].filter(lambda x: x['label'] == 1)

    # Select subset (limit to available number of samples)
    n_samples = min(n_samples_train, len(ds_real))
    subset = ds_real.select(range(n_samples))

    def process_images(examples):
        # Apply transforms and flatten
        processed_imgs = []
        for img in examples['image']:
            # Apply transform
            tensor_img = transform(img)
            # Flatten: [1, 32, 32] -> [1024]
            flat_img = tensor_img.flatten().numpy()
            processed_imgs.append(flat_img)

        return {'features': processed_imgs}

    print("🚀 Processing images (Grayscale + Normalize)...")
    subset = subset.map(
        process_images,
        batched=True,
        batch_size=100,
        remove_columns=subset.column_names,
        desc="Preprocessing images"
    )

    # Zero-copy conversion to numpy
    subset.set_format("numpy")
    x_train = np.array(subset['features'])

    print(f"Training data shape: {x_train.shape}")
    print(f"Min value: {x_train.min():.2f}, Max value: {x_train.max():.2f}")
    return input_dim, x_train


@app.cell
def train_model(
    DataLoader,
    OneClassNet,
    TensorDataset,
    evaluate,
    get_device,
    get_radius,
    initialize_center,
    input_dim,
    logger,
    one_class_loss,
    random_split,
    save_model_checkpoint,
    soft_boundary_loss,
    torch,
    tqdm,
    train_epoch,
    x_train,
):
    """Main training cell."""

    # Hyperparameters
    HIDDEN_DIM = 128
    REP_DIM = 32
    NUM_LAYERS = 3
    DROPOUT_PROB = 0.2
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    BATCH_SIZE = 128
    EPOCHS = 50
    VAL_SPLIT = 0.1
    NU = 0.1
    WARM_UP_EPOCHS = 10
    LOSS_TYPE = "SoftBoundary"

    # Get device
    device = get_device()

    # Prepare data
    x_tensor = torch.from_numpy(x_train).float()
    dataset = TensorDataset(x_tensor)

    # Split into train and validation
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(device.type != "mps"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=(device.type != "mps"),
    )

    # Initialize model
    model = OneClassNet(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        rep_dim=REP_DIM,
        num_layers=NUM_LAYERS,
        dropout_prob=DROPOUT_PROB,
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Select loss function
    loss_fn = soft_boundary_loss if LOSS_TYPE == "SoftBoundary" else one_class_loss

    # Initialize center
    logger.info("Initializing hypersphere center...")
    center = initialize_center(model, train_loader, device).to(device)
    radius = 0.0

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
    }

    # Training loop
    logger.info(f"Starting training for {EPOCHS} epochs...")
    for epoch in tqdm(range(EPOCHS), desc="Epochs"):
        # Train
        train_loss, radius = train_epoch(
            model,
            train_loader,
            optimizer,
            center,
            radius,
            NU,
            device,
            loss_fn,
            LOSS_TYPE,
        )
        history["train_loss"].append(train_loss)

        # Update radius after warm-up
        if epoch >= WARM_UP_EPOCHS and LOSS_TYPE == "SoftBoundary":
            model.eval()
            with torch.no_grad():
                dists = []
                for batch in train_loader:
                    inputs = batch[0].to(device)
                    outputs = model(inputs)
                    dist = torch.sum((outputs - center) ** 2, dim=1)
                    dists.append(dist)
                all_dists = torch.cat(dists)
                radius = get_radius(all_dists, NU)

        # Validate
        val_loss = evaluate(
            model,
            val_loader,
            center,
            radius,
            NU,
            device,
            loss_fn,
            LOSS_TYPE,
        )
        history["val_loss"].append(val_loss)

        # Log progress
        if (epoch + 1) % 5 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{EPOCHS} - "
                f"Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}, "
                f"Radius: {radius:.4f}"
            )

    logger.info("✅ Training complete!")

    # Save checkpoint
    save_model_checkpoint(
        model,
        center,
        radius,
        history,
        save_path="checkpoints/one_class_model_final.pth",
    )
    return (history,)


@app.cell
def plot_results(history):
    import matplotlib.pyplot as plt
    import matplotx

    # Nur die Basis-Farben für Text/Hintergrund/Grid werden durch den Stil gesetzt.
    # Spezifische Linien- und Textfarben werden entfernt.
    # Die Linien übernehmen die Standardfarben des Stils.

    # --- Use matplotx 'github' style in a context manager ---
    with plt.style.context(matplotx.styles.github["light"]):
        fig, ax = plt.subplots(figsize=(12, 7))

        epochs = range(1, len(history["train_loss"]) + 1)

        # Plot lines with markers
        # 'color' Parameter entfernt: Linienfarbe wird durch den Stil bestimmt (Standard-Farbzyklus)
        ax.plot(
            epochs,
            history["train_loss"],
            linewidth=2.5,
            marker="o",
            markersize=4,
            markevery=max(1, len(epochs) // 20),
            label="Train Loss",
            zorder=3,
        )[0]

        ax.plot(
            epochs,
            history["val_loss"],
            linewidth=2.5,
            marker="s",
            markersize=4,
            markevery=max(1, len(epochs) // 20),
            label="Val Loss",
            zorder=3,
        )[0]

        # Add direct labels on lines (at the end)
        # 'color' Parameter entfernt: Textfarbe wird auf die Standard-Vordergrundfarbe des Stils gesetzt
        ax.text(
            len(epochs),
            history["train_loss"][-1],
            "  Train",
            fontsize=11,
            fontweight="bold",
            va="center",
            ha="left",
        )

        ax.text(
            len(epochs),
            history["val_loss"][-1],
            "  Validation",
            fontsize=11,
            fontweight="bold",
            va="center",
            ha="left",
        )

        # Styling
        # Der Stil übernimmt die Farben der Achsenbeschriftungen
        ax.set_xlabel("Epoch", fontsize=13, fontweight="bold")
        ax.set_ylabel("Loss", fontsize=13, fontweight="bold")
        ax.set_title(
            "One-Class Network Training History", fontsize=15, fontweight="bold", pad=20
        )

        # Grid (Die Farbe des Grids wird durch den Stil bestimmt, falls nicht explizit gesetzt)
        ax.grid(True, alpha=0.5, linestyle="-", linewidth=1)
        ax.set_axisbelow(True)

        # Die Spine- und Tick-Farben werden automatisch vom Stil gehandhabt.

        plt.tight_layout()
        plt.show()
    return


@app.cell
def embedding_projector_export():
    # """Generate TensorFlow Embedding Projector visualization files.

    # Creates:
    # - vectors.tsv: t-SNE reduced embeddings
    # - metadata.tsv: labels (real/fake)
    # - sprite.png: sprite sheet of images
    # """
    # from PIL import Image
    # from sklearn.manifold import TSNE
    # from transformers import CLIPProcessor, CLIPModel
    # import math

    # print("🔄 Loading CLIP model...")
    # clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # clip_model.eval()

    # # Sample images (adjust n_samples as needed)
    # n_samples = 1000  # 500 real + 500 fake
    # n_real = n_samples // 2
    # n_fake = n_samples // 2

    # print(f"📸 Sampling {n_samples} images ({n_real} real, {n_fake} fake)...")

    # # Filter datasets
    # ds_real = dataset['train'].filter(lambda x: x['label'] == 1)
    # ds_fake = dataset['train'].filter(lambda x: x['label'] == 0)

    # images = []
    # labels = []

    # # Sample real images
    # for i in range(min(n_real, len(ds_real))):
    #     img = ds_real[i]['image']
    #     images.append(img)
    #     labels.append('real')

    # # Sample fake images
    # for i in range(min(n_fake, len(ds_fake))):
    #     img = ds_fake[i]['image']
    #     images.append(img)
    #     labels.append('fake')

    # # Extract CLIP embeddings
    # print("🧠 Extracting CLIP embeddings...")
    # embeddings = []
    # batch_size = 32

    # for i in range(0, len(images), batch_size):
    #     batch_images = images[i : i + batch_size]
    #     inputs_clip = clip_processor(images=batch_images, return_tensors="pt", padding=True)

    #     with torch.no_grad():
    #         image_features = clip_model.get_image_features(**inputs_clip)
    #         embeddings.append(image_features.cpu().numpy())

    # embeddings = np.vstack(embeddings)
    # print(f"✅ CLIP embeddings shape: {embeddings.shape}")

    # # Apply t-SNE
    # print("📊 Applying t-SNE dimensionality reduction...")
    # tsne = TSNE(
    #     n_components=3,  # 3D for better visualization
    #     perplexity=30,
    #     n_iter=1000,
    #     random_state=42,
    #     verbose=1,
    # )
    # embeddings_tsne = tsne.fit_transform(embeddings)
    # print(f"✅ t-SNE embeddings shape: {embeddings_tsne.shape}")

    # # Create output directory
    # output_dir = Path("embedding_projector")
    # output_dir.mkdir(exist_ok=True)

    # # Save vectors.tsv
    # print("💾 Saving vectors.tsv...")
    # np.savetxt(output_dir / "vectors.tsv", embeddings_tsne, delimiter="\t", fmt="%.6f")

    # # Save metadata.tsv
    # print("💾 Saving metadata.tsv...")
    # with open(output_dir / "metadata.tsv", "w") as f:
    #     f.write("label\n")  # Header
    #     for label in labels:
    #         f.write(f"{label}\n")

    # # Create sprite sheet
    # print("🖼️  Creating sprite sheet...")
    # sprite_size = 64  # Size of each thumbnail
    # images_per_row = int(math.ceil(math.sqrt(n_samples)))

    # # Resize all images
    # thumbnails = []
    # for img in images:
    #     # Convert to RGB if needed
    #     if img.mode != "RGB":
    #         img = img.convert("RGB")
    #     # Resize to sprite_size x sprite_size
    #     thumb = img.resize((sprite_size, sprite_size), Image.Resampling.LANCZOS)
    #     thumbnails.append(np.array(thumb))

    # # Create sprite sheet
    # sprite_height = images_per_row * sprite_size
    # sprite_width = images_per_row * sprite_size
    # sprite = np.zeros((sprite_height, sprite_width, 3), dtype=np.uint8)

    # for idx, thumb in enumerate(thumbnails):
    #     row = idx // images_per_row
    #     col = idx % images_per_row
    #     y = row * sprite_size
    #     x = col * sprite_size
    #     sprite[y : y + sprite_size, x : x + sprite_size] = thumb

    # # Save sprite
    # sprite_img = Image.fromarray(sprite)
    # sprite_img.save(output_dir / "sprite.png")
    # print(f"✅ Sprite sheet saved: {sprite_size}x{sprite_size} thumbnails")

    # # Create projector_config.pbtxt
    # print("📝 Creating projector_config.pbtxt...")
    # config_content = f"""embeddings {{
    #   tensor_path: "vectors.tsv"
    #   metadata_path: "metadata.tsv"
    #   sprite {{
    # image_path: "sprite.png"
    # single_image_dim: {sprite_size}
    # single_image_dim: {sprite_size}
    #   }}
    # }}
    # """
    # with open(output_dir / "projector_config.pbtxt", "w") as f:
    #     f.write(config_content)

    # print("\n✅ TensorFlow Embedding Projector files created!")
    # print(f"📁 Output directory: {output_dir.absolute()}")
    # print("\n📖 To visualize:")
    # print("1. Go to: https://projector.tensorflow.org/")
    # print("2. Click 'Load' in the left panel")
    # print(f"3. Upload all files from: {output_dir.absolute()}")
    # print("   - vectors.tsv")
    # print("   - metadata.tsv")
    # print("   - sprite.png")
    # print("   - projector_config.pbtxt (optional)")
    return


if __name__ == "__main__":
    app.run()
