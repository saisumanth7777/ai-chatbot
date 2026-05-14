"""
Phase 1 — Simple GAN (Generative Adversarial Network)
======================================================
CONCEPT: A GAN trains two networks against each other:
  - Generator  : takes random noise → outputs fake data
  - Discriminator: takes data (real or fake) → outputs a probability (real=1, fake=0)

We train on a simple 1-D distribution (a sine wave) so you can run this on any
laptop without a GPU. The same pattern scales to images, text, etc.

Run: python phase1_generative_ai/simple_gan.py
"""

import torch
import torch.nn as nn
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Define what "real" data looks like
# ─────────────────────────────────────────────────────────────────────────────
def real_data_samples(batch_size: int) -> torch.Tensor:
    """
    Returns a batch of 'real' data points.
    We use points sampled from y = sin(x) — a simple, learnable distribution.
    The Generator's job is to learn to produce numbers that look like these.
    """
    x = np.random.uniform(-np.pi, np.pi, batch_size)
    y = np.sin(x)
    # Stack x and y into pairs so each sample is [x, sin(x)]
    data = np.column_stack([x, y]).astype(np.float32)
    return torch.tensor(data)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Build the Generator network
# ─────────────────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    """
    CONCEPT — Generator:
    Takes random noise (latent vector) as input and outputs data that
    should look like the real distribution.

    Architecture: noise (latent_dim) → hidden layers → output (data_dim)
    """

    def __init__(self, latent_dim: int = 8, data_dim: int = 2):
        super().__init__()
        # nn.Sequential chains layers in order — input flows through each one
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 16),  # latent noise → hidden layer with 16 neurons
            nn.ReLU(),                   # ReLU activation: max(0, x) — adds non-linearity
            nn.Linear(16, 16),           # hidden → hidden
            nn.ReLU(),
            nn.Linear(16, data_dim),     # hidden → output (same shape as real data)
            nn.Tanh(),                   # Tanh squashes output to [-1, 1]
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.network(noise)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Build the Discriminator network
# ─────────────────────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    """
    CONCEPT — Discriminator:
    Takes a data point (real or fake) and outputs a single number between 0–1.
      - Close to 1 → "I think this is real"
      - Close to 0 → "I think this is fake"

    Architecture: data (data_dim) → hidden layers → probability (1 number)
    """

    def __init__(self, data_dim: int = 2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(data_dim, 16),
            nn.LeakyReLU(0.2),          # LeakyReLU lets small negatives through — better for Discriminators
            nn.Linear(16, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid(),               # Sigmoid squashes to [0, 1] — gives us a probability
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.network(data)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train_gan(epochs: int = 2000, batch_size: int = 64, latent_dim: int = 8):
    """
    CONCEPT — Training a GAN:
    Each training step has two phases:

    Phase A — Train the Discriminator:
      1. Feed it real data  → it should output ~1  (loss = how far from 1)
      2. Feed it fake data  → it should output ~0  (loss = how far from 0)
      3. Update Discriminator weights to get better at classifying

    Phase B — Train the Generator:
      1. Generate fake data
      2. Feed it to the Discriminator
      3. Generator's goal: fool the Discriminator into outputting ~1
      4. Update Generator weights to get better at fooling
    """

    generator = Generator(latent_dim=latent_dim)
    discriminator = Discriminator()

    # Binary Cross-Entropy loss — standard loss for binary classification (real vs fake)
    loss_fn = nn.BCELoss()

    # Adam optimizer adapts learning rates per parameter — much better than plain SGD
    opt_g = torch.optim.Adam(generator.parameters(), lr=0.001)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    print("Training GAN...\n")
    print(f"{'Epoch':>6} | {'D Loss':>8} | {'G Loss':>8}")
    print("-" * 30)

    for epoch in range(1, epochs + 1):

        # ── Phase A: Train Discriminator ─────────────────────────────────────
        opt_d.zero_grad()  # clear old gradients before computing new ones

        real = real_data_samples(batch_size)
        real_labels = torch.ones(batch_size, 1)   # real data should be classified as 1
        fake_labels = torch.zeros(batch_size, 1)  # fake data should be classified as 0

        # Discriminator on real data
        real_loss = loss_fn(discriminator(real), real_labels)

        # Generate noise → fake data (detach so Generator gradients don't flow here)
        noise = torch.randn(batch_size, latent_dim)
        fake = generator(noise).detach()
        fake_loss = loss_fn(discriminator(fake), fake_labels)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        opt_d.step()

        # ── Phase B: Train Generator ──────────────────────────────────────────
        opt_g.zero_grad()

        noise = torch.randn(batch_size, latent_dim)
        fake = generator(noise)
        # Generator wants discriminator to call its output "real" (label=1)
        g_loss = loss_fn(discriminator(fake), real_labels)

        g_loss.backward()
        opt_g.step()

        if epoch % 400 == 0:
            print(f"{epoch:>6} | {d_loss.item():>8.4f} | {g_loss.item():>8.4f}")

    print("\nTraining complete!")

    # ── Verify: sample from the trained Generator ─────────────────────────────
    print("\nGenerator output samples (should look like [x, sin(x)] pairs):")
    with torch.no_grad():                          # no_grad = don't track gradients (inference only)
        noise = torch.randn(5, latent_dim)
        samples = generator(noise).numpy()
        for i, (x, y) in enumerate(samples):
            expected = np.sin(x)
            print(f"  Sample {i+1}: x={x:.3f}, generated_y={y:.3f}, sin(x)={expected:.3f}")


if __name__ == "__main__":
    train_gan()
