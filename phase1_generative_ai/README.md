# Phase 1 — Generative AI

## What you'll learn
- What a GAN is and how its two networks compete
- How to build and train a GAN in PyTorch
- How to call the Claude (Anthropic) API to generate text

## Files
| File | What it teaches |
|------|----------------|
| `simple_gan.py` | Build a GAN from scratch — Generator vs Discriminator |
| `text_generation.py` | Call Claude API, stream a response, engineer a prompt |

## Run order
```bash
# 1. Train the GAN (no API key needed)
python phase1_generative_ai/simple_gan.py

# 2. Generate text with Claude (needs ANTHROPIC_API_KEY in .env)
python phase1_generative_ai/text_generation.py
```

## Concept: What is a GAN?
A GAN (Generative Adversarial Network) has two neural networks fighting each other:

- **Generator** — tries to create fake data that looks real
- **Discriminator** — tries to tell real data from fake data

They train together. The Generator gets better at fooling the Discriminator;
the Discriminator gets better at spotting fakes. Over time the Generator learns
to create very realistic outputs.

Think of it like a forger (Generator) vs an art detective (Discriminator).
