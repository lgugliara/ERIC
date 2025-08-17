import numpy as np

def compute_alignment(x, U, latent_dim):
    return 1 - np.linalg.norm(x - U) / np.sqrt(latent_dim)

def compute_resonance(x, S, latent_dim):
    return 1 - np.linalg.norm(x - S) / np.sqrt(latent_dim)

def multi_subject_coherence(x, U, subjects, weights, latent_dim, mu=0.5, nu=0.5):
    total = 0.0
    for w, S in zip(weights, subjects):
        total += w * (mu * compute_alignment(x, U, latent_dim) +
                      nu * compute_resonance(x, S, latent_dim))
    return total

def optimize_latent(U, subjects, weights, latent_dim, steps=100):
    best_x, best_score = None, -np.inf
    for _ in range(steps):
        x = np.random.randn(latent_dim)
        score = multi_subject_coherence(x, U, subjects, weights, latent_dim)
        if score > best_score:
            best_score, best_x = score, x
    return best_x, best_score
