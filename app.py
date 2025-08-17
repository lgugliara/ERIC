import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import cosine

# import dal package
from engine import (
    optimize_latent, multi_subject_coherence,
    compress_memory, catmull_rom_chain
)

# ========= Parametri =========
latent_dim = 4
num_subjects = 5
frame_skip = 1
U = np.random.randn(latent_dim)
subjects = [np.random.randn(latent_dim) for _ in range(num_subjects)]
subject_weights = np.ones(num_subjects) / num_subjects
subject_memory = [[] for _ in range(num_subjects)]
X_points, coherence_scores = [], []
subject_distances = [[] for _ in range(num_subjects)]
subject_awareness = [[] for _ in range(num_subjects)]
max_memory = 48
heatmap_res = 16

# ========= Setup plot =========
fig = plt.figure(figsize=(16, 9))
gs = gridspec.GridSpec(6, 9, figure=fig)
ax1 = fig.add_subplot(gs[0:3, 0:3])     # Egoic Resonance
ax2 = fig.add_subplot(gs[0:2, 3:7])     # Coherence Score Over Time
ax3 = fig.add_subplot(gs[0:2, 7:9])     # Coherence Heatmap
ax4 = fig.add_subplot(gs[4:6, 7:9])     # Subject Local View
ax5 = fig.add_subplot(gs[2:4, 7:9])     # Memory Density Map
ax6 = fig.add_subplot(gs[2:4, 3:7])     # Avg Distance from Memory
ax7 = fig.add_subplot(gs[3:6, 0:3])     # Aggregated Attention Matrix
ax8 = fig.add_subplot(gs[4:6, 3:7])     # Subjective Awareness Index

fig.suptitle("Egoic Resonance and Multi-Subject Coherence", fontsize=16)

ax1.set_xlim(-2, 2); ax1.set_ylim(-2, 2)
ax1.set_title("Egoic Resonance", fontsize=8)
ax1.set_xlabel("Latent Dim 1", fontsize=6)
ax1.set_ylabel("Latent Dim 2", fontsize=6)
ax1.set_aspect('equal')
U_dot, = ax1.plot([], [], 'ro', label="World (U)", markersize=8)
S_dots = [ax1.plot([], [], 'bo', markersize=6)[0] for _ in range(num_subjects)]
X_dot, = ax1.plot([], [], 'go', label="x_opt", markersize=8)
trace_line, = ax1.plot([], [], 'k--', alpha=0.5, label="Ego trajectory", linewidth=0.5)
ax1.legend()

ax2.set_xlim(0, 100); ax2.set_ylim(0, 1)
ax2.set_title("Coherence Score Over Time", fontsize=8)
ax2.set_xlabel("Timestep", fontsize=6)
ax2.set_ylabel("Coherence", fontsize=6)
coherence_line, = ax2.plot([], [], 'm-o', markersize=1.5, linewidth=0.5)

x_range = y_range = np.linspace(-2, 2, heatmap_res)
X_grid, Y_grid = np.meshgrid(x_range, y_range)
heatmap = ax3.imshow(np.zeros_like(X_grid), extent=(-2, 2, -2, 2),
                     origin='lower', cmap='plasma', vmin=0, vmax=1)
ax3.set_title("Coherence Heatmap", fontsize=8)
ax3.set_xlabel("Latent Dim 1", fontsize=6)
ax3.set_ylabel("Latent Dim 2", fontsize=6)
ax3.set_aspect('equal')

ax4.set_xlim(-1, 1); ax4.set_ylim(-1, 1)
ax4.set_title("Subject Local View", fontsize=8)
ax4.set_xlabel("Δ Dim 1", fontsize=6)
ax4.set_ylabel("Δ Dim 2", fontsize=6)
ax4.set_aspect('equal')
local_dots = [ax4.plot([], [], 'bo', markersize=4)[0] for _ in range(num_subjects)]

mem_heatmap = ax5.imshow(np.zeros_like(X_grid), extent=(-2, 2, -2, 2),
                         origin='lower', cmap='inferno', vmin=0, vmax=1)
ax5.set_title("Memory Density Map", fontsize=8)
ax5.set_xlabel("Latent Dim 1", fontsize=6)
ax5.set_ylabel("Latent Dim 2", fontsize=6)
ax5.set_aspect('equal')

ax6.set_xlim(0, 100); ax6.set_ylim(0, 2)
ax6.set_title("Avg Distance from Memory", fontsize=8)
ax6.set_xlabel("Timestep", fontsize=6)
ax6.set_ylabel("Distance", fontsize=6)
distance_lines = [ax6.plot([], [], linewidth=0.5, label=f"S{i}")[0] for i in range(num_subjects)]

agg_attention = ax7.imshow(np.zeros((1, 1)), cmap='Blues', vmin=0, vmax=1)
ax7.set_title("Aggregated Attention Matrix", fontsize=8)

ax8.set_xlim(0, 100); ax8.set_ylim(0, 1)
ax8.set_title("Subjective Awareness Index", fontsize=8)
ax8.set_xlabel("Timestep", fontsize=6)
ax8.set_ylabel("Awareness", fontsize=6)
awareness_lines = [ax8.plot([], [], linewidth=0.5, label=f"S{i}")[0] for i in range(num_subjects)]

t = 0

def init():
    U_dot.set_data([], [])
    for s_dot in S_dots: s_dot.set_data([], [])
    for i, l_dot in enumerate(local_dots):
        l_dot.set_data([subjects[i][0]], [subjects[i][1]])
    X_dot.set_data([], []); trace_line.set_data([], [])
    coherence_line.set_data([], [])
    heatmap.set_data(np.zeros_like(X_grid))
    mem_heatmap.set_data(np.zeros_like(X_grid))
    agg_attention.set_data(np.zeros((1, 1)))
    for d_line in distance_lines: d_line.set_data([], [])
    for a_line in awareness_lines: a_line.set_data([], [])
    return [U_dot, X_dot, trace_line, coherence_line, heatmap, mem_heatmap, agg_attention] + \
           S_dots + local_dots + distance_lines + awareness_lines

def update(_):
    global U, subjects, t
    x_opt, score = optimize_latent(U, subjects, subject_weights, latent_dim)
    X_points.append(x_opt[:2]); coherence_scores.append(score)

    for i in range(num_subjects):
        subject_memory[i].append(x_opt.copy())
        subject_memory[i] = compress_memory(subject_memory[i], max_memory)
        mem = np.array(subject_memory[i])
        dists = np.linalg.norm(mem - subjects[i], axis=1)
        subject_distances[i].append(np.mean(dists))

    U = 0.9 * U + 0.1 * x_opt
    for i in range(num_subjects):
        if subject_memory[i]:
            mem = np.mean(subject_memory[i], axis=0)
            subjects[i] = 0.8 * subjects[i] + 0.2 * mem

    # Plot 1
    U_dot.set_data([U[0]], [U[1]])
    for i, s_dot in enumerate(S_dots):
        s_dot.set_data([subjects[i][0]], [subjects[i][1]])
    X_dot.set_data([x_opt[0]], [x_opt[1]])
    trace_data = np.array(X_points); trace_line.set_data(trace_data[:, 0], trace_data[:, 1])

    # Plot 2
    ax2.set_xlim(0, t // frame_skip + 1)
    coherence_line.set_data(range(len(coherence_scores)), coherence_scores)

    # Plot 3
    grid_vals = np.zeros_like(X_grid)
    for i in range(heatmap_res):
        for j in range(heatmap_res):
            test_x = np.zeros(latent_dim)
            test_x[0], test_x[1] = X_grid[i, j], Y_grid[i, j]
            grid_vals[i, j] = multi_subject_coherence(test_x, U, subjects, subject_weights, latent_dim)
    heatmap.set_data(grid_vals)

    # Plot 5
    all_memory_points = np.concatenate(subject_memory)
    hist, _, _ = np.histogram2d(all_memory_points[:, 0], all_memory_points[:, 1],
                                bins=heatmap_res, range=[[-2, 2], [-2, 2]])
    mem_heatmap.set_data(hist.T)
    mem_heatmap.set_clim(vmin=np.min(hist), vmax=np.max(hist))

    # Plot 6
    mem_framecurrent = t // frame_skip
    for i, line in enumerate(distance_lines):
        line.set_data(range(len(subject_distances[i])), subject_distances[i])
    ax6.set_xlim(0, mem_framecurrent + 1)
    if any(len(d) for d in subject_distances):
        ax6.set_ylim(0, max((np.max(d) if len(d) else 0) for d in subject_distances) + 1)

    # Plot 7
    if len(all_memory_points) > 2:
        # X = np.array(all_memory_points)  # shape (N, D)
        
        # Normalizza ogni vettore
        norms = np.linalg.norm(all_memory_points, axis=1, keepdims=True)
        X_norm = all_memory_points / (norms + 1e-10)
    
        # Prodotto scalare vettoriale (cosine similarity)
        similarity_matrix = X_norm @ X_norm.T
        agg_attention.set_data(similarity_matrix)
        agg_attention.set_clim(0, 1)

    # Plot 8
    for i in range(num_subjects):
        mem = np.array(subject_memory[i])
        if len(mem) > 2:
            sim_matrix = np.zeros((len(mem), len(mem)))
            for j in range(len(mem)):
                for k in range(len(mem)):
                    sim_matrix[j, k] = 1 - cosine(mem[j], mem[k])
            internal_attention = np.mean(np.diag(sim_matrix))
            awareness = internal_attention - coherence_scores[-1]
            subject_awareness[i].append(awareness)
            awareness_lines[i].set_data(range(len(subject_awareness[i])), subject_awareness[i])
            ax8.set_xlim(0, len(subject_awareness[i]) + 1)
            ax8.set_ylim(0, max(subject_awareness[i]) + 1)

    t += frame_skip
    return [U_dot, X_dot, trace_line, coherence_line, heatmap, mem_heatmap, agg_attention] + \
           S_dots + local_dots + distance_lines + awareness_lines

ani = animation.FuncAnimation(
    fig, update, init_func=init, blit=True,
    interval=1, repeat=False, save_count=0, cache_frame_data=False
)
plt.tight_layout(); plt.show()
