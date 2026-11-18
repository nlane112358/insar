import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

RAD_TO_MM = 55.4658 / (4.0 * np.pi)

if len(sys.argv) > 2:
    W = int(sys.argv[1])
    H = int(sys.argv[2])
elif len(sys.argv) > 1:
    W = H = int(sys.argv[1])
else:
    default_dim = 1024
    if os.path.exists("wrapped_phase.f32"):
        n_floats = os.path.getsize("wrapped_phase.f32") // 4
        side = math.isqrt(n_floats)
        if side * side == n_floats and side > 0:
            default_dim = side
    W = H = default_dim

wrapped = np.fromfile("wrapped_phase.f32", dtype=np.float32).reshape(H, W)
coh     = np.fromfile("coherence.f32", dtype=np.float32).reshape(H, W)
unw     = np.fromfile("unwrapped_phase.f32", dtype=np.float32).reshape(H, W) * RAD_TO_MM
truth   = np.fromfile("true_phase.f32", dtype=np.float32).reshape(H, W) * RAD_TO_MM

# Align arbitrary constant offset
unw -= np.median(unw[coh > 0.6] - truth[coh > 0.6])

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
axes[0].imshow(wrapped, cmap="twilight", vmin=-np.pi, vmax=np.pi); axes[0].set_title("Wrapped Phase (rad)")
axes[1].imshow(coh, cmap="gray", vmin=0, vmax=1);                  axes[1].set_title("Coherence")
im = axes[2].imshow(unw, cmap="RdBu_r", vmin=-100, vmax=20);       axes[2].set_title("CPU Unwrapped LOS (mm)")
axes[3].imshow(unw - truth, cmap="coolwarm", vmin=-5, vmax=5);     axes[3].set_title("Residual Error (mm)")
fig.colorbar(im, ax=axes[2], fraction=0.046)
plt.tight_layout()
plt.savefig("insar_proto_check.png", dpi=150)
print("Saved insar_proto_check.png")
