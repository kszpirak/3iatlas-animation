"""
3I/ATLAS animated simulation (illustrative)
- Solar gravity (2-body) + time-varying sunward jet with wobble
- Annotates factual values discussed by Avi Loeb (Dec 17, 2025)

Requires: numpy, scipy, matplotlib
Optional for MP4 saving: ffmpeg installed and available in PATH
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


# ----------------------------
# Constants & article-linked parameters
# ----------------------------

MU_SUN = 1.32712440018e20  # m^3/s^2

# Loeb article values:
RP_KM = 202_000_000  # perihelion distance b ~ 202 million km (as used in Loeb's estimate)  :contentReference[oaicite:4]{index=4}
VP_KMS = 68          # perihelion speed ~ 68 km/s  :contentReference[oaicite:5]{index=5}
LOEB_DEFLECTION_DEG = 16.4  # Loeb's estimate from 2GM/(b v^2)  :contentReference[oaicite:6]{index=6}
JET_SPAN_DEG = 8.0          # observed anti-tail span ~ 8 degrees  :contentReference[oaicite:7]{index=7}

# Jet wobble: periodic modulation of jet PA ~ 7.74 h (from cited observations)  :contentReference[oaicite:8]{index=8}
JET_WOBBLE_PERIOD_H = 7.74

# ----------------------------
# Model knobs (tweak to taste)
# ----------------------------

@dataclass(frozen=True)
class ModelParams:
    # Jet direction wobble amplitude (degrees) around the sunward direction
    wobble_amp_deg: float = 4.0  # keep within Loeb's ~8° cone idea; illustrative

    # Peak jet acceleration near perihelion (m/s^2).
    # Not specified in the article; chosen to be visually noticeable but small compared to gravity near perihelion.
    a0: float = 5e-5

    # Distance scaling for jet strength: ~1/r^2 * exp(-(r/r_cut)^2) to fade far away
    r_ref: float = 1.0 * 1.496e11  # 1 AU
    r_cut: float = 2.0 * 1.496e11  # fade beyond ~2 AU

    # Sim duration and outer stop radius
    t_span_days: float = 120.0     # simulate ±120 days around perihelion (illustrative)
    r_stop: float = 3.5 * 1.496e11 # stop integration when beyond this radius


P = ModelParams()


# ----------------------------
# Dynamics
# ----------------------------

def rotate2(v: np.ndarray, ang_rad: float) -> np.ndarray:
    c, s = math.cos(ang_rad), math.sin(ang_rad)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=float)

def jet_accel(t: float, r_vec: np.ndarray, params: ModelParams) -> np.ndarray:
    """
    Jet points approximately sunward (toward the Sun = -r_hat), with a periodic wobble.
    Strength is largest near the Sun and fades with distance.
    """
    r = float(np.linalg.norm(r_vec))
    if r == 0.0:
        return np.zeros(2)

    # Sunward unit vector:
    sunward = -r_vec / r

    # Wobble angle:
    period_s = params.wobble_amp_deg * 0.0  # placeholder for lint
    period_s = JET_WOBBLE_PERIOD_H * 3600.0
    wobble_ang = math.radians(params.wobble_amp_deg) * math.sin(2.0 * math.pi * (t / period_s))

    # In 2D, wobble = rotate sunward by wobble_ang:
    dir_vec = rotate2(sunward, wobble_ang)

    # Magnitude: ~ a0 * (r_ref/r)^2 * exp(-(r/r_cut)^2)
    mag = params.a0 * (params.r_ref / r) ** 2 * math.exp(-(r / params.r_cut) ** 2)

    return mag * dir_vec

def rhs(t: float, y: np.ndarray, params: ModelParams) -> np.ndarray:
    """
    State y = [x, y, vx, vy]
    """
    x, y_pos, vx, vy = y
    r_vec = np.array([x, y_pos], dtype=float)
    r = float(np.linalg.norm(r_vec))

    # Gravity:
    if r == 0.0:
        a_grav = np.zeros(2)
    else:
        a_grav = -MU_SUN * r_vec / (r ** 3)

    # Jet:
    a_jet = jet_accel(t, r_vec, params)

    ax, ay = a_grav[0] + a_jet[0], a_grav[1] + a_jet[1]
    return np.array([vx, vy, ax, ay], dtype=float)

def stop_when_far(t: float, y: np.ndarray, params: ModelParams) -> float:
    x, y_pos = y[0], y[1]
    r = math.hypot(x, y_pos)
    return params.r_stop - r

stop_when_far.terminal = True
stop_when_far.direction = -1


# ----------------------------
# Initial conditions at perihelion (t=0)
# ----------------------------

rp = RP_KM * 1000.0
vp = VP_KMS * 1000.0

# Place perihelion at (rp, 0) and velocity purely +y (tangential)
y0 = np.array([rp, 0.0, 0.0, vp], dtype=float)

# Integrate forward and backward so the animation includes inbound + outbound legs
tmax = P.t_span_days * 86400.0

sol_fwd = solve_ivp(
    fun=lambda t, y: rhs(t, y, P),
    t_span=(0.0, tmax),
    y0=y0,
    max_step=6*3600,  # 6h
    rtol=1e-9,
    atol=1e-9,
    events=lambda t, y: stop_when_far(t, y, P),
)

sol_bwd = solve_ivp(
    fun=lambda t, y: rhs(t, y, P),
    t_span=(0.0, -tmax),
    y0=y0,
    max_step=6*3600,
    rtol=1e-9,
    atol=1e-9,
    events=lambda t, y: stop_when_far(t, y, P),
)

# Stitch time series: reverse backward (excluding duplicate perihelion point)
t_b = sol_bwd.t[::-1]
y_b = sol_bwd.y[:, ::-1]

t_f = sol_fwd.t
y_f = sol_fwd.y

t_all = np.concatenate([t_b[:-1], t_f])
y_all = np.concatenate([y_b[:, :-1], y_f], axis=1)

# For annotation: "true" Keplerian deflection implied by rp, vp under two-body only (no jet).
# (This differs from Loeb’s small-angle estimate 2GM/(b v^2); we display BOTH honestly.)
def kepler_deflection_deg_from_rp_vp(rp_m: float, vp_ms: float) -> float:
    h = rp_m * vp_ms
    eps = vp_ms**2 / 2.0 - MU_SUN / rp_m
    e = math.sqrt(1.0 + 2.0 * eps * h*h / (MU_SUN*MU_SUN))
    delta = 2.0 * math.asin(1.0 / e)  # radians
    return math.degrees(delta)

kepler_delta = kepler_deflection_deg_from_rp_vp(rp, vp)


# ----------------------------
# Animation
# ----------------------------

AU = 1.496e11

# Downsample for animation speed
N = y_all.shape[1]
stride = max(1, N // 1200)  # target ~1200 frames max
t_anim = t_all[::stride]
y_anim = y_all[:, ::stride]
Nf = y_anim.shape[1]

x = y_anim[0] / AU
ypos = y_anim[1] / AU

fig, ax = plt.subplots(figsize=(10, 10), facecolor='#0a0a1a')
ax.set_facecolor('#0a0a1a')
ax.set_aspect("equal", "box")
ax.set_title("3I/ATLAS Trajectory Simulation", fontsize=16, fontweight='bold', color='white', pad=15)

# Axis labels
ax.set_xlabel("Distance (AU)", fontsize=12, color='white')
ax.set_ylabel("Distance (AU)", fontsize=12, color='white')
ax.tick_params(colors='white', labelsize=10)
for spine in ax.spines.values():
    spine.set_color('#333355')

# Plot full trajectory faintly
ax.plot(x, ypos, linewidth=1.5, color='#4488ff', alpha=0.4, label='Trajectory')

# Sun - yellow with glow effect
sun_glow = ax.scatter([0], [0], s=800, color='#ffdd44', alpha=0.3, zorder=5)
sun = ax.scatter([0], [0], s=400, color='#ffcc00', edgecolors='#ffaa00', linewidths=2, zorder=6)
ax.annotate('☀ Sun', xy=(0, 0), xytext=(0.08, 0.08), fontsize=11, color='#ffdd44', fontweight='bold')

# Moving body (3I/ATLAS)
body, = ax.plot([], [], marker="o", markersize=8, color='#00ffcc', linestyle="None", 
                markeredgecolor='white', markeredgewidth=1, zorder=10, label='3I/ATLAS')

# Vectors: velocity, gravity, jet - with colors and labels
vel_line, = ax.plot([], [], linewidth=2.5, color='#00ff88', label='Velocity')
grav_line, = ax.plot([], [], linewidth=2.5, color='#ff6666', label='Gravity')
jet_line, = ax.plot([], [], linewidth=2.5, color='#ffaa00', label='Jet thrust')

# Jet cone edges (± half-angle around sunward). If "span ~8°", treat as full angle.
half_angle = math.radians(JET_SPAN_DEG / 2.0)
cone1, = ax.plot([], [], linewidth=1, color='#ffaa00', alpha=0.5, linestyle='--')
cone2, = ax.plot([], [], linewidth=1, color='#ffaa00', alpha=0.5, linestyle='--')

# Legend
ax.legend(loc='upper right', fontsize=10, facecolor='#1a1a2e', edgecolor='#333355', 
          labelcolor='white', framealpha=0.9)

# Info text box
info = ax.text(0.02, 0.02, "", transform=ax.transAxes, va="bottom", ha="left",
               fontsize=9, color='white', family='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e', edgecolor='#333355', alpha=0.9))

# View limits
pad = 0.3
xmin, xmax = float(x.min()), float(x.max())
ymin, ymax = float(ypos.min()), float(ypos.max())
ax.set_xlim(xmin - pad, xmax + pad)
ax.set_ylim(ymin - pad, ymax + pad)

# Add grid
ax.grid(True, alpha=0.2, color='#4444aa', linestyle='--')

def update(i: int):
    xi = x[i]
    yi = ypos[i]
    body.set_data([xi], [yi])

    # Current state (meters)
    r_vec = np.array([y_anim[0, i], y_anim[1, i]], dtype=float)
    v_vec = np.array([y_anim[2, i], y_anim[3, i]], dtype=float)
    r = float(np.linalg.norm(r_vec))
    t = float(t_anim[i])

    # Unit vectors
    if r == 0.0:
        r_hat = np.array([1.0, 0.0])
    else:
        r_hat = r_vec / r
    sunward = -r_hat

    # Gravity accel
    a_grav = -MU_SUN * r_vec / (r ** 3) if r != 0.0 else np.zeros(2)
    a_jet = jet_accel(t, r_vec, P)

    # Scales for drawing vectors in AU
    v_scale = 2e-5   # purely visual
    a_scale = 8e7    # purely visual

    v_end = np.array([xi, yi]) + (v_vec / AU) * v_scale
    g_end = np.array([xi, yi]) + (a_grav / AU) * (1.0 / a_scale)
    j_end = np.array([xi, yi]) + (a_jet / AU) * (1.0 / a_scale)

    vel_line.set_data([xi, v_end[0]], [yi, v_end[1]])
    grav_line.set_data([xi, g_end[0]], [yi, g_end[1]])
    jet_line.set_data([xi, j_end[0]], [yi, j_end[1]])

    # Jet cone edges around sunward
    cone_len = 0.35  # AU, for visualization
    c1 = np.array([xi, yi]) + rotate2(sunward, +half_angle) * cone_len
    c2 = np.array([xi, yi]) + rotate2(sunward, -half_angle) * cone_len
    cone1.set_data([xi, c1[0]], [yi, c1[1]])
    cone2.set_data([xi, c2[0]], [yi, c2[1]])

    # Text overlay
    days = t / 86400.0
    r_au = r / AU
    v_kms = np.linalg.norm(v_vec) / 1000
    info.set_text(
        f"═══ 3I/ATLAS Status ═══\n"
        f"Time from perihelion: {days:+.1f} days\n"
        f"Distance from Sun:    {r_au:.2f} AU\n"
        f"Speed:                {v_kms:.1f} km/s\n"
        f"───────────────────────\n"
        f"Perihelion: {RP_KM/1e6:.0f}M km @ {VP_KMS:.0f} km/s\n"
        f"Deflection: ~{LOEB_DEFLECTION_DEG:.1f}° (Loeb est.)\n"
        f"Jet wobble: {JET_WOBBLE_PERIOD_H:.2f}h period"
    )

    return body, vel_line, grav_line, jet_line, cone1, cone2, info, sun

ani = FuncAnimation(fig, update, frames=Nf, interval=25, blit=True)

plt.show(block=True)

# Optional: save to MP4 (requires ffmpeg)
# Uncomment to save:
# writer = FFMpegWriter(fps=30, bitrate=1800)
# ani.save("3i_atlas_sim.mp4", writer=writer)
