"""
3I/ATLAS Interactive Simulation with Visual Anti-Tail (Improved)
- Solar gravity (2-body) + time-varying sunward jet with wobble
- Interactive claims panel
- Fast jet visualization using LineCollection
- FIX: solve_ivp event keeps .terminal/.direction (no lambda-loss)

Requires: numpy, scipy, matplotlib
"""

from __future__ import annotations

import math
import webbrowser
import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from matplotlib.patches import Arc, FancyBboxPatch
from matplotlib.collections import LineCollection

# ----------------------------
# Constants & article-linked parameters
# ----------------------------

MU_SUN = 1.32712440018e20  # m^3/s^2
AU = 1.496e11  # meters

# Loeb article values (Dec 17, 2025 Medium post)
RP_KM = 202_000_000          # perihelion distance (b parameter)
VP_KMS = 68                   # velocity at perihelion
LOEB_DEFLECTION_DEG = 16.4    # 2GM/(bv²) = 0.286 rad
JET_SPAN_DEG = 8.0            # anti-tail jet cone full width
JET_HALF_ANGLE_DEG = JET_SPAN_DEG / 2.0  # 4° half-angle
JET_WOBBLE_PERIOD_H = 7.74    # rotation period (from cited preprint, not Dec 17 post)
ECLIPTIC_ALIGNMENT_DEG = 5

# Physical jet scale (Loeb: ~1 million km)
JET_LEN_KM_REAL = 1_000_000   # ~1 million km per Dec 15 imagery
JET_LEN_AU_REAL = JET_LEN_KM_REAL * 1000 / AU  # ≈ 0.0067 AU
JET_VIS_SCALE = 60            # exaggeration factor for visibility
JET_LEN_AU_VIS = JET_LEN_AU_REAL * JET_VIS_SCALE  # ~0.4 AU displayed

# ----------------------------
# Claims panel
# ----------------------------

CLAIMS = {
    1: {
        "title": "16.4° Deflection = 2× Cone Angle",
        "short": "Deflection = 2× Jet",
        "loeb_claim": """At perihelion, the direction of motion of 3I/ATLAS is shifted by:

  2GM/(bv²) = 0.286 rad = 16.4°

Using b = 202 million km and v = 68 km/s.

This deflection angle is TWICE the jet cone opening angle (~8°).

Loeb's geometric argument:
If one edge of the jet cone overlapped the sunward direction
BEFORE perihelion, then an identical cone on the OPPOSITE POLE
can overlap the sunward direction AFTER perihelion.

The jet doesn't "steer" — the fixed geometry + orbital turn
keeps the Sun within the cone throughout the flyby.""",
        "commentary": """What the simulation shows:

• FIXED jet axis (not actively pointing at Sun)
• Pre-perihelion: +A cone contains Sun direction
• Post-perihelion: −A cone (opposite pole) contains Sun
• The 16.4° orbital turn = 2 × 8° cone angle

This is Loeb's core geometric coincidence claim.

The simulation computes the actual deflection from the
trajectory — check if it matches 16.4°.""",
        "source": "https://avi-loeb.medium.com/six-anomalies-of-3i-atlas-f9a4dbb06db7",
        "highlight": "deflection",
    },
    2: {
        "title": "Anti-Tail Jet (~8° cone, ~1M km)",
        "short": "Anti-Tail 8°",
        "loeb_claim": """From Dec 15 imagery (Loeb, Dec 17 post):

"A prominent tightly-collimated anti-tail jet...
field of view spans 1.6 × 0.7 million km...
the jet spans about 8° out to a distance of
order a million km."

Key observations:
• Jet angular width: ~8° (full cone)
• Physical extent: ~1,000,000 km
• Direction: TOWARD the Sun (anti-tail)

This is opposite to normal comet tails which
point AWAY from the Sun due to radiation pressure.""",
        "commentary": """Physical scale in simulation:

• Real jet length: ~1 million km ≈ 0.0067 AU
• Display length: exaggerated ×{:.0f} for visibility
• Cone angle: 8° (4° half-angle)

Normal comet "anti-tails" are optical illusions
from viewing geometry. Loeb claims this is a
PHYSICAL sunward jet — material actually moving
toward the Sun against radiation pressure.

The simulation shows the jet visual; whether it
represents propulsion or unusual outgassing is
an open question.""".format(JET_VIS_SCALE),
        "source": "https://avi-loeb.medium.com/six-anomalies-of-3i-atlas-f9a4dbb06db7",
        "highlight": "jet_cone",
    },
    3: {
        "title": "Jet Wobble (~7.74h period)",
        "short": "Wobble: ~7.74h",
        "loeb_claim": """From Loeb's Dec 17 post:

"The observed wobble of the pre-perihelion sunward jet
requires the base of the jet to be within 8° from the
sun-facing pole."

The wobble parameters (7.74h period, ±4° amplitude)
are from cited observational preprints, not the Dec 17
post itself.

Interpretation:
• Nucleus spinning with ~7.74h rotation period
• Jet source near (not exactly at) rotation pole
• Wobble traces out a cone as body rotates""",
        "commentary": """Simulation parameters (from cited preprints):
• Period: {:.2f} hours
• Amplitude: ±{:.1f}°

This is actually conventional cometary physics:
• Many asteroids rotate in 2-20 hour range
• Off-axis jets common (e.g., 67P)
• Wobble implies jet is surface-fixed, not steered

Loeb's constraint: jet base within 8° of sun-facing
pole for the wobble to keep Sun in cone.

This may be the LEAST anomalous claim — standard
comet behavior. A designed probe would likely have
more stable pointing.""".format(JET_WOBBLE_PERIOD_H, P.wobble_amp_deg if 'P' in dir() else 4.0),
        "source": "https://avi-loeb.medium.com/six-anomalies-of-3i-atlas-f9a4dbb06db7",
        "highlight": "wobble",
    },
    4: {
        "title": "Sunward Jet Before & After Perihelion",
        "short": "Pre+Post Sunward",
        "loeb_claim": """From Loeb's Dec 17 post:

A prominent tightly-collimated sunward anti-tail appears
BOTH BEFORE and AFTER perihelion.

CRITICAL: The jet does NOT actively "track" the Sun!

Loeb's geometric argument:
• Fixed jet axis on rotating body
• Pre-perihelion: Sun within +A cone edge
• Post-perihelion: Sun within −A cone edge (opposite pole)
• The 16.4° orbital deflection = 2 × 8° cone angle
  makes this geometry possible

The sunward appearance is maintained by the COINCIDENCE
of deflection = 2× cone angle, not active steering.""",
        "commentary": """What the simulation demonstrates:

• Jet axis is FIXED in inertial space
• At perihelion, active cone switches from +A to −A
• Sun direction shown: green if inside cone, red if outside

Loeb's mechanism:
The physical sunward jet stays tightly collimated to
~million km scales throughout the flyby.

This is the CORE geometric coincidence:
If deflection ≠ 2× cone angle, Sun would exit the cone
during the flyby.

Watch the "Sun in cone" indicator as 3I/ATLAS passes
perihelion — does the geometry actually work?""",
        "source": "https://avi-loeb.medium.com/six-anomalies-of-3i-atlas-f9a4dbb06db7",
        "highlight": "sunward",
    },
    5: {
        "title": "Perihelion Parameters",
        "short": "Closest Approach",
        "loeb_claim": f"""Orbital parameters at closest approach:

• Perihelion distance: {RP_KM/1e6:.0f} million km (~1.35 AU)
• Velocity at perihelion: {VP_KMS} km/s
• Hyperbolic excess velocity: ~45 km/s

These values precisely satisfy:
  2GM/(b × v²) = 16.4°

Loeb suggests these "tuned" parameters allow the jet
geometry to work — not too fast, not too slow, not too
close, not too far.""",
        "commentary": """The orbital mechanics are well-constrained by observations.

What's actually measured:
• Position over time → orbital elements
• Velocity from Doppler shifts
• These calculations are robust

The "fine-tuning" argument:
• Loeb notes parameters seem "convenient"
• But ANY orbit has specific parameters
• We're pattern-matching after the fact

Counter-argument:
• Billions of interstellar objects pass through
• We only notice ones with interesting properties
• Selection bias explains "special" parameters

The perihelion values themselves are unremarkable —
many comets have similar approaches. What matters is
the COMBINATION with other claimed anomalies.""",
        "source": "https://avi-loeb.medium.com/six-anomalies-of-3i-atlas-f9a4dbb06db7",
        "highlight": "perihelion",
    },
    6: {
        "title": "Ecliptic Alignment (±5°)",
        "short": "Ecliptic ±5°",
        "loeb_claim": """From Loeb's Dec 17 post:

"Retrograde trajectory... aligned to within 5° with the
ecliptic plane... probability 0.2%."

The ecliptic is the plane containing Earth's orbit and
(approximately) all major planets.

Random probability calculation:
• Sky is a sphere, ecliptic is a narrow band
• Chance of random object being within ±5°: ~0.2%

Loeb suggests this alignment might indicate targeting
at our solar system's planetary plane.""",
        "commentary": """⚠️ NOT MODELED in this 2D simulation.

This simulation runs in the orbital plane (2D).
To properly visualize ecliptic alignment, would need:
• 3D state vector (x,y,z,vx,vy,vz)
• Defined ecliptic plane (z=0)
• Initial inclination of ~5°

Caveats on the claim:

Observational bias:
• Pan-STARRS, ATLAS focus near ecliptic
• Objects there MORE LIKELY to be discovered

Small number statistics (n=3):
• 1I 'Oumuamua: ~30° from ecliptic
• 2I Borisov: ~44° from ecliptic
• 3I is the OUTLIER, not the norm

Verdict: May be selection effect.""",
        "source": "https://avi-loeb.medium.com/six-anomalies-of-3i-atlas-f9a4dbb06db7",
        "highlight": "ecliptic",
    },
}

# ----------------------------
# Model parameters
# ----------------------------

@dataclass(frozen=True)
class ModelParams:
    wobble_amp_deg: float = 4.0
    a0: float = 5e-5        # m/s^2 (peak)
    r_ref: float = 1.0 * AU
    r_cut: float = 2.0 * AU
    t_span_days: float = 120.0
    r_stop: float = 3.5 * AU
    max_step_s: float = 6 * 3600

P = ModelParams()

# ----------------------------
# Dynamics
# ----------------------------

def rotate2(v: np.ndarray, ang_rad: float) -> np.ndarray:
    c, s = math.cos(ang_rad), math.sin(ang_rad)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=float)

def jet_accel(t: float, r_vec: np.ndarray, params: ModelParams) -> np.ndarray:
    r = float(np.linalg.norm(r_vec))
    if r == 0.0:
        return np.zeros(2)

    sunward = -r_vec / r
    period_s = JET_WOBBLE_PERIOD_H * 3600.0
    wobble_ang = math.radians(params.wobble_amp_deg) * math.sin(2.0 * math.pi * (t / period_s))
    dir_vec = rotate2(sunward, wobble_ang)

    # Your chosen magnitude law
    mag = params.a0 * (params.r_ref / r) ** 2 * math.exp(-(r / params.r_cut) ** 2)
    return mag * dir_vec

def rhs(t: float, state: np.ndarray, params: ModelParams) -> np.ndarray:
    x, y, vx, vy = state
    r_vec = np.array([x, y], dtype=float)
    r = float(np.linalg.norm(r_vec))

    if r == 0.0:
        a_grav = np.zeros(2)
    else:
        a_grav = -MU_SUN * r_vec / (r ** 3)

    a_jet = jet_accel(t, r_vec, params)
    ax, ay = a_grav[0] + a_jet[0], a_grav[1] + a_jet[1]
    return np.array([vx, vy, ax, ay], dtype=float)

def make_stop_event(params: ModelParams):
    # IMPORTANT: no lambda here, or you lose .terminal / .direction
    def stop_when_far(t: float, y: np.ndarray) -> float:
        return params.r_stop - math.hypot(y[0], y[1])

    stop_when_far.terminal = True
    stop_when_far.direction = -1
    return stop_when_far

# ----------------------------
# Compute trajectory
# ----------------------------

print("Computing trajectory...")

rp = RP_KM * 1000.0
vp = VP_KMS * 1000.0
y0 = np.array([rp, 0.0, 0.0, vp], dtype=float)
tmax = P.t_span_days * 86400.0

event = make_stop_event(P)

sol_fwd = solve_ivp(
    fun=lambda t, y: rhs(t, y, P),
    t_span=(0.0, tmax),
    y0=y0,
    max_step=P.max_step_s,
    rtol=1e-9,
    atol=1e-9,
    events=event,
)

sol_bwd = solve_ivp(
    fun=lambda t, y: rhs(t, y, P),
    t_span=(0.0, -tmax),
    y0=y0,
    max_step=P.max_step_s,
    rtol=1e-9,
    atol=1e-9,
    events=event,
)

t_b, y_b = sol_bwd.t[::-1], sol_bwd.y[:, ::-1]
t_f, y_f = sol_fwd.t, sol_fwd.y
t_all = np.concatenate([t_b[:-1], t_f])
y_all = np.concatenate([y_b[:, :-1], y_f], axis=1)

# Downsample (keep at most ~1400 frames)
N = y_all.shape[1]
stride = max(1, N // 1400)
t_anim = t_all[::stride]
y_anim = y_all[:, ::stride]
Nf = y_anim.shape[1]

x_traj = y_anim[0] / AU
y_traj = y_anim[1] / AU

distances = np.hypot(x_traj, y_traj)
perihelion_idx = int(np.argmin(distances))

print(f"Trajectory computed: {Nf} frames, perihelion frame {perihelion_idx}")

# ----------------------------
# Precompute animation vectors (FAST)
# ----------------------------

# Position in AU
pos_au = np.column_stack([x_traj, y_traj])  # (Nf,2)

# Velocity (unit direction for drawing)
vel = np.column_stack([y_anim[2], y_anim[3]])  # m/s
vel_norm = np.linalg.norm(vel, axis=1)
vel_hat = np.divide(vel, vel_norm[:, None], out=np.zeros_like(vel), where=vel_norm[:, None] != 0)

# Sunward unit direction
r_m = np.column_stack([y_anim[0], y_anim[1]])
r_norm = np.linalg.norm(r_m, axis=1)
sunward_hat = np.divide(-r_m, r_norm[:, None], out=np.zeros_like(r_m), where=r_norm[:, None] != 0)

# ----------------------------
# LOEB'S FIXED-AXIS JET GEOMETRY
# ----------------------------
# The jet axis is FIXED in inertial space, NOT actively pointing at Sun.
# Pre-perihelion: cone centered on +A_hat
# Post-perihelion: cone centered on -A_hat (opposite pole)
# The 16.4° deflection = 2 × 8° cone allows Sun to stay in cone edges.

# Compute actual deflection angle from trajectory
v_in_hat = vel_hat[0]   # inbound velocity direction (far pre-perihelion)
v_out_hat = vel_hat[-1] # outbound velocity direction (far post-perihelion)
dot_product = np.clip(np.dot(v_in_hat, v_out_hat), -1.0, 1.0)
computed_deflection_rad = np.arccos(dot_product)
computed_deflection_deg = np.degrees(computed_deflection_rad)
print(f"Computed deflection angle: {computed_deflection_deg:.2f}° (Loeb claims: {LOEB_DEFLECTION_DEG}°)")

# Fixed jet axis: set to sunward direction at far pre-perihelion
# This is the key: the axis does NOT change during the flyby
jet_axis_pre = sunward_hat[0].copy()  # +A_hat: fixed inertial direction
jet_axis_post = -jet_axis_pre         # -A_hat: opposite pole

# Wobble angle (rad) - rotation around jet axis
period_s = JET_WOBBLE_PERIOD_H * 3600.0
wobble_phase = 2.0 * np.pi * (t_anim / period_s)
wobble_deg = P.wobble_amp_deg * np.sin(wobble_phase)
wobble_rad = np.deg2rad(wobble_deg)

# For each frame, compute:
# 1. Which cone is active (+A pre-perihelion, -A post-perihelion)
# 2. Whether Sun direction falls inside the active cone
# 3. Jet direction with wobble applied to the fixed axis

jet_half_angle_rad = np.deg2rad(JET_HALF_ANGLE_DEG)

# Precompute jet directions and sun-in-cone status
jet_hat = np.zeros((Nf, 2))
sun_in_cone = np.zeros(Nf, dtype=bool)
active_axis = np.zeros((Nf, 2))

for i in range(Nf):
    # Select active cone based on phase (perihelion at t=0)
    if i < perihelion_idx:
        axis = jet_axis_pre
    else:
        axis = jet_axis_post
    active_axis[i] = axis
    
    # Apply wobble rotation to axis
    c, s = np.cos(wobble_rad[i]), np.sin(wobble_rad[i])
    jet_hat[i, 0] = c * axis[0] - s * axis[1]
    jet_hat[i, 1] = s * axis[0] + c * axis[1]
    
    # Check if Sun direction falls inside the cone
    angle_to_sun = np.arccos(np.clip(np.dot(axis, sunward_hat[i]), -1.0, 1.0))
    sun_in_cone[i] = angle_to_sun <= jet_half_angle_rad

print(f"Sun in cone: {np.sum(sun_in_cone)}/{Nf} frames ({100*np.sum(sun_in_cone)/Nf:.1f}%)")

# ----------------------------
# Interactive animation
# ----------------------------

class InteractiveSimulation:
    def __init__(self):
        self.current_claim = None
        self.paused = False
        self.details_fig = None  # Store reference to prevent garbage collection

        self.fig = plt.figure(figsize=(20, 11), facecolor="#0a0a1a")
        self.ax = self.fig.add_axes([0.02, 0.06, 0.62, 0.90])  # narrower to leave room for legend
        self.ax.set_facecolor("#0a0a1a")
        self.ax.set_aspect("equal", "box")

        self.annotations = []
        self.extra_artists = []

        self._setup_main_plot()
        self._setup_claims_panel()
        self._bind_keys()

        # Use blit for speed (requires returning artists from update)
        self.ani = FuncAnimation(self.fig, self._update, frames=Nf, interval=30, blit=True, repeat=True)

    def _setup_main_plot(self):
        ax = self.ax
        ax.set_title("3I/ATLAS: Loeb's Fixed-Axis Jet Geometry (Dec 17, 2025)", fontsize=16, fontweight="bold", color="white", pad=15)
        ax.set_xlabel("Distance from Sun (AU)", fontsize=12, color="white")
        ax.set_ylabel("Distance from Sun (AU)", fontsize=12, color="white")
        ax.tick_params(colors="white", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#333355")

        # full trajectory
        ax.plot(x_traj, y_traj, linewidth=2, color="#4488ff", alpha=0.3)

        # Sun
        ax.scatter([0], [0], s=3000, color="#ffdd44", alpha=0.1, zorder=5)
        ax.scatter([0], [0], s=1500, color="#ffdd44", alpha=0.2, zorder=5)
        ax.scatter([0], [0], s=700, color="#ffcc00", edgecolors="#ff8800", linewidths=3, zorder=6)
        ax.text(0, 0, "☀", fontsize=24, ha="center", va="center", color="#ffff88", zorder=7)
        ax.text(
            0, -0.32, "SUN", fontsize=14, ha="center", va="top", color="#ffdd44", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a0a", edgecolor="#ffaa00", alpha=0.9),
        )

        # moving body (scatter = faster than plot marker updates)
        self.body = ax.scatter([], [], s=110, color="#00ffcc", edgecolors="white", linewidths=2, zorder=20)
        self.body_label = ax.text(0, 0, "", fontsize=9, color="#00ffcc", fontweight="bold", ha="left", va="bottom", zorder=21)

        # trail
        self.trail, = ax.plot([], [], linewidth=2, color="#66aaff", alpha=0.9, zorder=12)
        self.trail_len = 120

        # Jet “particle rays” using LineCollection (single artist)
        self.n_rays = 15
        self.jet_len_au = JET_LEN_AU_VIS  # Physically-based (exaggerated) length
        self.half_angle = math.radians(JET_SPAN_DEG / 2.0)
        segs = np.zeros((self.n_rays, 2, 2), dtype=float)
        self.jet_lc = LineCollection(segs, linewidths=np.linspace(3.0, 1.0, self.n_rays),
                                     colors=[(1.0, 0.67, 0.0, 0.8)], zorder=15)
        ax.add_collection(self.jet_lc)

        self.jet_label = ax.text(0, 0, "", fontsize=10, color="#ffaa00", fontweight="bold", ha="center", va="bottom", zorder=22)
        
        # Anti-tail label (always visible)
        self.antitail_label = ax.text(0, 0, "Anti-Tail", fontsize=9, color="#ffaa00", fontweight="bold", 
                                       ha="left", va="center", zorder=22)
        
        # Fixed jet axis indicator (shows the inertial axis, not the wobbling jet)
        self.axis_line, = ax.plot([], [], linewidth=2, color="#ff00ff", linestyle=":", alpha=0.6, zorder=13)
        self.axis_label = ax.text(0, 0, "", fontsize=8, color="#ff00ff", ha="center", va="bottom", zorder=22)
        
        # Sun-in-cone indicator (top-right corner)
        self.sun_cone_indicator = ax.text(0.98, 0.98, "", transform=ax.transAxes, fontsize=11, 
                                          fontweight="bold", ha="right", va="top", zorder=30,
                                          bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", 
                                                   edgecolor="#444466", alpha=0.95))

        # velocity vector
        self.vel_line, = ax.plot([], [], linewidth=3, color="#00ff88", zorder=18)
        # gravity vector
        self.grav_line, = ax.plot([], [], linewidth=2, color="#ff6666", linestyle="--", alpha=0.7, zorder=17)
        
        # "Normal tail" arrow - shows what a NORMAL comet tail would do (point AWAY from sun)
        from matplotlib.patches import FancyArrowPatch
        self.normal_tail_arrow = FancyArrowPatch((0, 0), (0, 0), 
                                                  arrowstyle='->', mutation_scale=15,
                                                  color='#6688aa', linewidth=3, 
                                                  linestyle='--', alpha=0.9, zorder=14)
        ax.add_patch(self.normal_tail_arrow)
        self.normal_tail_label = ax.text(0, 0, "", fontsize=8, color="#6688aa", ha="center", va="top", 
                                          fontweight="bold", zorder=22)

        # info box
        self.info = ax.text(
            0.02, 0.02, "", transform=ax.transAxes, va="bottom", ha="left",
            fontsize=10, color="white", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e", edgecolor="#333355", alpha=0.95),
            zorder=25,
        )
        
        # Jet scale indicator (bottom-right)
        ax.text(0.98, 0.02, 
                f"Jet: ~{JET_LEN_KM_REAL/1e6:.0f}M km (×{JET_VIS_SCALE:.0f} exaggerated)\n"
                f"Deflection: {computed_deflection_deg:.1f}° computed",
                transform=ax.transAxes, va="bottom", ha="right",
                fontsize=8, color="#888899", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", edgecolor="#333355", alpha=0.85),
                zorder=25)

        # ===== "Normal Comet" comparison in corner =====
        # Draw a small schematic showing how normal comets behave
        self._draw_normal_comet_comparison(ax)

        # ===== Legend - positioned outside plot on the right =====
        legend_items = [
            plt.Line2D([0], [0], color='#00ffcc', marker='o', markersize=10, linestyle='None', label='3I/ATLAS'),
            plt.Line2D([0], [0], color='#66aaff', linewidth=2, label='Recent path (trail)'),
            plt.Line2D([0], [0], color='#ffaa00', linewidth=4, label=f'ANTI-TAIL (8° cone)'),
            plt.Line2D([0], [0], color='#ff00ff', linewidth=2, linestyle=':', label='Fixed jet axis (+A/-A)'),
            plt.Line2D([0], [0], color='#6688aa', linewidth=3, linestyle=':', label='Normal tail (away)'),
            plt.Line2D([0], [0], color='#00ff88', linewidth=3, label='Velocity'),
            plt.Line2D([0], [0], color='#ff6666', linewidth=2, linestyle='--', label='Gravity -> Sun'),
        ]
        ax.legend(handles=legend_items, loc='upper left', bbox_to_anchor=(1.01, 1.0),
                  fontsize=9, facecolor='#1a1a2e', edgecolor='#333355', 
                  labelcolor='white', framealpha=0.95)

        # view limits - 6 AU wide centered on trajectory
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.grid(True, alpha=0.15, color="#4444aa", linestyle="--")

    def _draw_normal_comet_comparison(self, ax):
        """Draw a small inset showing normal comet behavior vs 3I/ATLAS"""
        # Position in upper-left of plot (in data coordinates)
        cx, cy = -2.3, 2.3  # center of comparison diagram
        scale = 0.35
        
        # Background box
        from matplotlib.patches import Rectangle
        bg = Rectangle((cx - 0.55, cy - 0.7), 1.1, 1.0, 
                       facecolor='#0a0a1a', edgecolor='#444466', 
                       linewidth=2, alpha=0.95, zorder=30)
        ax.add_patch(bg)
        
        # Title
        ax.text(cx, cy + 0.22, "NORMAL vs 3I/ATLAS", fontsize=8, fontweight='bold',
               color='#aaaacc', ha='center', va='bottom', zorder=31)
        
        # --- Normal comet (left side) ---
        nc_x = cx - 0.25
        nc_y = cy - 0.15
        # Comet body
        ax.scatter([nc_x], [nc_y], s=40, color='#888888', zorder=31)
        # Normal tail (points AWAY from sun, which is at origin)
        # Direction away from sun
        tail_dir = np.array([nc_x, nc_y])
        tail_dir = tail_dir / np.linalg.norm(tail_dir) * scale
        ax.arrow(nc_x, nc_y, tail_dir[0], tail_dir[1], 
                head_width=0.05, head_length=0.03, fc='#6688aa', ec='#6688aa', 
                linewidth=2, zorder=31)
        ax.text(nc_x, nc_y - 0.35, "Normal", fontsize=7, color='#6688aa', 
               ha='center', va='top', zorder=31)
        ax.text(nc_x, nc_y - 0.45, "Tail AWAY", fontsize=6, color='#6688aa', 
               ha='center', va='top', zorder=31)
        
        # --- 3I/ATLAS (right side) ---
        at_x = cx + 0.25
        at_y = cy - 0.15
        # Comet body
        ax.scatter([at_x], [at_y], s=40, color='#00ffcc', zorder=31)
        # Anti-tail (points TOWARD sun at origin)
        tail_dir = -np.array([at_x, at_y])
        tail_dir = tail_dir / np.linalg.norm(tail_dir) * scale
        ax.arrow(at_x, at_y, tail_dir[0], tail_dir[1], 
                head_width=0.05, head_length=0.03, fc='#ffaa00', ec='#ffaa00', 
                linewidth=2, zorder=31)
        ax.text(at_x, at_y - 0.35, "3I/ATLAS", fontsize=7, color='#ffaa00', 
               ha='center', va='top', zorder=31)
        ax.text(at_x, at_y - 0.45, "Jet TO Sun!", fontsize=6, color='#ffaa00', 
               ha='center', va='top', zorder=31)
        
        # Small sun indicator in the comparison box
        ax.scatter([cx], [cy + 0.05], s=30, color='#ffcc00', marker='*', zorder=31)
        ax.text(cx, cy + 0.12, "Sun", fontsize=6, color='#ffcc00', ha='center', va='bottom', zorder=31)

    def _setup_claims_panel(self):
        self.fig.text(0.88, 0.93, "LOEB'S ANOMALIES", fontsize=14, fontweight="bold", color="#ffcc00", ha="center")
        self.fig.text(0.88, 0.905, "Click to visualize", fontsize=9, color="#888899", ha="center")

        self.buttons = []
        button_h = 0.060
        start_y = 0.84

        # IMPORTANT: avoid late-binding lambda issues by using a closure helper
        def make_cb(cid):
            return lambda event: self.select_claim(cid)

        for i, (cid, claim) in enumerate(CLAIMS.items()):
            y_pos = start_y - i * (button_h + 0.010)
            btn_ax = self.fig.add_axes([0.76, y_pos, 0.22, button_h])
            btn = Button(btn_ax, f"{cid}. {claim['short']}", color="#1a1a3e", hovercolor="#3a3a6e")
            btn.label.set_color("white")
            btn.label.set_fontsize(9)
            btn.label.set_fontweight("bold")
            btn.on_clicked(make_cb(cid))
            self.buttons.append((cid, btn, btn_ax))

        reset_ax = self.fig.add_axes([0.76, 0.15, 0.10, 0.042])
        self.reset_btn = Button(reset_ax, "RESET", color="#333355", hovercolor="#555588")
        self.reset_btn.label.set_color("white")
        self.reset_btn.label.set_fontweight("bold")
        self.reset_btn.on_clicked(self.reset_view)

        # Pause/Play button
        pause_ax = self.fig.add_axes([0.87, 0.15, 0.11, 0.042])
        self.pause_btn = Button(pause_ax, "PAUSE", color="#4a2a2a", hovercolor="#6a4a4a")
        self.pause_btn.label.set_color("white")
        self.pause_btn.label.set_fontweight("bold")
        self.pause_btn.on_clicked(self._toggle_pause)

        # "View Details" button to open scrollable window
        details_ax = self.fig.add_axes([0.76, 0.10, 0.22, 0.042])
        self.details_btn = Button(details_ax, "[+] VIEW DETAILS", color="#2a4a2a", hovercolor="#3a6a3a")
        self.details_btn.label.set_color("white")
        self.details_btn.label.set_fontweight("bold")
        self.details_btn.on_clicked(self._open_details_window)

        # Brief claim title shown in main window
        self.desc_title = self.fig.text(0.76, 0.43, "Select a claim", fontsize=10, color="#ffcc00",
                                        ha="left", va="top", fontweight="bold")
        self.desc_text = self.fig.text(0.76, 0.40, "Click button for full analysis", fontsize=8, color="#888899",
                                       ha="left", va="top")

    def _open_details_window(self, event=None):
        """Open a matplotlib figure window with claim details"""
        # Close previous details window if open
        if self.details_fig is not None:
            try:
                # Check if figure still exists before closing
                if plt.fignum_exists(self.details_fig.number):
                    plt.close(self.details_fig)
            except Exception:
                pass
            self.details_fig = None
        
        # Show details for current claim, or all claims if none selected
        if self.current_claim:
            self._show_single_claim_details(self.current_claim)
        else:
            self._show_all_claims_details()

    def _on_details_close(self, evt):
        """Handle details window close event"""
        self.details_fig = None

    def _show_single_claim_details(self, claim_id):
        """Show detailed view of a single claim"""
        claim = CLAIMS[claim_id]
        
        fig = plt.figure(figsize=(18, 10), facecolor="#0a0a1a")
        self.details_fig = fig  # Store reference
        
        # Handle window close
        fig.canvas.mpl_connect('close_event', self._on_details_close)
        
        fig.suptitle(f"Claim #{claim_id}: {claim['title']}", fontsize=18, fontweight="bold", 
                     color="#ffcc00", y=0.97)
        
        # Two panels side by side with gap between them
        # Left panel: Loeb's Claim
        left_ax = fig.add_axes([0.02, 0.06, 0.46, 0.85])
        left_ax.set_facecolor("#0d1a0d")
        for spine in left_ax.spines.values():
            spine.set_color("#88ff88")
            spine.set_linewidth(2)
        left_ax.set_xlim(0, 1)
        left_ax.set_ylim(0, 1)
        left_ax.set_xticks([])
        left_ax.set_yticks([])
        
        # Left panel content - left aligned
        left_ax.text(0.5, 0.98, "LOEB'S CLAIM", fontsize=14, fontweight="bold",
                    color="#88ff88", ha="center", va="top")
        left_ax.text(0.03, 0.92, claim["loeb_claim"], fontsize=10, color="#ccddcc",
                    ha="left", va="top", family="monospace", linespacing=1.4)
        
        # Right panel: Analysis
        right_ax = fig.add_axes([0.52, 0.06, 0.46, 0.85])
        right_ax.set_facecolor("#1a0d1a")
        for spine in right_ax.spines.values():
            spine.set_color("#ff88ff")
            spine.set_linewidth(2)
        right_ax.set_xlim(0, 1)
        right_ax.set_ylim(0, 1)
        right_ax.set_xticks([])
        right_ax.set_yticks([])
        
        # Right panel content - left aligned
        right_ax.text(0.5, 0.98, "ANALYSIS", fontsize=14, fontweight="bold",
                     color="#ff88ff", ha="center", va="top")
        right_ax.text(0.03, 0.92, claim["commentary"], fontsize=10, color="#ddccdd",
                     ha="left", va="top", family="monospace", linespacing=1.4)
        
        # Source link at bottom
        fig.text(0.5, 0.02, f"Source: {claim['source']}", fontsize=9, color="#66aaff",
                ha="center", style="italic")
        
        plt.show(block=False)

    def _show_all_claims_details(self):
        """Show overview of all claims in a scrollable-like grid"""
        fig = plt.figure(figsize=(16, 12), facecolor="#0a0a1a")
        self.details_fig = fig  # Store reference
        
        # Handle window close
        fig.canvas.mpl_connect('close_event', self._on_details_close)
        
        fig.suptitle("LOEB'S SIX ANOMALIES OF 3I/ATLAS", fontsize=18, fontweight="bold",
                     color="#ffcc00", y=0.97)
        fig.text(0.5, 0.94, "Select a claim in main window, then click VIEW DETAILS for full analysis",
                fontsize=10, color="#888899", ha="center")
        
        # Create 3x2 grid of claim summaries
        for i, (cid, claim) in enumerate(CLAIMS.items()):
            row = i // 2
            col = i % 2
            
            x = 0.03 + col * 0.49
            y = 0.62 - row * 0.30
            
            ax = fig.add_axes([x, y, 0.46, 0.26])
            ax.set_facecolor("#1a1a2e")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#444466")
                spine.set_linewidth(2)
            
            # Title
            ax.text(0.02, 0.92, f"#{cid}", fontsize=14, fontweight="bold", color="#00ffcc", va="top")
            ax.text(0.12, 0.92, claim["title"], fontsize=12, fontweight="bold", color="#ffcc00", va="top")
            
            # Brief content - truncate if needed
            loeb_short = claim["loeb_claim"][:80] + "..." if len(claim["loeb_claim"]) > 80 else claim["loeb_claim"]
            analysis_short = claim["commentary"][:80] + "..." if len(claim["commentary"]) > 80 else claim["commentary"]
            
            ax.text(0.02, 0.70, "CLAIM:", fontsize=9, fontweight="bold", color="#88ff88", va="top")
            ax.text(0.02, 0.55, loeb_short, fontsize=9, color="#aaaacc", va="top", wrap=True)
            
            ax.text(0.02, 0.35, "ANALYSIS:", fontsize=9, fontweight="bold", color="#ff88ff", va="top")
            ax.text(0.02, 0.20, analysis_short, fontsize=9, color="#aaaacc", va="top", wrap=True)
        
        plt.show(block=False)

    def _bind_keys(self):
        def on_key(event):
            if event.key == " ":
                self._toggle_pause()
            elif event.key in ("right", "d"):
                # step forward one frame when paused
                if self.paused:
                    self._manual_step(+1)
            elif event.key in ("left", "a"):
                if self.paused:
                    self._manual_step(-1)
            elif event.key and event.key.isdigit():
                cid = int(event.key)
                if cid in CLAIMS:
                    self.select_claim(cid)
        self.fig.canvas.mpl_connect("key_press_event", on_key)

    def _toggle_pause(self, event=None):
        """Toggle pause/play state"""
        self.paused = not self.paused
        if self.paused:
            self.pause_btn.label.set_text("PLAY")
            self.pause_btn.ax.set_facecolor("#2a4a2a")
            self.pause_btn.hovercolor = "#4a6a4a"
        else:
            self.pause_btn.label.set_text("PAUSE")
            self.pause_btn.ax.set_facecolor("#4a2a2a")
            self.pause_btn.hovercolor = "#6a4a4a"
        self.fig.canvas.draw_idle()

    def _manual_step(self, delta: int):
        # move current frame index by delta; FuncAnimation will call update again
        # easiest: set an attribute and use it in update
        if not hasattr(self, "_frame_override"):
            self._frame_override = 0
        self._frame_override = (getattr(self, "_last_frame", 0) + delta) % Nf
        self.fig.canvas.draw_idle()

    def select_claim(self, claim_id: int):
        self.current_claim = claim_id
        claim = CLAIMS[claim_id]

        for cid, btn, btn_ax in self.buttons:
            btn_ax.set_facecolor("#4a4a8e" if cid == claim_id else "#1a1a3e")

        self.desc_title.set_text(f"#{claim_id}: {claim['title']}")
        self.desc_text.set_text("Click 'VIEW DETAILS' for full analysis")

        self._clear_annotations()

        h = claim["highlight"]
        if h == "deflection":
            self._show_deflection()
        elif h == "jet_cone":
            self._show_jet_cone()
        elif h == "wobble":
            self._show_wobble()
        elif h == "sunward":
            self._show_sunward()
        elif h == "perihelion":
            self._show_perihelion()
        elif h == "ecliptic":
            self._show_ecliptic()

        self.fig.canvas.draw_idle()

    def reset_view(self, event=None):
        self.current_claim = None
        self._clear_annotations()
        self.desc_title.set_text("Select a claim")
        self.desc_text.set_text("")
        for _, _, btn_ax in self.buttons:
            btn_ax.set_facecolor("#1a1a3e")
        self.fig.canvas.draw_idle()

    def _clear_annotations(self):
        for a in self.annotations:
            try:
                a.remove()
            except Exception:
                pass
        self.annotations = []

        for a in self.extra_artists:
            try:
                a.remove()
            except Exception:
                pass
        self.extra_artists = []

    # --- claim visuals ---
    def _show_deflection(self):
        # Compute the actual deflection from trajectory asymptotes
        # v_in_hat and v_out_hat are the inbound/outbound velocity directions
        theta1_deg = np.degrees(np.arctan2(v_in_hat[1], v_in_hat[0]))
        theta2_deg = np.degrees(np.arctan2(v_out_hat[1], v_out_hat[0]))
        
        # Draw arc showing the actual deflection
        arc = Arc((0, 0), 1.5, 1.5, angle=0, 
                  theta1=min(theta1_deg, theta2_deg), 
                  theta2=max(theta1_deg, theta2_deg),
                  color="#ffff00", lw=4, zorder=25)
        self.ax.add_patch(arc)
        self.extra_artists.append(arc)
        
        # Draw the inbound and outbound asymptote lines
        line_in, = self.ax.plot([0, -v_in_hat[0]*2.5], [0, -v_in_hat[1]*2.5], 
                                '--', color='#88ff88', lw=2, alpha=0.7, zorder=24)
        line_out, = self.ax.plot([0, v_out_hat[0]*2.5], [0, v_out_hat[1]*2.5], 
                                 '--', color='#ff8888', lw=2, alpha=0.7, zorder=24)
        self.extra_artists.extend([line_in, line_out])

        t = self.ax.text(
            0.02, 0.97, 
            f"Computed deflection: {computed_deflection_deg:.1f}°\n"
            f"Loeb's value: {LOEB_DEFLECTION_DEG}° = 2 × {JET_SPAN_DEG}° cone",
            transform=self.ax.transAxes, fontsize=11, color="#ffff88", ha="left", va="top", fontweight="bold",
            bbox=dict(facecolor="#2a2a1a", edgecolor="#ffff00", alpha=0.95, pad=6),
        )
        self.annotations.append(t)

    def _show_jet_cone(self):
        t = self.ax.text(
            0.02, 0.97, "ANTI-TAIL: Jet points TOWARD Sun (opposite of normal comets!)",
            transform=self.ax.transAxes, fontsize=11, color="#ffaa00", ha="left", va="top", fontweight="bold",
            bbox=dict(facecolor="#2a2a1a", edgecolor="#ffaa00", alpha=0.95, pad=6),
        )
        self.annotations.append(t)

    def _show_wobble(self):
        t = self.ax.text(
            0.02, 0.97, f"Jet wobbles ±{P.wobble_amp_deg}° with {JET_WOBBLE_PERIOD_H}h period",
            transform=self.ax.transAxes, fontsize=12, color="#00ffcc", ha="left", va="top", fontweight="bold",
            bbox=dict(facecolor="#1a2a2a", edgecolor="#00ffcc", alpha=0.95, pad=6),
        )
        self.annotations.append(t)

    def _show_sunward(self):
        t = self.ax.text(
            0.02, 0.97, 
            "FIXED-AXIS JET: Sun stays in cone due to 16.4° = 2×8° geometry\n"
            "(Watch 'SUN IN CONE' indicator as orbit turns)",
            transform=self.ax.transAxes, fontsize=11, color="#ffdd44", ha="left", va="top", fontweight="bold",
            bbox=dict(facecolor="#2a2a1a", edgecolor="#ffdd44", alpha=0.95, pad=6),
        )
        self.annotations.append(t)

    def _show_perihelion(self):
        px, py = x_traj[perihelion_idx], y_traj[perihelion_idx]
        mark = self.ax.scatter([px], [py], s=350, marker="*", color="#00ffcc",
                               edgecolors="white", linewidths=2, zorder=25)
        self.annotations.append(mark)

    def _show_ecliptic(self):
        # Note: This is just a reference line - 2D sim doesn't truly model 3D inclination
        line, = self.ax.plot([-3, 3], [0, 0], "-", color="#88ff88", lw=4, alpha=0.5, zorder=4)
        self.annotations.append(line)
        t = self.ax.text(
            0.02, 0.97, 
            f"⚠ NOT MODELED: Ecliptic ±{ECLIPTIC_ALIGNMENT_DEG}° (requires 3D)\n"
            "This 2D sim shows orbital plane, not ecliptic alignment",
            transform=self.ax.transAxes, fontsize=11, color="#88ff88", ha="left", va="top", fontweight="bold",
            bbox=dict(facecolor="#1a2a1a", edgecolor="#88ff88", alpha=0.95, pad=6),
        )
        self.annotations.append(t)

    # --- animation update ---
    def _update(self, frame: int):
        # pause / manual stepping support
        if hasattr(self, "_frame_override"):
            frame = self._frame_override
            delattr(self, "_frame_override")
        if self.paused:
            frame = getattr(self, "_last_frame", frame)

        self._last_frame = frame

        # position
        xi, yi = pos_au[frame]
        self.body.set_offsets([[xi, yi]])
        self.body_label.set_position((xi + 0.08, yi + 0.08))
        self.body_label.set_text("3I/ATLAS")

        # trail
        i0 = max(0, frame - self.trail_len)
        self.trail.set_data(x_traj[i0:frame + 1], y_traj[i0:frame + 1])

        # --- LOEB'S FIXED-AXIS JET GEOMETRY ---
        # Jet rays spread around the FIXED axis (not always-sunward!)
        axis = active_axis[frame]  # +A pre-perihelion, -A post-perihelion
        base = jet_hat[frame]      # axis with wobble applied
        sun = sunward_hat[frame]

        # create offsets across cone around the fixed axis
        idx = np.arange(self.n_rays) - (self.n_rays // 2)
        ang = wobble_rad[frame] + idx * (self.half_angle / max(1, (self.n_rays // 2)))
        ca = np.cos(ang)
        sa = np.sin(ang)
        # rotate the FIXED AXIS by each angle (not sunward)
        dirs = np.column_stack([ca * axis[0] - sa * axis[1], sa * axis[0] + ca * axis[1]])

        # length texture
        phase = (frame * 0.12 + idx * 0.7) % 1.0
        lengths = self.jet_len_au * (0.75 + 0.25 * phase)

        start = np.array([xi, yi])
        end = start + dirs * lengths[:, None]
        segs = np.stack([np.repeat(start[None, :], self.n_rays, axis=0), end], axis=1)
        self.jet_lc.set_segments(segs)

        # Show fixed jet axis line
        axis_end = start + axis * (self.jet_len_au * 1.1)
        self.axis_line.set_data([xi, axis_end[0]], [yi, axis_end[1]])
        
        # Axis label shows which pole is active
        phase_label = "+A (pre)" if frame < perihelion_idx else "-A (post)"
        self.axis_label.set_position((axis_end[0], axis_end[1] + 0.05))
        self.axis_label.set_text(f"Fixed Axis\n{phase_label}")

        # Sun-in-cone indicator
        in_cone = sun_in_cone[frame]
        if in_cone:
            self.sun_cone_indicator.set_text("SUN IN CONE")
            self.sun_cone_indicator.set_color("#00ff88")
            self.sun_cone_indicator.get_bbox_patch().set_edgecolor("#00ff88")
        else:
            self.sun_cone_indicator.set_text("SUN OUTSIDE CONE")
            self.sun_cone_indicator.set_color("#ff4444")
            self.sun_cone_indicator.get_bbox_patch().set_edgecolor("#ff4444")

        # jet label only for claim 2
        if self.current_claim == 2:
            jet_end = start + base * (self.jet_len_au + 0.12)
            self.jet_label.set_position((jet_end[0], jet_end[1]))
            self.jet_label.set_text(f"ANTI-TAIL\n({JET_SPAN_DEG}° cone)")
        else:
            self.jet_label.set_text("")

        # velocity vector
        v_end = start + vel_hat[frame] * 0.30
        self.vel_line.set_data([xi, v_end[0]], [yi, v_end[1]])

        # gravity vector (toward sun in AU-space)
        g = -start
        gn = np.linalg.norm(g)
        g_dir = (g / gn) if gn > 0 else np.array([0.0, 0.0])
        g_end = start + g_dir * 0.20
        self.grav_line.set_data([xi, g_end[0]], [yi, g_end[1]])

        # Normal tail - what a NORMAL comet would do (point AWAY from sun)
        # This is the OPPOSITE of the anti-tail jet
        normal_dir = -g_dir  # Away from sun
        normal_end = start + normal_dir * 0.50  # Length of normal tail arrow
        self.normal_tail_arrow.set_positions((xi, yi), (normal_end[0], normal_end[1]))
        # Label position at end of normal tail
        self.normal_tail_label.set_position((normal_end[0] + normal_dir[0]*0.08, normal_end[1] + normal_dir[1]*0.08))
        self.normal_tail_label.set_text("Normal\ntail")
        
        # Anti-tail label - position next to the jet (using fixed axis, not sunward)
        jet_side = start + axis * (self.jet_len_au * 0.5)  # midpoint of jet
        # Offset perpendicular to jet direction
        perp = np.array([-axis[1], axis[0]]) * 0.15
        self.antitail_label.set_position((jet_side[0] + perp[0], jet_side[1] + perp[1]))

        # info
        t_days = t_anim[frame] / 86400.0
        r_au = distances[frame]
        v_kms = vel_norm[frame] / 1000.0
        phase_txt = "INCOMING" if frame < perihelion_idx else "OUTGOING"
        extra = f"\nWobble angle: {wobble_deg[frame]:+.1f}°" if self.current_claim == 3 else ""
        
        # Add sun-in-cone status to info when claim 1 or 4 is selected
        cone_info = ""
        if self.current_claim in [1, 4]:
            cone_info = f"\nSun in cone: {'YES' if in_cone else 'NO'}"

        self.info.set_text(
            f"═══ 3I/ATLAS ═══\n"
            f"Phase: {phase_txt}\n"
            f"Day: {t_days:+.0f}\n"
            f"Distance: {r_au:.2f} AU\n"
            f"Speed: {v_kms:.0f} km/s{extra}{cone_info}"
        )

        # return artists for blitting
        return (
            self.body,
            self.body_label,
            self.trail,
            self.jet_lc,
            self.jet_label,
            self.antitail_label,
            self.axis_line,
            self.axis_label,
            self.sun_cone_indicator,
            self.vel_line,
            self.grav_line,
            self.normal_tail_arrow,
            self.normal_tail_label,
            self.info,
            *self.annotations,
        )

    def run(self):
        plt.show(block=True)


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    print("Starting interactive simulation...")
    print("Controls: SPACE=Pause, Left/Right=Step (when paused), 1-6=Select claim\n")
    sim = InteractiveSimulation()
    sim.run()
