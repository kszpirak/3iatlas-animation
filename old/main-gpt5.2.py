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

# Loeb article values (as you provided)
RP_KM = 202_000_000
VP_KMS = 68
LOEB_DEFLECTION_DEG = 16.4
JET_SPAN_DEG = 8.0
JET_WOBBLE_PERIOD_H = 7.74
ECLIPTIC_ALIGNMENT_DEG = 5

# ----------------------------
# Claims panel
# ----------------------------

CLAIMS = {
    1: {
        "title": "16.4° Gravitational Deflection",
        "short": "Deflection = 2× Jet",
        "loeb_claim": """The gravitational deflection angle of 3I/ATLAS is precisely 16.4°.

This is EXACTLY twice the jet cone angle (8°).

Loeb argues this is not coincidental — it's the exact geometry
needed for a sunward-pointing jet to remain aimed at the Sun
both BEFORE and AFTER perihelion passage.

The math: deflection = 2GM/(b × v²) where b is impact
parameter and v is velocity at infinity.""",
        "commentary": """The 2:1 ratio is geometrically elegant but not impossible naturally.

Standard physics explanation:
• Deflection angle follows from Newtonian gravity
• The specific value depends on approach speed & distance
• Many hyperbolic orbits could have this deflection

Skeptical view:
• Coincidences happen — we notice "special" numbers
• Selection bias: we're examining THIS object because it's unusual
• The ratio being exactly 2:1 may be approximate

Verdict: Interesting but not conclusive evidence of design.""",
        "source": "https://avi-loeb.medium.com/six-anomalies-of-3i-atlas-f9a4dbb06db7",
        "highlight": "deflection",
    },
    2: {
        "title": "Anti-Tail Jet (~8° cone)",
        "short": "Anti-Tail Jet 8°",
        "loeb_claim": """3I/ATLAS displays a jet pointing TOWARD the Sun.

Key observations:
• Jet spans approximately 8° angular width
• Extends over 1 million kilometers
• Points sunward throughout the flyby

This is the OPPOSITE of normal comet behavior!

Normal comets have tails pointing AWAY from Sun due to
solar radiation pressure pushing dust and gas outward.

No known natural comet exhibits a persistent physical
sunward jet of this nature.""",
        "commentary": """Critical distinction: "anti-tail" vs physical sunward jet.

Normal comet "anti-tails" explained:
• Dust tail lies in orbital plane
• When Earth crosses this plane, perspective makes
  part of tail APPEAR to point sunward
• This is purely an OPTICAL ILLUSION

What Loeb claims for 3I/ATLAS:
• Material physically moving TOWARD the Sun
• Fighting against solar radiation pressure
• Requires energy input to maintain

If confirmed as a real sunward jet, this would be
truly anomalous. Key question: Is it propulsion or
unusual outgassing physics we don't yet understand?""",
        "source": "https://avi-loeb.medium.com/six-anomalies-of-3i-atlas-f9a4dbb06db7",
        "highlight": "jet_cone",
    },
    3: {
        "title": "7.74h Wobble Period",
        "short": "Wobble: 7.74 hours",
        "loeb_claim": """The jet exhibits periodic wobble with:
• Amplitude: ±4° deviation from mean direction
• Period: 7.74 hours

Loeb's interpretation:
The nucleus is SPINNING with a 7.74-hour rotation period.
The jet source is located near (but not exactly at) the
rotation pole, causing it to precess as the body spins.

This is like a spinning top with a slightly tilted axis —
the jet traces out a cone as it rotates.""",
        "commentary": """This is actually a reasonable and conventional interpretation.

Supporting evidence:
• Many asteroids have rotation periods of 2-20 hours
• 7.74h is well within normal range
• Off-axis jets are common on comets (e.g., 67P)

The wobble tells us:
• Nucleus is relatively small (fast rotation)
• Jet is fixed to surface, not actively steered
• Rotation axis roughly aligned with Sun direction

This is perhaps the LEAST anomalous of Loeb's claims.
Standard cometary physics can explain wobbling jets.

Verdict: Normal behavior, actually argues AGAINST
artificial origin (a probe would likely have stable pointing).""",
        "source": "https://avi-loeb.medium.com/six-anomalies-of-3i-atlas-f9a4dbb06db7",
        "highlight": "wobble",
    },
    4: {
        "title": "Sunward Jet Maintained",
        "short": "Always → Sun",
        "loeb_claim": """The jet maintains sunward orientation throughout the flyby.

Normal comet behavior:
• Tails point AWAY from Sun (radiation pressure)
• Orientation changes as comet moves

3I/ATLAS behavior:
• Jet points TOWARD Sun — the opposite!
• Maintained before, during, AND after perihelion
• Tracks the Sun as the object's position changes

This requires the jet direction to continuously adjust
to keep pointing at the Sun as 3I/ATLAS moves along
its hyperbolic trajectory.""",
        "commentary": """This is the CORE anomaly that drives Loeb's hypothesis.

The physics problem:
• Solar radiation exerts outward pressure (~4.5 μN/m² at 1 AU)
• Any emitted material should be pushed AWAY from Sun
• Maintaining sunward flow requires continuous energy input

Possible explanations:

1. PROPULSION (Loeb's suggestion)
   • Deliberate thrust toward Sun
   • Would require fuel/energy source

2. UNUSUAL OUTGASSING
   • Unknown volatile with strange sublimation behavior?
   • Magnetic field interactions?
   • We don't fully understand interstellar composition

3. MEASUREMENT ERROR
   • Is the direction truly sunward or misinterpreted?
   • Need independent confirmation

This remains genuinely puzzling if observations are correct.""",
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
        "title": "Ecliptic Alignment",
        "short": "Ecliptic ±5°",
        "loeb_claim": """3I/ATLAS travels within 5° of the ecliptic plane.

The ecliptic is the plane containing Earth's orbit and
(approximately) all major planets.

Random probability calculation:
• Sky is a sphere, ecliptic is a narrow band
• Chance of random object being within ±5°: ~0.2%
• This seems suspiciously well-aligned

Loeb suggests this alignment might indicate the object
was TARGETED at our solar system, traveling in the
plane where planets (and observers) are located.""",
        "commentary": """Statistically interesting but several caveats apply.

Observational bias:
• We survey the ecliptic more intensively
• Pan-STARRS, ATLAS focus near ecliptic
• Objects there are MORE LIKELY to be discovered
• This alone could explain the alignment

Small number statistics:
• Only 3 confirmed interstellar objects (1I, 2I, 3I)
• Drawing conclusions from n=3 is dangerous
• 1I 'Oumuamua was ~30° from ecliptic
• 2I Borisov was ~44° from ecliptic
• So 3I is actually the OUTLIER, not the norm

Physical considerations:
• Interstellar objects come from random directions
• No known mechanism would align them to OUR ecliptic
• Each star system has different orbital plane

Verdict: Likely observational selection effect.""",
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

# Wobble angle (rad)
period_s = JET_WOBBLE_PERIOD_H * 3600.0
wobble_deg = P.wobble_amp_deg * np.sin(2.0 * np.pi * (t_anim / period_s))
wobble_rad = np.deg2rad(wobble_deg)

# Rotate sunward by wobble to get jet direction in AU-space (unit-ish)
c = np.cos(wobble_rad)
s = np.sin(wobble_rad)
jet_hat = np.column_stack([c * sunward_hat[:, 0] - s * sunward_hat[:, 1],
                           s * sunward_hat[:, 0] + c * sunward_hat[:, 1]])

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
        ax.set_title("3I/ATLAS: Interstellar Object with Anti-Tail Jet", fontsize=16, fontweight="bold", color="white", pad=15)
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
        self.jet_len_au = 0.60
        self.half_angle = math.radians(JET_SPAN_DEG / 2.0)
        segs = np.zeros((self.n_rays, 2, 2), dtype=float)
        self.jet_lc = LineCollection(segs, linewidths=np.linspace(3.0, 1.0, self.n_rays),
                                     colors=[(1.0, 0.67, 0.0, 0.8)], zorder=15)
        ax.add_collection(self.jet_lc)

        self.jet_label = ax.text(0, 0, "", fontsize=10, color="#ffaa00", fontweight="bold", ha="center", va="bottom", zorder=22)
        
        # Anti-tail label (always visible)
        self.antitail_label = ax.text(0, 0, "Anti-Tail", fontsize=9, color="#ffaa00", fontweight="bold", 
                                       ha="left", va="center", zorder=22)

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

        # ===== "Normal Comet" comparison in corner =====
        # Draw a small schematic showing how normal comets behave
        self._draw_normal_comet_comparison(ax)

        # ===== Legend - positioned outside plot on the right =====
        legend_items = [
            plt.Line2D([0], [0], color='#00ffcc', marker='o', markersize=10, linestyle='None', label='3I/ATLAS'),
            plt.Line2D([0], [0], color='#66aaff', linewidth=2, label='Recent path (trail)'),
            plt.Line2D([0], [0], color='#ffaa00', linewidth=4, label='ANTI-TAIL (jet -> Sun)'),
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
                plt.close(self.details_fig)
            except:
                pass
            self.details_fig = None
        
        # Show details for current claim, or all claims if none selected
        if self.current_claim:
            self._show_single_claim_details(self.current_claim)
        else:
            self._show_all_claims_details()

    def _show_single_claim_details(self, claim_id):
        """Show detailed view of a single claim"""
        claim = CLAIMS[claim_id]
        
        fig = plt.figure(figsize=(18, 10), facecolor="#0a0a1a")
        self.details_fig = fig  # Store reference
        
        # Handle window close
        def on_close(evt):
            self.details_fig = None
        fig.canvas.mpl_connect('close_event', on_close)
        
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
        arc = Arc((0, 0), 1.2, 1.2, angle=0, theta1=70, theta2=70 + LOEB_DEFLECTION_DEG,
                  color="#ffff00", lw=4, zorder=25)
        self.ax.add_patch(arc)
        self.extra_artists.append(arc)

        t = self.ax.text(
            0.02, 0.97, f"Deflection ({LOEB_DEFLECTION_DEG}°) = 2 × Jet ({JET_SPAN_DEG}°)",
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
            0.02, 0.97, "Jet ALWAYS points toward Sun throughout flyby!",
            transform=self.ax.transAxes, fontsize=12, color="#ffdd44", ha="left", va="top", fontweight="bold",
            bbox=dict(facecolor="#2a2a1a", edgecolor="#ffdd44", alpha=0.95, pad=6),
        )
        self.annotations.append(t)

    def _show_perihelion(self):
        px, py = x_traj[perihelion_idx], y_traj[perihelion_idx]
        mark = self.ax.scatter([px], [py], s=350, marker="*", color="#00ffcc",
                               edgecolors="white", linewidths=2, zorder=25)
        self.annotations.append(mark)

    def _show_ecliptic(self):
        line, = self.ax.plot([-3, 3], [0, 0], "-", color="#88ff88", lw=4, alpha=0.8, zorder=4)
        self.annotations.append(line)
        t = self.ax.text(
            0.02, 0.97, f"Aligned within {ECLIPTIC_ALIGNMENT_DEG}° of ecliptic",
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

        # jet rays as segments
        # spread around the cone; add small “flow” length variation for texture
        base = jet_hat[frame]  # unit-ish, in AU-space
        sun = sunward_hat[frame]
        wob = wobble_rad[frame]

        # create offsets across cone
        idx = np.arange(self.n_rays) - (self.n_rays // 2)
        ang = wob + idx * (self.half_angle / max(1, (self.n_rays // 2)))
        ca = np.cos(ang)
        sa = np.sin(ang)
        # rotate sunward by each ang
        dirs = np.column_stack([ca * sun[0] - sa * sun[1], sa * sun[0] + ca * sun[1]])

        # length texture
        phase = (frame * 0.12 + idx * 0.7) % 1.0
        lengths = self.jet_len_au * (0.75 + 0.25 * phase)

        start = np.array([xi, yi])
        end = start + dirs * lengths[:, None]
        segs = np.stack([np.repeat(start[None, :], self.n_rays, axis=0), end], axis=1)
        self.jet_lc.set_segments(segs)

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
        
        # Anti-tail label - position next to the jet
        jet_side = start + sun * (self.jet_len_au * 0.5)  # midpoint of jet
        # Offset perpendicular to jet direction
        perp = np.array([-sun[1], sun[0]]) * 0.15
        self.antitail_label.set_position((jet_side[0] + perp[0], jet_side[1] + perp[1]))

        # info
        t_days = t_anim[frame] / 86400.0
        r_au = distances[frame]
        v_kms = vel_norm[frame] / 1000.0
        phase_txt = "INCOMING" if frame < perihelion_idx else "OUTGOING"
        extra = f"\nWobble angle: {wobble_deg[frame]:+.1f}°" if self.current_claim == 3 else ""

        self.info.set_text(
            f"═══ 3I/ATLAS ═══\n"
            f"Phase: {phase_txt}\n"
            f"Day: {t_days:+.0f}\n"
            f"Distance: {r_au:.2f} AU\n"
            f"Speed: {v_kms:.0f} km/s{extra}"
        )

        # return artists for blitting
        return (
            self.body,
            self.body_label,
            self.trail,
            self.jet_lc,
            self.jet_label,
            self.antitail_label,
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
