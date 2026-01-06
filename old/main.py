"""
3I/ATLAS Interactive Simulation with Visual Anti-Tail
- Solar gravity (2-body) + time-varying sunward jet with wobble
- Interactive claims from Avi Loeb's article (Dec 17, 2025)
- Visual anti-tail jet stream with particles

Requires: numpy, scipy, matplotlib
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from matplotlib.patches import Arc, Wedge, FancyArrowPatch, Polygon
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors


# ----------------------------
# Constants & article-linked parameters
# ----------------------------

MU_SUN = 1.32712440018e20  # m^3/s^2
AU = 1.496e11  # meters

# Loeb article values:
RP_KM = 202_000_000  # perihelion distance b ~ 202 million km
VP_KMS = 68          # perihelion speed ~ 68 km/s
LOEB_DEFLECTION_DEG = 16.4  # Loeb's estimate from 2GM/(bv²)
JET_SPAN_DEG = 8.0          # observed anti-tail span ~ 8 degrees
JET_WOBBLE_PERIOD_H = 7.74  # wobble period from observations
ECLIPTIC_ALIGNMENT_DEG = 5  # retrograde trajectory aligned within 5° of ecliptic

# ----------------------------
# Claims from Loeb's Article
# ----------------------------

CLAIMS = {
    1: {
        "title": "16.4° Gravitational Deflection",
        "short": "Deflection = 2× Jet",
        "description": "Deflection (16.4°) = 2× jet angle (8°)\nAllows jet to stay sunward before\nAND after perihelion passage.",
        "highlight": "deflection"
    },
    2: {
        "title": "Anti-Tail Jet (~8° cone)", 
        "short": "Anti-Tail Jet 8°",
        "description": "ANTI-TAIL: Jet points TOWARD Sun\n(opposite of normal comet tails)\nSpans 8°, extends >1 million km\nNo known comet does this!",
        "highlight": "jet_cone"
    },
    3: {
        "title": "7.74h Wobble Period",
        "short": "Wobble: 7.74 hours",
        "description": "Jet wobbles ±4° with 7.74h period\nIndicates spinning nucleus with\njet base near rotation pole.",
        "highlight": "wobble"
    },
    4: {
        "title": "Sunward Jet Maintained",
        "short": "Always → Sun",
        "description": "Unlike normal comets (tail away),\n3I/ATLAS jet ALWAYS points at Sun.\nMaintained through entire flyby!",
        "highlight": "sunward"
    },
    5: {
        "title": "Perihelion Parameters",
        "short": "Closest Approach",
        "description": f"PERIHELION (closest to Sun):\nDistance: {RP_KM/1e6:.0f} million km\nSpeed: {VP_KMS} km/s\nFormula: 2GM/(bv²) = 16.4°",
        "highlight": "perihelion"
    },
    6: {
        "title": "Ecliptic Alignment",
        "short": "Ecliptic ±5°",
        "description": "Travels in plane of planets!\nWithin 5° of ecliptic plane.\nChance probability: only 0.2%",
        "highlight": "ecliptic"
    },
}


# ----------------------------
# Model Parameters
# ----------------------------

@dataclass(frozen=True)
class ModelParams:
    wobble_amp_deg: float = 4.0
    a0: float = 5e-5
    r_ref: float = 1.0 * AU
    r_cut: float = 2.0 * AU
    t_span_days: float = 120.0
    r_stop: float = 3.5 * AU


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
    mag = params.a0 * (params.r_ref / r) ** 2 * math.exp(-(r / params.r_cut) ** 2)
    return mag * dir_vec


def rhs(t: float, y: np.ndarray, params: ModelParams) -> np.ndarray:
    x, y_pos, vx, vy = y
    r_vec = np.array([x, y_pos], dtype=float)
    r = float(np.linalg.norm(r_vec))
    a_grav = np.zeros(2) if r == 0.0 else -MU_SUN * r_vec / (r ** 3)
    a_jet = jet_accel(t, r_vec, params)
    ax, ay = a_grav[0] + a_jet[0], a_grav[1] + a_jet[1]
    return np.array([vx, vy, ax, ay], dtype=float)


def stop_when_far(t: float, y: np.ndarray, params: ModelParams) -> float:
    return params.r_stop - math.hypot(y[0], y[1])


stop_when_far.terminal = True
stop_when_far.direction = -1


# ----------------------------
# Compute Trajectory
# ----------------------------

print("Computing trajectory...")
rp = RP_KM * 1000.0
vp = VP_KMS * 1000.0
y0 = np.array([rp, 0.0, 0.0, vp], dtype=float)
tmax = P.t_span_days * 86400.0

sol_fwd = solve_ivp(
    fun=lambda t, y: rhs(t, y, P),
    t_span=(0.0, tmax), y0=y0, max_step=6*3600,
    rtol=1e-9, atol=1e-9,
    events=lambda t, y: stop_when_far(t, y, P),
)

sol_bwd = solve_ivp(
    fun=lambda t, y: rhs(t, y, P),
    t_span=(0.0, -tmax), y0=y0, max_step=6*3600,
    rtol=1e-9, atol=1e-9,
    events=lambda t, y: stop_when_far(t, y, P),
)

t_b, y_b = sol_bwd.t[::-1], sol_bwd.y[:, ::-1]
t_f, y_f = sol_fwd.t, sol_fwd.y
t_all = np.concatenate([t_b[:-1], t_f])
y_all = np.concatenate([y_b[:, :-1], y_f], axis=1)

# Downsample
N = y_all.shape[1]
stride = max(1, N // 1200)
t_anim = t_all[::stride]
y_anim = y_all[:, ::stride]
Nf = y_anim.shape[1]
x_traj = y_anim[0] / AU
y_traj = y_anim[1] / AU

# Find perihelion index (closest to sun)
distances = np.sqrt(x_traj**2 + y_traj**2)
perihelion_idx = np.argmin(distances)

print(f"Trajectory computed: {Nf} frames, perihelion at frame {perihelion_idx}")


# ----------------------------
# Interactive Animation
# ----------------------------

class InteractiveSimulation:
    def __init__(self):
        self.current_claim = None
        self.frame_idx = 0
        self.annotations = []
        self.extra_artists = []
        
        # Create figure - extra wide
        self.fig = plt.figure(figsize=(20, 11), facecolor='#0a0a1a')
        
        # Main animation axes - leave room on right for legend
        self.ax = self.fig.add_axes([0.02, 0.06, 0.62, 0.90])
        self.ax.set_facecolor('#0a0a1a')
        self.ax.set_aspect("equal", "box")
        
        self._setup_main_plot()
        self._setup_claims_panel()
        self._setup_animation()
        
    def _setup_main_plot(self):
        ax = self.ax
        ax.set_title("3I/ATLAS: Interstellar Object with Anti-Tail Jet", 
                     fontsize=16, fontweight='bold', color='white', pad=15)
        ax.set_xlabel("Distance from Sun (AU)", fontsize=12, color='white')
        ax.set_ylabel("Distance from Sun (AU)", fontsize=12, color='white')
        ax.tick_params(colors='white', labelsize=10)
        for spine in ax.spines.values():
            spine.set_color('#333355')
        
        # Full trajectory (faint)
        ax.plot(x_traj, y_traj, linewidth=2, color='#4488ff', alpha=0.3, label='Trajectory')
        
        # Mark incoming vs outgoing with arrows
        # Incoming arrow
        mid_in = perihelion_idx // 2
        ax.annotate('', xy=(x_traj[mid_in+5], y_traj[mid_in+5]), 
                   xytext=(x_traj[mid_in-5], y_traj[mid_in-5]),
                   arrowprops=dict(arrowstyle='->', color='#6699ff', lw=2))
        ax.text(x_traj[mid_in], y_traj[mid_in]+0.15, 'INCOMING', fontsize=9, 
                color='#6699ff', ha='center', fontweight='bold')
        
        # Outgoing arrow  
        mid_out = perihelion_idx + (Nf - perihelion_idx) // 2
        ax.annotate('', xy=(x_traj[min(mid_out+5, Nf-1)], y_traj[min(mid_out+5, Nf-1)]), 
                   xytext=(x_traj[mid_out-5], y_traj[mid_out-5]),
                   arrowprops=dict(arrowstyle='->', color='#6699ff', lw=2))
        ax.text(x_traj[mid_out], y_traj[mid_out]-0.15, 'OUTGOING', fontsize=9, 
                color='#6699ff', ha='center', fontweight='bold')
        
        # ===== SUN (prominent) =====
        ax.scatter([0], [0], s=3000, color='#ffdd44', alpha=0.1, zorder=5)
        ax.scatter([0], [0], s=1500, color='#ffdd44', alpha=0.2, zorder=5)
        ax.scatter([0], [0], s=700, color='#ffcc00', edgecolors='#ff8800', 
                   linewidths=3, zorder=6)
        ax.text(0, 0, '☀', fontsize=24, ha='center', va='center', color='#ffff88', zorder=7)
        ax.text(0, -0.32, 'SUN', fontsize=14, ha='center', va='top', 
                color='#ffdd44', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a0a', 
                         edgecolor='#ffaa00', alpha=0.9))
        
        # ===== 3I/ATLAS body =====
        self.body, = ax.plot([], [], marker="o", markersize=14, color='#00ffcc', 
                             linestyle="None", markeredgecolor='white', 
                             markeredgewidth=2, zorder=20, label='3I/ATLAS')
        self.body_label = ax.text(0, 0, '', fontsize=9, color='#00ffcc', 
                                  fontweight='bold', ha='left', va='bottom', zorder=21)
        
        # ===== ANTI-TAIL JET (the key visual!) =====
        # Create multiple lines for particle stream effect
        self.jet_particles = []
        n_particles = 15
        for i in range(n_particles):
            alpha = 0.8 - (i * 0.05)
            width = 3 - (i * 0.15)
            line, = ax.plot([], [], linewidth=width, color='#ffaa00', alpha=alpha, zorder=15)
            self.jet_particles.append(line)
        
        # Jet cone fill (wedge shape)
        self.jet_wedge = None
        
        # Jet label
        self.jet_label = ax.text(0, 0, '', fontsize=10, color='#ffaa00', 
                                 fontweight='bold', ha='center', va='bottom', zorder=22)
        
        # ===== Vectors =====
        self.vel_arrow, = ax.plot([], [], linewidth=3, color='#00ff88', 
                                  solid_capstyle='round', zorder=18)
        self.vel_head = ax.scatter([], [], s=100, marker='>', color='#00ff88', zorder=18)
        
        # Gravity vector (pointing to sun)
        self.grav_arrow, = ax.plot([], [], linewidth=2, color='#ff6666', 
                                   linestyle='--', alpha=0.7, zorder=17)
        
        # ===== Legend - positioned outside plot on the right =====
        legend_items = [
            plt.Line2D([0], [0], color='#00ffcc', marker='o', markersize=10, linestyle='None', label='3I/ATLAS'),
            plt.Line2D([0], [0], color='#ffaa00', linewidth=4, label='ANTI-TAIL (jet → Sun)'),
            plt.Line2D([0], [0], color='#00ff88', linewidth=3, label='Velocity'),
            plt.Line2D([0], [0], color='#ff6666', linewidth=2, linestyle='--', label='Gravity → Sun'),
        ]
        ax.legend(handles=legend_items, loc='upper left', bbox_to_anchor=(1.01, 1.0),
                  fontsize=9, facecolor='#1a1a2e', edgecolor='#333355', 
                  labelcolor='white', framealpha=0.95)
        
        # ===== Info box - lower right corner =====
        self.info = ax.text(0.99, 0.02, "", transform=ax.transAxes, va="bottom", ha="right",
                           fontsize=10, color='white', family='monospace',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e', 
                                    edgecolor='#333355', alpha=0.95), zorder=25)
        
        # View limits - generous padding
        pad = 0.8
        ax.set_xlim(x_traj.min() - pad, x_traj.max() + pad)
        ax.set_ylim(y_traj.min() - pad, y_traj.max() + pad)
        ax.grid(True, alpha=0.15, color='#4444aa', linestyle='--')
        
        # Store half angle for jet cone
        self.half_angle = math.radians(JET_SPAN_DEG / 2.0)
        
    def _setup_claims_panel(self):
        # Title
        self.fig.text(0.90, 0.93, "LOEB'S ANOMALIES", fontsize=14, fontweight='bold', 
                     color='#ffcc00', ha='center', transform=self.fig.transFigure)
        self.fig.text(0.90, 0.905, "Click to visualize", fontsize=9, 
                     color='#888899', ha='center', transform=self.fig.transFigure)
        
        # Create buttons
        self.buttons = []
        button_height = 0.060
        start_y = 0.84
        
        for i, (claim_id, claim) in enumerate(CLAIMS.items()):
            y_pos = start_y - i * (button_height + 0.010)
            btn_ax = self.fig.add_axes([0.78, y_pos, 0.20, button_height])
            btn = Button(btn_ax, f"{claim_id}. {claim['short']}", 
                        color='#1a1a3e', hovercolor='#3a3a6e')
            btn.label.set_color('white')
            btn.label.set_fontsize(9)
            btn.label.set_fontweight('bold')
            btn.on_clicked(lambda event, cid=claim_id: self.select_claim(cid))
            self.buttons.append((btn, btn_ax))
        
        # Reset button
        reset_ax = self.fig.add_axes([0.78, 0.10, 0.20, 0.042])
        self.reset_btn = Button(reset_ax, "RESET VIEW", color='#333355', hovercolor='#555588')
        self.reset_btn.label.set_color('white')
        self.reset_btn.label.set_fontweight('bold')
        self.reset_btn.on_clicked(self.reset_view)
        
        # Description box
        desc_ax = self.fig.add_axes([0.78, 0.15, 0.20, 0.18])
        desc_ax.set_facecolor('#1a1a2e')
        for spine in desc_ax.spines.values():
            spine.set_color('#444466')
            spine.set_linewidth(2)
        desc_ax.set_xticks([])
        desc_ax.set_yticks([])
        self.desc_ax = desc_ax
        
        self.desc_title = desc_ax.text(0.5, 0.88, "Select a claim", fontsize=11, 
                                       color='#ffcc00', ha='center', va='top', fontweight='bold')
        self.desc_text = desc_ax.text(0.5, 0.72, "", fontsize=9, color='#ccccee',
                                      ha='center', va='top', linespacing=1.5)
        
    def _setup_animation(self):
        self.ani = FuncAnimation(self.fig, self._update, frames=Nf, 
                                 interval=30, blit=False, repeat=True)
        
    def select_claim(self, claim_id):
        self.current_claim = claim_id
        claim = CLAIMS[claim_id]
        
        # Update button colors
        for i, (btn, btn_ax) in enumerate(self.buttons):
            btn_ax.set_facecolor('#4a4a8e' if i + 1 == claim_id else '#1a1a3e')
        
        # Update description
        self.desc_title.set_text(claim['title'])
        self.desc_text.set_text(claim['description'])
        
        # Clear old annotations
        self._clear_annotations()
        
        # Show claim-specific visuals
        highlight = claim['highlight']
        if highlight == 'deflection':
            self._show_deflection()
        elif highlight == 'jet_cone':
            self._show_jet_cone()
        elif highlight == 'wobble':
            self._show_wobble()
        elif highlight == 'sunward':
            self._show_sunward()
        elif highlight == 'perihelion':
            self._show_perihelion()
        elif highlight == 'ecliptic':
            self._show_ecliptic()
            
        self.fig.canvas.draw_idle()
        
    def _clear_annotations(self):
        for item in self.annotations:
            try:
                item.remove()
            except:
                pass
        self.annotations = []
        for item in self.extra_artists:
            try:
                item.remove()
            except:
                pass
        self.extra_artists = []
        
    def _show_deflection(self):
        """Show the 16.4° deflection with incoming/outgoing paths"""
        # Draw velocity vectors at start and end
        # Incoming velocity direction
        v_in = np.array([y_anim[2, 10], y_anim[3, 10]])
        v_in = v_in / np.linalg.norm(v_in) * 0.8
        start_pos = np.array([x_traj[10], y_traj[10]])
        
        arr1 = self.ax.annotate('', xy=start_pos + v_in, xytext=start_pos,
                                arrowprops=dict(arrowstyle='->', color='#88ff88', lw=3))
        self.annotations.append(arr1)
        lbl1 = self.ax.text(start_pos[0] + v_in[0], start_pos[1] + v_in[1] + 0.1,
                           'V incoming', color='#88ff88', fontsize=10, ha='center', fontweight='bold')
        self.annotations.append(lbl1)
        
        # Outgoing velocity direction
        v_out = np.array([y_anim[2, -10], y_anim[3, -10]])
        v_out = v_out / np.linalg.norm(v_out) * 0.8
        end_pos = np.array([x_traj[-10], y_traj[-10]])
        
        arr2 = self.ax.annotate('', xy=end_pos + v_out, xytext=end_pos,
                                arrowprops=dict(arrowstyle='->', color='#ff8888', lw=3))
        self.annotations.append(arr2)
        lbl2 = self.ax.text(end_pos[0] + v_out[0], end_pos[1] + v_out[1] - 0.1,
                           'V outgoing', color='#ff8888', fontsize=10, ha='center', fontweight='bold')
        self.annotations.append(lbl2)
        
        # Draw deflection arc at perihelion
        perihelion_x = x_traj[perihelion_idx]
        arc = Arc((0, 0), 1.2, 1.2, angle=0, 
                  theta1=70, theta2=70+LOEB_DEFLECTION_DEG, 
                  color='#ffff00', lw=4, zorder=25)
        self.ax.add_patch(arc)
        self.extra_artists.append(arc)
        
        # Angle label - positioned away from trajectory
        lbl3 = self.ax.text(-0.5, 0.9, f'DEFLECTION\n{LOEB_DEFLECTION_DEG}°', 
                           fontsize=14, color='#ffff00', ha='center', fontweight='bold',
                           bbox=dict(facecolor='#1a1a0a', edgecolor='#ffff00', alpha=0.9, pad=5))
        self.annotations.append(lbl3)
        
        # Explanation - at top left, not center
        lbl4 = self.ax.text(0.02, 0.97, f'Deflection ({LOEB_DEFLECTION_DEG}°) = 2 × Jet angle ({JET_SPAN_DEG}°)',
                           transform=self.ax.transAxes, fontsize=11, color='#ffff88',
                           ha='left', va='top', fontweight='bold',
                           bbox=dict(facecolor='#2a2a1a', edgecolor='#ffff00', alpha=0.95, pad=6))
        self.annotations.append(lbl4)
        
    def _show_jet_cone(self):
        """Highlight the anti-tail jet cone"""
        # Big label explaining anti-tail - top left
        lbl = self.ax.text(0.02, 0.97, 
                          'ANTI-TAIL: Jet points TOWARD Sun (opposite of normal comets!)',
                          transform=self.ax.transAxes, fontsize=11, color='#ffaa00',
                          ha='left', va='top', fontweight='bold',
                          bbox=dict(facecolor='#2a2a1a', edgecolor='#ffaa00', alpha=0.95, pad=6))
        self.annotations.append(lbl)
        
        # Draw a big cone from current position to sun to show the jet
        lbl2 = self.ax.text(0.02, 0.88, f'Jet opening angle: {JET_SPAN_DEG}°\nExtends >1 million km!',
                           transform=self.ax.transAxes, fontsize=10, color='#ffcc00',
                           va='top', fontweight='bold',
                           bbox=dict(facecolor='#1a1a0a', edgecolor='#ffaa00', alpha=0.9, pad=4))
        self.annotations.append(lbl2)
        
        # Normal comet comparison
        lbl3 = self.ax.text(0.02, 0.74, 'Normal comet: tail points AWAY from Sun\n3I/ATLAS: jet points AT Sun!',
                           transform=self.ax.transAxes, fontsize=9, color='#aaaacc',
                           va='top', style='italic')
        self.annotations.append(lbl3)
        
    def _show_wobble(self):
        """Show the wobble motion"""
        lbl = self.ax.text(0.02, 0.97, f'Jet wobbles ±{P.wobble_amp_deg}° with {JET_WOBBLE_PERIOD_H}h period',
                          transform=self.ax.transAxes, fontsize=12, color='#00ffcc',
                          ha='left', va='top', fontweight='bold',
                          bbox=dict(facecolor='#1a2a2a', edgecolor='#00ffcc', alpha=0.95, pad=6))
        self.annotations.append(lbl)
        
        lbl2 = self.ax.text(0.02, 0.88, 'Watch the jet oscillate\nback and forth!',
                           transform=self.ax.transAxes, fontsize=10, color='#88ffff',
                           va='top', fontweight='bold')
        self.annotations.append(lbl2)
        
    def _show_sunward(self):
        """Show jet always pointing at sun"""
        # Draw lines from multiple trajectory points to sun
        for idx in range(0, Nf, Nf//8):
            x, y = x_traj[idx], y_traj[idx]
            line, = self.ax.plot([x, 0], [y, 0], '-', color='#ffaa00', alpha=0.3, lw=1)
            self.annotations.append(line)
        
        lbl = self.ax.text(0.02, 0.97, 'Jet ALWAYS points toward Sun throughout flyby!',
                          transform=self.ax.transAxes, fontsize=12, color='#ffdd44',
                          ha='left', va='top', fontweight='bold',
                          bbox=dict(facecolor='#2a2a1a', edgecolor='#ffdd44', alpha=0.95, pad=6))
        self.annotations.append(lbl)
        
        lbl2 = self.ax.text(0.02, 0.88, 'Orange lines show jet\ndirection at each point',
                           transform=self.ax.transAxes, fontsize=9, color='#ffcc88', va='top')
        self.annotations.append(lbl2)
        
    def _show_perihelion(self):
        """Mark perihelion point"""
        px, py = x_traj[perihelion_idx], y_traj[perihelion_idx]
        
        # Big marker
        mark = self.ax.scatter([px], [py], s=400, marker='*', color='#00ffcc', 
                               edgecolors='white', linewidths=2, zorder=25)
        self.annotations.append(mark)
        
        # Distance line
        line, = self.ax.plot([0, px], [0, py], '-', color='#00ffcc', lw=2, alpha=0.7)
        self.annotations.append(line)
        
        # Label
        dist_au = np.sqrt(px**2 + py**2)
        lbl = self.ax.annotate(
            f'PERIHELION\nClosest approach\n{RP_KM/1e6:.0f}M km = {dist_au:.2f} AU\nSpeed: {VP_KMS} km/s',
            xy=(px, py), xytext=(px + 0.5, py + 0.4),
            fontsize=11, color='#00ffcc', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#00ffcc', lw=2),
            bbox=dict(facecolor='#1a2a2a', edgecolor='#00ffcc', alpha=0.95, pad=8))
        self.annotations.append(lbl)
        
    def _show_ecliptic(self):
        """Show ecliptic plane"""
        # Draw ecliptic as thick line
        line, = self.ax.plot([-3, 3], [0, 0], '-', color='#88ff88', lw=4, alpha=0.8, zorder=4)
        self.annotations.append(line)
        
        # Fill above/below
        fill1 = self.ax.fill_between([-3, 3], [0, 0], [0.15, 0.15], color='#88ff88', alpha=0.1)
        fill2 = self.ax.fill_between([-3, 3], [0, 0], [-0.15, -0.15], color='#88ff88', alpha=0.1)
        self.extra_artists.append(fill1)
        self.extra_artists.append(fill2)
        
        lbl1 = self.ax.text(2.0, 0.12, 'ECLIPTIC PLANE', fontsize=11, color='#88ff88', 
                           fontweight='bold', ha='center')
        self.annotations.append(lbl1)
        
        lbl2 = self.ax.text(2.0, -0.05, '(plane of planets)', fontsize=9, color='#66cc66', 
                           ha='center', style='italic')
        self.annotations.append(lbl2)
        
        lbl3 = self.ax.text(0.02, 0.97, f'3I/ATLAS aligned within {ECLIPTIC_ALIGNMENT_DEG}° of ecliptic (0.2% chance!)',
                           transform=self.ax.transAxes, fontsize=11, color='#88ff88',
                           ha='left', va='top', fontweight='bold',
                           bbox=dict(facecolor='#1a2a1a', edgecolor='#88ff88', alpha=0.95, pad=6))
        self.annotations.append(lbl3)
        
    def reset_view(self, event=None):
        self.current_claim = None
        self._clear_annotations()
        self.desc_title.set_text("Select a claim")
        self.desc_text.set_text("")
        for btn, btn_ax in self.buttons:
            btn_ax.set_facecolor('#1a1a3e')
        self.fig.canvas.draw_idle()
        
    def _update(self, i):
        xi, yi = x_traj[i], y_traj[i]
        self.body.set_data([xi], [yi])
        self.body_label.set_position((xi + 0.08, yi + 0.08))
        self.body_label.set_text('3I/ATLAS')
        
        # Current state
        r_vec = np.array([y_anim[0, i], y_anim[1, i]], dtype=float)
        v_vec = np.array([y_anim[2, i], y_anim[3, i]], dtype=float)
        r = float(np.linalg.norm(r_vec))
        t = float(t_anim[i])
        
        # Sunward direction (for anti-tail)
        sunward = -r_vec / r if r > 0 else np.array([1.0, 0.0])
        sunward_au = sunward  # unit vector in AU space
        
        # Calculate wobble angle
        period_s = JET_WOBBLE_PERIOD_H * 3600.0
        wobble_ang = P.wobble_amp_deg * math.sin(2.0 * math.pi * (t / period_s))
        wobble_rad = math.radians(wobble_ang)
        
        # ===== ANTI-TAIL JET VISUAL =====
        jet_length = 0.6  # AU for visualization
        # Main jet direction with wobble
        jet_dir = rotate2(sunward_au, wobble_rad)
        
        # Draw particle streams in the jet cone
        for j, line in enumerate(self.jet_particles):
            # Spread particles across the cone angle
            particle_angle = wobble_rad + (j - len(self.jet_particles)//2) * (self.half_angle / 8)
            p_dir = rotate2(sunward_au, particle_angle)
            # Varying lengths for particle effect
            p_len = jet_length * (0.7 + 0.3 * ((i + j * 10) % 20) / 20)
            end_x = xi + p_dir[0] * p_len
            end_y = yi + p_dir[1] * p_len
            line.set_data([xi, end_x], [yi, end_y])
        
        # Jet label (positioned at end of jet)
        jet_end = np.array([xi, yi]) + jet_dir * (jet_length + 0.1)
        self.jet_label.set_position((jet_end[0], jet_end[1]))
        if self.current_claim == 2:  # Only show label when jet claim is selected
            self.jet_label.set_text(f'ANTI-TAIL\n({JET_SPAN_DEG}° cone)')
        else:
            self.jet_label.set_text('')
        
        # ===== Velocity vector =====
        v_scale = 2.5e-5
        v_dir = v_vec / np.linalg.norm(v_vec)
        v_len = 0.3
        v_end = np.array([xi, yi]) + v_dir * v_len
        self.vel_arrow.set_data([xi, v_end[0]], [yi, v_end[1]])
        
        # ===== Gravity vector (toward sun) =====
        g_dir = -np.array([xi, yi])
        g_norm = np.linalg.norm(g_dir)
        if g_norm > 0:
            g_dir = g_dir / g_norm * 0.2
        g_end = np.array([xi, yi]) + g_dir
        self.grav_arrow.set_data([xi, g_end[0]], [yi, g_end[1]])
        
        # ===== Info text =====
        days = t / 86400.0
        r_au = r / AU
        v_kms = np.linalg.norm(v_vec) / 1000
        
        phase = "INCOMING" if i < perihelion_idx else "OUTGOING"
        
        extra = ""
        if self.current_claim == 3:  # Wobble
            extra = f"\nWobble angle: {wobble_ang:+.1f}°"
        
        self.info.set_text(
            f"═══ 3I/ATLAS ═══\n"
            f"Phase: {phase}\n"
            f"Day: {days:+.0f}\n"
            f"Distance: {r_au:.2f} AU\n"
            f"Speed: {v_kms:.0f} km/s{extra}"
        )
        
        return []
    
    def run(self):
        plt.show(block=True)


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    print("Starting interactive simulation...")
    print("\nKEY FEATURE: The orange streams show the ANTI-TAIL jet")
    print("(pointing TOWARD the Sun, opposite of normal comet tails)\n")
    sim = InteractiveSimulation()
    sim.run()
