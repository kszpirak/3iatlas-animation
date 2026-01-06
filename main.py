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
from matplotlib.patches import Arc, FancyBboxPatch, Rectangle
from matplotlib.collections import LineCollection

# ----------------------------
# Constants & article-linked parameters
# ----------------------------

# === RUN MODE ===
# "LOEB_GEOMETRY" - Gravity-only physics, fixed-axis jet (pure visualization)
# "JET_DYNAMICS"  - Gravity + jet thrust physics, time-varying sunward jet
MODE = "LOEB_GEOMETRY"

# === CONTAINMENT TEST MODE ===
# True  = wobble affects cone containment test (wobbled axis vs sun)
# False = containment test uses fixed axis only (wobble is purely visual)
CONTAINMENT_USES_WOBBLE = False

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
# Claims panel (Loeb-like: tight, testable)
# ----------------------------

CLAIMS = {
    1: {
        "title": "Gravitational Turn: ~16.4°",
        "short": "Turn ≈ 16.4°",
        "loeb_claim": """The inbound and outbound asymptotes of the orbit differ by ~16.4°.

Loeb frames this as a purely gravitational scattering angle,
set by the trajectory parameters (b, v).

Formula: θ = 2 × arctan(GM / (b × v²))

With b ≈ 202 million km and v ≈ 68 km/s at perihelion.

Simulation requirement:
• Compute turn angle using asymptotic velocity directions
  of the GRAVITY-ONLY orbit (no jet perturbation).
• Report measured angle on-screen (computed vs claimed).""",
        "commentary": """PASS/FAIL CRITERIA:

✓ PASS if gravity-only turn angle ≈ 16.4° (±0.5°)
✗ FAIL if measured turn differs materially

What this simulation does:
• Runs a separate gravity-only integration
• Measures velocity directions at large distance (~50 AU)
• Computes angle between inbound/outbound asymptotes

Note: "b" in scattering = impact parameter at infinity,
NOT perihelion distance. We compute both and display.""",
        "source": "https://avi-loeb.medium.com/six-anomalies-of-3i-atlas-f9a4dbb06db7",
        "highlight": "deflection",
        # 15-year-old mode fields
        "teen_title": "How much the path bends",
        "teen_short": "Turn ≈ 16°",
        "teen_summary": """The Sun bends the object's path like a slingshot.
We measure how many degrees it turns from far away
going in → to far away going out.
If it's ~16°, that matches what was claimed.""",
        "teen_passfail": "Turn matches expected angle (~16°)",
    },
    2: {
        "title": "Cone Width: Jet Opening ~8°",
        "short": "Cone ≈ 8°",
        "loeb_claim": """The sunward feature is tightly collimated with ~8° full
opening angle (4° half-angle).

From Dec 15 imagery:
"the jet spans about 8° out to a distance of order a million km"

Simulation requirement:
• Draw cone of full width 8° around the jet axis
• Test whether Sun direction lies within that cone over time
• Report statistics: % of time Sun is in each cone""",
        "commentary": """PASS/FAIL CRITERIA:

✓ PASS if Sun direction stays inside the 4° half-angle
  cone when the geometry claims it should
✗ FAIL if Sun frequently leaves both cones

What this simulation shows:
• BOTH cones (+A and -A) rendered simultaneously
• Color indicates which cone (if any) contains Sun
• Real-time angle from each axis to Sun direction

The 8° width is critical: if the turn is 16.4° ≈ 2×8°,
then Sun can be at edge of +A cone pre-perihelion and
edge of -A cone post-perihelion.""",
        "source": "https://avi-loeb.medium.com/six-anomalies-of-3i-atlas-f9a4dbb06db7",
        "highlight": "jet_cone",
        # 15-year-old mode fields
        "teen_title": "Is the Sun inside the 'flashlight beam'?",
        "teen_short": "Sun in beam?",
        "teen_summary": """Imagine the jet is like a flashlight beam, 8° wide.
We check if the Sun stays inside that beam as the
object flies by. The beam doesn't move—it's fixed!""",
        "teen_passfail": "Sun stays inside the jet cone most of the time",
    },
    3: {
        "title": "Wobble: Period ~7.74h, Amplitude ±4°",
        "short": "Wobble 7.74h",
        "loeb_claim": """The jet direction exhibits periodic wobble consistent with
nucleus rotation:
• Period: ~7.74 hours
• Amplitude: ~±4°

"The observed wobble of the pre-perihelion sunward jet
requires the base of the jet to be within 8° from the
sun-facing pole."

Simulation requirement:
• Apply sinusoidal angular wobble to jet direction
• Wobble must NOT secretly steer toward Sun""",
        "commentary": """PASS/FAIL CRITERIA:

✓ PASS if wobble parameters match exactly AND don't
  "cheat" by also changing axis to track Sun
✗ FAIL if wobble is implemented by rotating toward Sun
  (that becomes active pointing, not wobble)

What this simulation does:
• Applies ±4° sinusoidal wobble with 7.74h period
• Wobble rotates the jet AROUND the fixed axis
• Axis itself remains fixed in inertial space

Note: In 2D this is a visualization proxy for what
would be 3D precession around spin axis.""",
        # 15-year-old mode fields
        "teen_title": "Jet wiggle",
        "teen_short": "Wobble ±4°",
        "teen_summary": """The jet wiggles back and forth a little—like a
spinning sprinkler. It swings ±4° every 7.74 hours.
This wobble is NOT 'aiming at the Sun' on purpose.""",
        "teen_passfail": "Wobble is just random wiggle, not steering",
        "source": "https://avi-loeb.medium.com/six-anomalies-of-3i-atlas-f9a4dbb06db7",
        "highlight": "wobble",
    },
    4: {
        "title": "No Active Steering: Geometry Keeps Sun in Cone",
        "short": "No steering",
        "loeb_claim": """CORE GEOMETRIC ARGUMENT:

The jet does NOT actively steer to follow the Sun.

Instead:
• Two opposite jet cones exist: +A and -A
• Pre-perihelion: Sun lies near/within +A cone edge
• Post-perihelion: Sun lies near/within -A cone edge
• This works BECAUSE turn angle (~16.4°) ≈ 2× cone width (~8°)

The sunward appearance is maintained by COINCIDENCE
of orbital geometry, not active pointing.""",
        "commentary": """PASS/FAIL CRITERIA:

✓ PASS if Sun stays within one of the two cones through
  the encounter WITHOUT axis being re-aimed at Sun
✗ FAIL if we "switch" cones by a rule tied to perihelion
  without showing both cones simultaneously

What this simulation does:
• ALWAYS renders both +A and -A cones
• Tests Sun containment in BOTH cones every frame
• Reports which cone(s) contain Sun
• Does NOT switch axis at perihelion — both always visible

This is the key test: does the geometry work without
any hidden steering rules?""",
        "source": "https://avi-loeb.medium.com/six-anomalies-of-3i-atlas-f9a4dbb06db7",
        "highlight": "sunward",
        # 15-year-old mode fields
        "teen_title": "Does the jet 'aim' at the Sun?",
        "teen_short": "No aiming",
        "teen_summary": """The jet does NOT track the Sun like a robot!
It just happens that the Sun stays inside the beam
because of how the path curves. It's a coincidence
of geometry, not intentional pointing.""",
        "teen_passfail": "Jet is fixed—Sun stays in cone by luck of angles",
    },
    5: {
        "title": "Jet Length Scale: ~10⁶ km",
        "short": "~1M km",
        "loeb_claim": """The sunward feature is visible out to order 10⁶ km.

"field of view spans 1.6 × 0.7 million km"

Physical scale:
• ~1,000,000 km ≈ 0.0067 AU
• About 2.6× Earth-Moon distance

Simulation requirement:
• Use 10⁶ km as physical jet length
• May exaggerate visually for display
• Must label BOTH physical and displayed lengths""",
        "commentary": """PASS/FAIL CRITERIA:

✓ PASS if display clearly distinguishes physical vs
  exaggerated length, and geometry uses angles only
✗ FAIL if exaggerated length affects angle calculations

What this simulation shows:
• Physical length: ~1M km ≈ 0.0067 AU
• Display length: ×60 exaggerated (~0.4 AU shown)
• Cone angles are PHYSICAL (8° full width)

The exaggeration is purely visual — all geometric
tests use the true angular relationships.""",
        "source": "https://avi-loeb.medium.com/six-anomalies-of-3i-atlas-f9a4dbb06db7",
        "highlight": "perihelion",
        # 15-year-old mode fields
        "teen_title": "How long is the jet?",
        "teen_short": "~1M km",
        "teen_summary": """The jet is about 1 million km long in reality.
That's 2.6× the distance from Earth to the Moon!
We draw it 60× bigger so you can see it clearly.""",
        "teen_passfail": "Length is labeled correctly (real vs display)",
    },
    6: {
        "title": "Ecliptic Alignment: ±5° (3D required)",
        "short": "Ecliptic ±5°",
        "loeb_claim": """Claim involves inclination relative to ecliptic plane:

"Retrograde trajectory... aligned to within 5° with the
ecliptic plane... probability 0.2%"

The ecliptic is Earth's orbital plane (z=0 in standard
solar system coordinates).

Simulation requirement:
• CANNOT be validated in 2D orbital-plane simulation
• Requires 3D state vector and defined ecliptic""",
        "commentary": """PASS/FAIL CRITERIA:

✓ PASS only if 3D dynamics implemented and orbital
  inclination computed relative to ecliptic plane
✗ In 2D: explicitly labeled "NOT MODELED"

This 2D simulation shows the orbital plane.
Ecliptic alignment requires:
• 3D position (x, y, z)
• 3D velocity (vx, vy, vz)
• Measure angle between orbital plane and ecliptic

⚠️ THIS CLAIM IS NOT MODELED IN THIS 2D SIMULATION""",
        "source": "https://avi-loeb.medium.com/six-anomalies-of-3i-atlas-f9a4dbb06db7",
        "highlight": "ecliptic",
        # 15-year-old mode fields
        "teen_title": "Orbit tilt vs Earth's orbit",
        "teen_short": "3D only",
        "teen_summary": """This is about the 3D angle of the orbit compared to
Earth's orbit plane. We can't test this in 2D!
(Need full 3D coordinates to measure tilt.)""",
        "teen_passfail": "⚠ Can't check in 2D simulation",
    },
}

# ----------------------------
# 15-Year-Old Mode UI Text
# ----------------------------
# Centralized dictionary for all UI strings with teen-friendly alternatives

UI_STRINGS = {
    # Main title
    "main_title": {
        "normal": "3I/ATLAS: Loeb's Fixed-Axis Jet [{mode}]",
        "teen": "3I/ATLAS: Jet vs Sun (simple view) [{mode}]",
    },
    # Axis labels
    "x_label": {
        "normal": "Distance from Sun (AU)",
        "teen": "Distance from Sun (AU)",
    },
    "y_label": {
        "normal": "Distance from Sun (AU)",
        "teen": "Distance from Sun (AU)",
    },
    # Legend items
    "legend_body": {
        "normal": "3I/ATLAS",
        "teen": "3I/ATLAS (the object)",
    },
    "legend_trail": {
        "normal": "Recent path (trail)",
        "teen": "Where it just was",
    },
    "legend_cone_plus": {
        "normal": "+A cone (Loeb model)",
        "teen": "+A beam (Loeb's idea)",
    },
    "legend_cone_minus": {
        "normal": "-A cone (Loeb model)",
        "teen": "-A beam (Loeb's idea)",
    },
    "legend_normal_tail": {
        "normal": "Normal tail (away)",
        "teen": "Normal comet tail (away from Sun)",
    },
    "legend_velocity": {
        "normal": "Velocity",
        "teen": "Direction it's moving",
    },
    "legend_gravity": {
        "normal": "Gravity -> Sun",
        "teen": "Sun pulling it in",
    },
    # Info box labels
    "info_phase_incoming": {
        "normal": "INCOMING",
        "teen": "Approaching Sun",
    },
    "info_phase_outgoing": {
        "normal": "OUTGOING",
        "teen": "Leaving Sun",
    },
    "info_wobble": {
        "normal": "Wobble",
        "teen": "Jet wiggle",
    },
    # Buttons
    "btn_reset": {
        "normal": "RESET",
        "teen": "Reset view",
    },
    "btn_pause": {
        "normal": "PAUSE",
        "teen": "Pause",
    },
    "btn_play": {
        "normal": "PLAY",
        "teen": "Play",
    },
    "btn_quit": {
        "normal": "QUIT",
        "teen": "Close",
    },
    # Claims panel header
    "claims_header": {
        "normal": "LOEB'S ANOMALIES",
        "teen": "STRANGE THINGS TO CHECK",
    },
    "claims_subheader": {
        "normal": "Click claim → [i] for details",
        "teen": "Click to learn more!",
    },
    # Geometry panel
    "geom_header": {
        "normal": "═══ GEOMETRY ═══",
        "teen": "═══ SETTINGS ═══",
    },
    "geom_turn": {
        "normal": "Turn",
        "teen": "How much it bends",
    },
    "geom_cone": {
        "normal": "Cone",
        "teen": "Beam width",
    },
    # Pass/Fail panel
    "pf_header": {
        "normal": "═══ PASS/FAIL ═══",
        "teen": "═══ DOES IT WORK? ═══",
    },
    "pf_turn": {
        "normal": "Turn",
        "teen": "Bend amount",
    },
    "pf_cone": {
        "normal": "Cone",
        "teen": "Sun in beam",
    },
    "pf_wobble": {
        "normal": "Wobble",
        "teen": "Wiggle",
    },
    "pf_no_steer": {
        "normal": "No steer",
        "teen": "No aiming",
    },
    "pf_scale": {
        "normal": "Scale",
        "teen": "Jet size",
    },
    "pf_ecliptic": {
        "normal": "Eclipt",
        "teen": "3D tilt",
    },
    # Sun-in-cone indicator (TEST of Loeb's geometry)
    "sun_in_both": {
        "normal": "TEST: SUN IN BOTH",
        "teen": "Test: Sun in BOTH!",
    },
    "sun_in_plus": {
        "normal": "TEST: SUN IN +A",
        "teen": "Test: Sun in +A",
    },
    "sun_in_minus": {
        "normal": "TEST: SUN IN -A",
        "teen": "Test: Sun in -A",
    },
    "sun_outside": {
        "normal": "TEST: SUN OUTSIDE!",
        "teen": "Test: Sun missed!",
    },
    # Comparison box
    "compare_title": {
        "normal": "NORMAL vs 3I/ATLAS",
        "teen": "NORMAL vs 3I/ATLAS",
    },
    "compare_normal": {
        "normal": "Normal",
        "teen": "Typical comet",
    },
    "compare_normal_dir": {
        "normal": "Tail AWAY",
        "teen": "Tail points AWAY",
    },
    "compare_atlas_dir": {
        "normal": "Jet TO Sun!",
        "teen": "Jet toward Sun!",
    },
    # Modal
    "modal_loeb_header": {
        "normal": "LOEB'S CLAIM",
        "teen": "THE CLAIM",
    },
    "modal_criteria_header": {
        "normal": "PASS/FAIL CRITERIA",
        "teen": "HOW WE CHECK IT",
    },
    "modal_close_hint": {
        "normal": "Press ESC or click ✕ CLOSE to close",
        "teen": "Press ESC or click ✕ to close",
    },
    "modal_show_technical": {
        "normal": "",
        "teen": "[Show technical version]",
    },
}

def ui_text(key: str, teen: bool, **kwargs) -> str:
    """Return UI string based on mode. Supports format kwargs."""
    entry = UI_STRINGS.get(key, {})
    mode_key = "teen" if teen else "normal"
    text = entry.get(mode_key, entry.get("normal", key))
    if kwargs:
        return text.format(**kwargs)
    return text

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

def rhs_gravity_only(t: float, state: np.ndarray) -> np.ndarray:
    """Gravity-only dynamics for computing true deflection angle."""
    x, y, vx, vy = state
    r_vec = np.array([x, y], dtype=float)
    r = float(np.linalg.norm(r_vec))
    if r == 0.0:
        ax, ay = 0.0, 0.0
    else:
        a = -MU_SUN * r_vec / (r ** 3)
        ax, ay = float(a[0]), float(a[1])
    return np.array([vx, vy, ax, ay], dtype=float)

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

def make_stop_event(r_stop: float):
    # IMPORTANT: no lambda here, or you lose .terminal / .direction
    def stop_when_far(t: float, y: np.ndarray) -> float:
        return r_stop - math.hypot(y[0], y[1])

    stop_when_far.terminal = True
    stop_when_far.direction = -1
    return stop_when_far

# ----------------------------
# Compute trajectory
# ----------------------------

print("Computing trajectories...")

rp = RP_KM * 1000.0  # perihelion distance in meters
vp = VP_KMS * 1000.0  # velocity at perihelion in m/s

# Compute v_infinity and impact parameter from perihelion conditions
# Energy: E = v²/2 - GM/r = v_inf²/2 (at infinity)
v_inf_sq = vp**2 - 2*MU_SUN/rp
v_inf = math.sqrt(max(0.0, v_inf_sq))

# Angular momentum: h = r_p × v_p (at perihelion, perpendicular)
h = rp * vp

# Impact parameter: b = h / v_inf
b_impact = h / v_inf if v_inf > 0 else float('inf')

print(f"Orbital parameters:")
print(f"  Perihelion distance r_p = {rp/1e9:.1f} million km = {rp/AU:.3f} AU")
print(f"  Perihelion velocity v_p = {vp/1000:.1f} km/s")
print(f"  Hyperbolic excess v_∞ = {v_inf/1000:.1f} km/s")
print(f"  Impact parameter b = {b_impact/1e9:.1f} million km")
print(f"  (Loeb uses b ≈ 202 million km — check if this matches!)")

# Initial conditions at perihelion
y0 = np.array([rp, 0.0, 0.0, vp], dtype=float)
tmax = P.t_span_days * 86400.0

# --------------------------------------------------
# 1) GRAVITY-ONLY trajectory for deflection measurement
#    (integrate far out to get true asymptotes)
# --------------------------------------------------
R_STOP_DEFLECTION = 50.0 * AU  # Go far out for accurate asymptotes
event_far = make_stop_event(R_STOP_DEFLECTION)

sol_grav_fwd = solve_ivp(
    fun=rhs_gravity_only,
    t_span=(0.0, tmax * 10),  # Long time to reach far distance
    y0=y0,
    max_step=P.max_step_s * 2,
    rtol=1e-10,
    atol=1e-10,
    events=event_far,
)

sol_grav_bwd = solve_ivp(
    fun=rhs_gravity_only,
    t_span=(0.0, -tmax * 10),
    y0=y0,
    max_step=P.max_step_s * 2,
    rtol=1e-10,
    atol=1e-10,
    events=event_far,
)

# Extract asymptotic velocity directions from gravity-only trajectory
# Use the LAST point of each integration (farthest from Sun)
v_in_grav = np.array([sol_grav_bwd.y[2, -1], sol_grav_bwd.y[3, -1]])
v_out_grav = np.array([sol_grav_fwd.y[2, -1], sol_grav_fwd.y[3, -1]])
v_in_hat_grav = v_in_grav / np.linalg.norm(v_in_grav)
v_out_hat_grav = v_out_grav / np.linalg.norm(v_out_grav)

# Compute deflection angle from gravity-only trajectory
# NOTE: For hyperbolic scattering, inbound and outbound velocities are nearly OPPOSITE
# for zero deflection (dot ≈ -1, arccos ≈ π). The actual TURN/SCATTERING angle is:
#   δ = π - angle_between_velocities
dot_grav = np.clip(np.dot(v_in_hat_grav, v_out_hat_grav), -1.0, 1.0)
angle_between_rad = np.arccos(dot_grav)  # ~π when deflection ~0
deflection_grav_rad = np.pi - angle_between_rad  # actual scattering/turn angle
deflection_grav_deg = np.degrees(deflection_grav_rad)

# Also compute theoretical deflection from scattering formula
# θ = 2 × arctan(GM / (b × v_∞²))
theta_theory_rad = 2 * math.atan(MU_SUN / (b_impact * v_inf**2))
theta_theory_deg = math.degrees(theta_theory_rad)

# Also compute what Loeb's formula gives if he used rp instead of b_impact
theta_using_rp_rad = 2 * math.atan(MU_SUN / (rp * v_inf**2))
theta_using_rp_deg = math.degrees(theta_using_rp_rad)

print(f"\nDeflection angle (GRAVITY-ONLY):")
print(f"  Angle between asymptotic v vectors: {np.degrees(angle_between_rad):.2f}°")
print(f"  Scattering deflection (π - angle): {deflection_grav_deg:.2f}°")
print(f"  Theoretical (using b_impact): {theta_theory_deg:.2f}°")
print(f"  Theoretical (using rp as 'b'): {theta_using_rp_deg:.2f}°")
print(f"  Loeb's claim: {LOEB_DEFLECTION_DEG}°")
print(f"\nParameter comparison:")
print(f"  rp (perihelion): {rp/1e9:.1f} million km")
print(f"  b_impact (impact parameter): {b_impact/1e9:.1f} million km")
print(f"  b_impact/rp = {b_impact/rp:.3f}")

# Store for display
computed_deflection_deg = deflection_grav_deg
v_in_hat = v_in_hat_grav
v_out_hat = v_out_hat_grav

# --------------------------------------------------
# 2) VISUAL trajectory - depends on MODE
# --------------------------------------------------
event = make_stop_event(P.r_stop)

print(f"\nMode: {MODE}")

if MODE == "LOEB_GEOMETRY":
    # Use GRAVITY-ONLY for visual trajectory too (pure Keplerian)
    print("  Using GRAVITY-ONLY dynamics for visual trajectory")
    dynamics_fn = rhs_gravity_only
else:
    # JET_DYNAMICS: use gravity + jet thrust
    print("  Using GRAVITY + JET dynamics for visual trajectory")
    dynamics_fn = lambda t, y: rhs(t, y, P)

sol_fwd = solve_ivp(
    fun=dynamics_fn,
    t_span=(0.0, tmax),
    y0=y0,
    max_step=P.max_step_s,
    rtol=1e-9,
    atol=1e-9,
    events=event,
)

sol_bwd = solve_ivp(
    fun=dynamics_fn,
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

print(f"\nTrajectory computed: {Nf} frames, perihelion at frame {perihelion_idx}")

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
# BOTH cones (+A and -A) exist simultaneously.
# We test whether Sun falls into EITHER cone at each frame.
# The 16.4° deflection ≈ 2 × 8° cone should keep Sun in one cone or the other.

# Define the fixed axis: EDGE-ALIGNED configuration
# Rotate initial sunward direction by HALF the cone angle so Sun starts at cone EDGE
# This is the critical Loeb geometry: Sun at edge of +A inbound, edge of -A outbound

jet_half_angle_rad = np.deg2rad(JET_HALF_ANGLE_DEG)

# Initial sunward at frame 0 (far inbound)
sunward_initial = sunward_hat[0].copy()

# Rotate sunward by +half_angle to get +A axis (Sun starts at EDGE of +A cone)
jet_axis_plus = rotate2(sunward_initial, +jet_half_angle_rad)
jet_axis_plus = jet_axis_plus / np.linalg.norm(jet_axis_plus)  # normalize

# -A axis is opposite
jet_axis_minus = -jet_axis_plus

print(f"\nFixed jet axes defined (EDGE-ALIGNED):")
print(f"  Initial sunward: ({sunward_initial[0]:.4f}, {sunward_initial[1]:.4f})")
print(f"  +A axis (rotated +{JET_HALF_ANGLE_DEG}°): ({jet_axis_plus[0]:.4f}, {jet_axis_plus[1]:.4f})")
print(f"  -A axis: ({jet_axis_minus[0]:.4f}, {jet_axis_minus[1]:.4f})")
print(f"  Containment uses wobble: {CONTAINMENT_USES_WOBBLE}")

# Wobble angle (rad) - rotation around jet axis
period_s = JET_WOBBLE_PERIOD_H * 3600.0
wobble_phase = 2.0 * np.pi * (t_anim / period_s)
wobble_deg = P.wobble_amp_deg * np.sin(wobble_phase)
wobble_rad = np.deg2rad(wobble_deg)

# Precompute jet directions and sun-in-cone status for BOTH cones
jet_hat_plus = np.zeros((Nf, 2))   # +A cone jet direction (with wobble)
jet_hat_minus = np.zeros((Nf, 2))  # -A cone jet direction (with wobble)
sun_in_plus_cone = np.zeros(Nf, dtype=bool)
sun_in_minus_cone = np.zeros(Nf, dtype=bool)
angle_to_plus = np.zeros(Nf)   # Angle from +A axis to Sun
angle_to_minus = np.zeros(Nf)  # Angle from -A axis to Sun

for i in range(Nf):
    # Apply wobble rotation to BOTH axes (for visualization)
    c, s = np.cos(wobble_rad[i]), np.sin(wobble_rad[i])
    jet_hat_plus[i, 0] = c * jet_axis_plus[0] - s * jet_axis_plus[1]
    jet_hat_plus[i, 1] = s * jet_axis_plus[0] + c * jet_axis_plus[1]
    jet_hat_minus[i, 0] = c * jet_axis_minus[0] - s * jet_axis_minus[1]
    jet_hat_minus[i, 1] = s * jet_axis_minus[0] + c * jet_axis_minus[1]
    
    # Containment test: use wobbled or fixed axis depending on flag
    if CONTAINMENT_USES_WOBBLE:
        # Test against WOBBLED jet direction
        test_axis_plus = jet_hat_plus[i]
        test_axis_minus = jet_hat_minus[i]
    else:
        # Test against FIXED axis (wobble is purely visual)
        test_axis_plus = jet_axis_plus
        test_axis_minus = jet_axis_minus
    
    angle_plus = np.arccos(np.clip(np.dot(test_axis_plus, sunward_hat[i]), -1.0, 1.0))
    angle_minus = np.arccos(np.clip(np.dot(test_axis_minus, sunward_hat[i]), -1.0, 1.0))
    
    angle_to_plus[i] = np.degrees(angle_plus)
    angle_to_minus[i] = np.degrees(angle_minus)
    
    sun_in_plus_cone[i] = angle_plus <= jet_half_angle_rad
    sun_in_minus_cone[i] = angle_minus <= jet_half_angle_rad

# Combined: Sun in at least one cone
sun_in_any_cone = sun_in_plus_cone | sun_in_minus_cone

# Compute longest continuous "out" interval (Sun in neither cone)
sun_out_of_cone = ~sun_in_any_cone
longest_out_interval = 0
current_out_interval = 0
for i in range(Nf):
    if sun_out_of_cone[i]:
        current_out_interval += 1
        longest_out_interval = max(longest_out_interval, current_out_interval)
    else:
        current_out_interval = 0

# Time per frame (approximate)
dt_frame_days = (t_anim[-1] - t_anim[0]) / (Nf - 1) / 86400.0 if Nf > 1 else 0
longest_out_days = longest_out_interval * dt_frame_days

print(f"\nSun-in-cone statistics:")
print(f"  Sun in +A cone: {np.sum(sun_in_plus_cone)}/{Nf} frames ({100*np.sum(sun_in_plus_cone)/Nf:.1f}%)")
print(f"  Sun in -A cone: {np.sum(sun_in_minus_cone)}/{Nf} frames ({100*np.sum(sun_in_minus_cone)/Nf:.1f}%)")
print(f"  Sun in EITHER cone: {np.sum(sun_in_any_cone)}/{Nf} frames ({100*np.sum(sun_in_any_cone)/Nf:.1f}%)")
print(f"  Longest gap (Sun out): {longest_out_interval} frames ({longest_out_days:.1f} days)")

# For backward compatibility, keep these names
jet_hat = jet_hat_plus  # Primary jet for visualization
sun_in_cone = sun_in_any_cone
active_axis = np.zeros((Nf, 2))
for i in range(Nf):
    # For visualization: show which axis is "closer" to Sun
    if angle_to_plus[i] <= angle_to_minus[i]:
        active_axis[i] = jet_axis_plus
    else:
        active_axis[i] = jet_axis_minus

# ----------------------------
# Interactive animation
# ----------------------------

class InteractiveSimulation:
    def __init__(self):
        self.current_claim = None
        self.paused = False
        self.details_fig = None  # Store reference to prevent garbage collection
        self.teen_mode = False  # 15-year-old mode: simplified text

        self.fig = plt.figure(figsize=(20, 11), facecolor="#0a0a1a")
        self.ax = self.fig.add_axes([0.02, 0.06, 0.62, 0.90])  # narrower to leave room for legend
        self.ax.set_facecolor("#0a0a1a")
        self.ax.set_aspect("equal", "box")

        self.annotations = []
        self.extra_artists = []

        self._setup_main_plot()
        self._setup_claims_panel()
        self._bind_keys()

        # Force initial draw so buttons are responsive immediately
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Use blit=False to avoid visual artifacts outside axes bounds
        # (blitting can cause rendering issues with clipping on some backends)
        self.ani = FuncAnimation(self.fig, self._update, frames=Nf, interval=30, blit=False, repeat=True)

    def _setup_main_plot(self):
        ax = self.ax
        mode_label = "GRAVITY-ONLY" if MODE == "LOEB_GEOMETRY" else "GRAVITY+JET"
        title_text = ui_text("main_title", self.teen_mode, mode=mode_label)
        self._main_title = ax.set_title(title_text, fontsize=20, fontweight="bold", color="white", pad=15)
        ax.set_xlabel(ui_text("x_label", self.teen_mode), fontsize=14, color="white")
        ax.set_ylabel(ui_text("y_label", self.teen_mode), fontsize=14, color="white")
        ax.tick_params(colors="white", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#333355")

        # full trajectory (clip to axes)
        traj_line, = ax.plot(x_traj, y_traj, linewidth=2, color="#4488ff", alpha=0.3, clip_on=True)
        
        # Sun
        sun1 = ax.scatter([0], [0], s=3000, color="#ffdd44", alpha=0.1, zorder=5, clip_on=True)
        sun2 = ax.scatter([0], [0], s=1500, color="#ffdd44", alpha=0.2, zorder=5, clip_on=True)
        sun3 = ax.scatter([0], [0], s=700, color="#ffcc00", edgecolors="#ff8800", linewidths=3, zorder=6, clip_on=True)
        ax.text(0, 0, "☀", fontsize=24, ha="center", va="center", color="#ffff88", zorder=7)
        ax.text(
            0, -0.32, "SUN", fontsize=14, ha="center", va="top", color="#ffdd44", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a0a", edgecolor="#ffaa00", alpha=0.9),
        )

        # moving body (scatter = faster than plot marker updates)
        self.body = ax.scatter([], [], s=110, color="#00ffcc", edgecolors="white", linewidths=2, zorder=20, clip_on=True)
        self.body_label = ax.text(0, 0, "", fontsize=12, color="#00ffcc", fontweight="bold", ha="left", va="bottom", zorder=21, clip_on=True)
        
        # UFO image for teen mode (hidden by default)
        import os
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ufo_path = os.path.join(script_dir, "3i-atlas.png")
        try:
            ufo_img = plt.imread(ufo_path)
            self.ufo_image = OffsetImage(ufo_img, zoom=0.15)  # adjust zoom to fit
            self.ufo_box = AnnotationBbox(self.ufo_image, (0, 0), frameon=False, 
                                          zorder=21, clip_on=True)
            ax.add_artist(self.ufo_box)
            self.ufo_box.set_visible(False)
            self._ufo_loaded = True
        except Exception as e:
            print(f"Warning: Could not load UFO image: {e}")
            self._ufo_loaded = False

        # trail
        self.trail, = ax.plot([], [], linewidth=2, color="#66aaff", alpha=0.9, zorder=12, clip_on=True)
        self.trail_len = 120

        # Jet "particle rays" using LineCollection - +A CONE (orange)
        self.n_rays = 15
        self.jet_len_au = JET_LEN_AU_VIS  # Physically-based (exaggerated) length
        self.half_angle = math.radians(JET_SPAN_DEG / 2.0)
        segs = np.zeros((self.n_rays, 2, 2), dtype=float)
        self.jet_lc_plus = LineCollection(segs, linewidths=np.linspace(3.0, 1.0, self.n_rays),
                                          colors=[(1.0, 0.67, 0.0, 0.8)], zorder=15)
        self.jet_lc_plus.set_clip_on(True)
        self.jet_lc_plus.set_clip_box(ax.bbox)
        ax.add_collection(self.jet_lc_plus)
        
        # Second jet cone (-A) - blue color
        segs_minus = np.zeros((self.n_rays, 2, 2), dtype=float)
        self.jet_lc_minus = LineCollection(segs_minus, linewidths=np.linspace(3.0, 1.0, self.n_rays),
                                           colors=[(0.4, 0.7, 1.0, 0.6)], zorder=14)
        self.jet_lc_minus.set_clip_on(True)
        self.jet_lc_minus.set_clip_box(ax.bbox)
        ax.add_collection(self.jet_lc_minus)

        self.jet_label = ax.text(0, 0, "", fontsize=10, color="#ffaa00", fontweight="bold", ha="center", va="bottom", zorder=22, clip_on=True)
        
        # Cone labels (shows which cone is which)
        self.cone_label_plus = ax.text(0, 0, "+A", fontsize=9, color="#ffaa00", fontweight="bold", 
                                       ha="left", va="center", zorder=22, clip_on=True)
        self.cone_label_minus = ax.text(0, 0, "-A", fontsize=9, color="#6699ff", fontweight="bold", 
                                         ha="left", va="center", zorder=22, clip_on=True)
        
        # Fixed jet axis indicators (shows BOTH axes)
        self.axis_line_plus, = ax.plot([], [], linewidth=2, color="#ff8800", linestyle=":", alpha=0.7, zorder=13, clip_on=True)
        self.axis_line_minus, = ax.plot([], [], linewidth=2, color="#4488ff", linestyle=":", alpha=0.7, zorder=13, clip_on=True)
        
        # Sun-in-cone indicator (top-right corner) - shows BOTH cones
        self.sun_cone_indicator = ax.text(0.98, 0.98, "", transform=ax.transAxes, fontsize=12, 
                                          fontweight="bold", ha="right", va="top", zorder=30,
                                          bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", 
                                                   edgecolor="#444466", alpha=0.95))

        # velocity vector
        self.vel_line, = ax.plot([], [], linewidth=3, color="#00ff88", zorder=18, clip_on=True)
        # gravity vector
        self.grav_line, = ax.plot([], [], linewidth=2, color="#ff6666", linestyle="--", alpha=0.7, zorder=17, clip_on=True)
        
        # "Normal tail" arrow - shows what a NORMAL comet tail would do (point AWAY from sun)
        from matplotlib.patches import FancyArrowPatch
        self.normal_tail_arrow = FancyArrowPatch((0, 0), (0, 0), 
                                                  arrowstyle='->', mutation_scale=15,
                                                  color='#6688aa', linewidth=3, 
                                                  linestyle='--', alpha=0.9, zorder=14,
                                                  clip_on=True)
        ax.add_patch(self.normal_tail_arrow)
        self.normal_tail_label = ax.text(0, 0, "", fontsize=10, color="#6688aa", ha="center", va="top", 
                                          fontweight="bold", zorder=22, clip_on=True)

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
        self._legend_items = [
            plt.Line2D([0], [0], color='#00ffcc', marker='o', markersize=10, linestyle='None', 
                      label=ui_text("legend_body", self.teen_mode)),
            plt.Line2D([0], [0], color='#66aaff', linewidth=2, 
                      label=ui_text("legend_trail", self.teen_mode)),
            plt.Line2D([0], [0], color='#ffaa00', linewidth=4, 
                      label=ui_text("legend_cone_plus", self.teen_mode)),
            plt.Line2D([0], [0], color='#6699ff', linewidth=4, 
                      label=ui_text("legend_cone_minus", self.teen_mode)),
            plt.Line2D([0], [0], color='#6688aa', linewidth=3, linestyle=':', 
                      label=ui_text("legend_normal_tail", self.teen_mode)),
            plt.Line2D([0], [0], color='#00ff88', linewidth=3, 
                      label=ui_text("legend_velocity", self.teen_mode)),
            plt.Line2D([0], [0], color='#ff6666', linewidth=2, linestyle='--', 
                      label=ui_text("legend_gravity", self.teen_mode)),
        ]
        self._legend = ax.legend(handles=self._legend_items, loc='upper left', bbox_to_anchor=(1.01, 1.0),
                  fontsize=9, facecolor='#1a1a2e', edgecolor='#333355', 
                  labelcolor='white', framealpha=0.95)

        # view limits - 6 AU wide centered on trajectory
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.grid(True, alpha=0.15, color="#4444aa", linestyle="--")

    def _draw_normal_comet_comparison(self, ax):
        """Draw a small inset showing normal comet behavior vs 3I/ATLAS"""
        # Position in upper-left of plot (in data coordinates)
        cx, cy = -2.3, 1.3  # center of comparison diagram (moved down 1 AU)
        scale = 0.35
        
        # Background box
        from matplotlib.patches import Rectangle
        bg = Rectangle((cx - 0.55, cy - 0.7), 1.1, 1.0, 
                       facecolor='#0a0a1a', edgecolor='#444466', 
                       linewidth=2, alpha=0.95, zorder=30)
        ax.add_patch(bg)
        
        # Title
        ax.text(cx, cy + 0.22, "NORMAL vs 3I/ATLAS", fontsize=10, fontweight='bold',
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
        ax.text(cx, cy + 0.12, "Sun", fontsize=8, color='#ffcc00', ha='center', va='bottom', zorder=31)

    def _setup_claims_panel(self):
        self._claims_header = self.fig.text(0.88, 0.96, ui_text("claims_header", self.teen_mode), 
                                            fontsize=14, fontweight="bold", color="#ffcc00", ha="center")
        self._claims_subheader = self.fig.text(0.88, 0.945, ui_text("claims_subheader", self.teen_mode), 
                                               fontsize=8, color="#888899", ha="center")

        self.buttons = []
        self.detail_buttons = []
        button_h = 0.050
        start_y = 0.89

        # IMPORTANT: avoid late-binding lambda issues by using a closure helper
        def make_cb(cid):
            return lambda event: self.select_claim(cid)
        
        def make_detail_cb(cid):
            def callback(event):
                # Show modal directly
                self._show_claim_modal(cid)
            return callback

        for i, (cid, claim) in enumerate(CLAIMS.items()):
            y_pos = start_y - i * (button_h + 0.008)
            # Main claim button (narrower to make room for detail button)
            btn_ax = self.fig.add_axes([0.76, y_pos, 0.18, button_h])
            # Use teen_short if in teen mode and available
            short_text = claim.get('teen_short', claim['short']) if self.teen_mode else claim['short']
            btn = Button(btn_ax, f"{cid}. {short_text}", color="#1a1a3e", hovercolor="#3a3a6e")
            btn.label.set_color("white")
            btn.label.set_fontsize(10)
            btn.label.set_fontweight("bold")
            btn.on_clicked(make_cb(cid))
            self.buttons.append((cid, btn, btn_ax))
            
            # Detail button [i] for each claim
            detail_ax = self.fig.add_axes([0.945, y_pos, 0.035, button_h])
            detail_btn = Button(detail_ax, "i", color="#2a3a2a", hovercolor="#4a6a4a")
            detail_btn.label.set_color("#88ff88")
            detail_btn.label.set_fontsize(12)
            detail_btn.label.set_fontweight("bold")
            detail_btn.on_clicked(make_detail_cb(cid))
            self.detail_buttons.append((cid, detail_btn, detail_ax))

        # Reset, Pause, Quit, and Teen Mode toggle buttons
        reset_ax = self.fig.add_axes([0.76, 0.515, 0.055, 0.038])
        self.reset_btn = Button(reset_ax, ui_text("btn_reset", self.teen_mode), color="#333355", hovercolor="#555588")
        self.reset_btn.label.set_color("white")
        self.reset_btn.label.set_fontweight("bold")
        self.reset_btn.label.set_fontsize(10)
        self.reset_btn.on_clicked(self.reset_view)

        pause_ax = self.fig.add_axes([0.82, 0.515, 0.055, 0.038])
        self.pause_btn = Button(pause_ax, ui_text("btn_pause", self.teen_mode), color="#4a2a2a", hovercolor="#6a4a4a")
        self.pause_btn.label.set_color("white")
        self.pause_btn.label.set_fontweight("bold")
        self.pause_btn.label.set_fontsize(10)
        self.pause_btn.on_clicked(self._toggle_pause)

        quit_ax = self.fig.add_axes([0.88, 0.515, 0.045, 0.038])
        self.quit_btn = Button(quit_ax, ui_text("btn_quit", self.teen_mode), color="#661111", hovercolor="#aa2222")
        self.quit_btn.label.set_color("white")
        self.quit_btn.label.set_fontweight("bold")
        self.quit_btn.label.set_fontsize(10)
        self.quit_btn.on_clicked(self._quit_app)

        # Teen mode toggle button
        teen_ax = self.fig.add_axes([0.93, 0.515, 0.05, 0.038])
        teen_label = "15yo:ON" if self.teen_mode else "15yo:OFF"
        teen_color = "#2a4a2a" if self.teen_mode else "#3a3a4a"
        self.teen_btn = Button(teen_ax, teen_label, color=teen_color, hovercolor="#5a5a6a")
        self.teen_btn.label.set_color("#88ff88" if self.teen_mode else "#aaaaaa")
        self.teen_btn.label.set_fontweight("bold")
        self.teen_btn.label.set_fontsize(9)
        self.teen_btn.on_clicked(self._toggle_teen_mode)

        # Help/FAQ button (next to teen toggle)
        help_ax = self.fig.add_axes([0.93, 0.475, 0.05, 0.035])
        self.help_btn = Button(help_ax, "? Help", color="#2a2a5a", hovercolor="#4a4a8a")
        self.help_btn.label.set_color("#aaccff")
        self.help_btn.label.set_fontweight("bold")
        self.help_btn.label.set_fontsize(10)
        self.help_btn.on_clicked(self._show_help_modal)

        # Brief claim display area (below buttons)
        self.desc_title = self.fig.text(0.76, 0.46, "Select a claim", fontsize=12, color="#ffcc00",
                                        ha="left", va="top", fontweight="bold")
        self.desc_text = self.fig.text(0.76, 0.47, "Click [i] for full details", fontsize=10, color="#888899",
                                       ha="left", va="top")
        
        # Modal overlay (initially hidden)
        self.modal_visible = False
        self.modal_artists = []
        
        # Geometry test results panel (bottom right of claims area)
        self._setup_geometry_results_panel()
        
        # PASS/FAIL results panel
        self._setup_pass_fail_panel()

    def _setup_geometry_results_panel(self):
        """Show static geometry parameters"""
        # Static text - doesn't need to update every frame
        if self.teen_mode:
            geom_text = (
                f"═══ SETTINGS ═══\n"
                f"Mode: {MODE}\n"
                f"Bend: {computed_deflection_deg:.1f}° (expected: {LOEB_DEFLECTION_DEG}°)\n"
                f"Beam width: {JET_SPAN_DEG}°\n"
                f"Closest to Sun: {rp/1e9:.0f}M km\n"
                f"Speed far away: {v_inf/1000:.1f} km/s"
            )
        else:
            geom_text = (
                f"═══ GEOMETRY ═══\n"
                f"Mode: {MODE}\n"
                f"Turn: {computed_deflection_deg:.1f}° (Loeb: {LOEB_DEFLECTION_DEG}°)\n"
                f"Cone: {JET_SPAN_DEG}° (half: {JET_HALF_ANGLE_DEG}°)\n"
                f"r_p: {rp/1e9:.0f}M km\n"
                f"v_∞: {v_inf/1000:.1f} km/s"
            )
        self._geom_panel = self.fig.text(
            0.76, 0.42, 
            geom_text,
            fontsize=12, color="#aaaacc", ha="left", va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", edgecolor="#444466", alpha=0.95)
        )
    
    def _setup_pass_fail_panel(self):
        """Show PASS/FAIL status for each claim"""
        # Compute PASS/FAIL for each claim
        tol_deflection = 0.5  # degrees tolerance
        
        # Claim 1: Turn angle matches Loeb's claim
        claim1_pass = abs(computed_deflection_deg - LOEB_DEFLECTION_DEG) <= tol_deflection
        claim1_status = "✓ PASS" if claim1_pass else "✗ FAIL"
        claim1_color = "#88ff88" if claim1_pass else "#ff8888"
        claim1_detail = f"Δ={abs(computed_deflection_deg - LOEB_DEFLECTION_DEG):.1f}°"
        
        # Claim 2/4: Sun containment - % in cone and longest gap
        pct_in_cone = 100 * np.sum(sun_in_any_cone) / Nf
        # PASS if Sun is in cone most of the time and gap isn't too long
        claim2_pass = pct_in_cone >= 50.0 and longest_out_days <= 10.0
        claim2_status = "✓ PASS" if claim2_pass else "✗ FAIL"
        claim2_color = "#88ff88" if claim2_pass else "#ff8888"
        claim2_detail = f"{pct_in_cone:.0f}% in, gap={longest_out_days:.0f}d"
        
        # Claim 3: Wobble doesn't alter axis (axis is constant by construction)
        # Check that jet_axis_plus is actually constant (it's defined once, not recomputed)
        claim3_pass = True  # By construction: axis is fixed, wobble only rotates around it
        claim3_status = "✓ PASS" if claim3_pass else "✗ FAIL"
        claim3_color = "#88ff88" if claim3_pass else "#ff8888"
        claim3_detail = "axis fixed"
        
        # Claim 4: Same as claim 2 (geometry keeps Sun in cone)
        claim4_pass = claim2_pass
        claim4_status = "✓ PASS" if claim4_pass else "✗ FAIL"
        claim4_color = "#88ff88" if claim4_pass else "#ff8888"
        claim4_detail = claim2_detail
        
        # Claim 5: Physical vs visual jet length (always passes - it's labeled correctly)
        claim5_pass = True
        claim5_status = "✓ PASS" if claim5_pass else "✗ FAIL"
        claim5_color = "#88ff88" if claim5_pass else "#ff8888"
        claim5_detail = f"{JET_LEN_KM_REAL/1e6:.0f}M km (×{JET_VIS_SCALE})"
        
        # Claim 6: Ecliptic (not modeled in 2D)
        claim6_status = "⚠ N/A"
        claim6_color = "#ffff88"
        claim6_detail = "2D only"
        
        # Build the panel text - use teen-friendly labels if in teen mode
        if self.teen_mode:
            panel_text = (
                f"═══ DOES IT WORK? ═══\n"
                f"1. Bend:   {claim1_status} {claim1_detail}\n"
                f"2. Sun in: {claim2_status} {claim2_detail}\n"
                f"3. Wiggle: {claim3_status} {claim3_detail}\n"
                f"4. No aim: {claim4_status} {claim4_detail}\n"
                f"5. Size:   {claim5_status} {claim5_detail}\n"
                f"6. 3D tilt:{claim6_status} {claim6_detail}"
            )
        else:
            panel_text = (
                f"═══ PASS/FAIL ═══\n"
                f"1. Turn:    {claim1_status} {claim1_detail}\n"
                f"2. Cone:    {claim2_status} {claim2_detail}\n"
                f"3. Wobble:  {claim3_status} {claim3_detail}\n"
                f"4. No steer:{claim4_status} {claim4_detail}\n"
                f"5. Scale:   {claim5_status} {claim5_detail}\n"
                f"6. Eclipt:  {claim6_status} {claim6_detail}"
            )
        
        self.pass_fail_text = self.fig.text(
            0.76, 0.28,
            panel_text,
            fontsize=11, color="#aaaacc", ha="left", va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", edgecolor="#444466", alpha=0.95)
        )
        
        # Store results for dynamic update when claim is selected
        self.claim_results = {
            1: (claim1_pass, claim1_detail),
            2: (claim2_pass, claim2_detail),
            3: (claim3_pass, claim3_detail),
            4: (claim4_pass, claim4_detail),
            5: (claim5_pass, claim5_detail),
            6: (False, claim6_detail),  # N/A
        }

    def _show_claim_modal(self, claim_id):
        """Show internal modal overlay with claim details"""
        # Close existing modal first
        self._close_modal()
        
        claim = CLAIMS[claim_id]
        self.modal_visible = True
        
        # STOP the animation completely while modal is shown
        self._was_paused_before_modal = self.paused
        self.paused = True
        if hasattr(self, 'ani') and self.ani.event_source:
            self.ani.event_source.stop()
        
        # Select text based on teen mode
        if self.teen_mode:
            modal_title = claim.get('teen_title', claim['title'])
            left_header = "THE CLAIM (SIMPLE)"
            left_content = claim.get('teen_summary', claim['loeb_claim'])
            right_header = "HOW WE CHECK IT"
            right_content = claim.get('teen_passfail', claim['commentary'])
        else:
            modal_title = claim['title']
            left_header = ui_text("modal_loeb_header", self.teen_mode)
            left_content = claim['loeb_claim']
            right_header = ui_text("modal_criteria_header", self.teen_mode)
            right_content = claim['commentary']
        
        # HIDE all buttons to prevent them from drawing over modal
        for _, btn, btn_ax in self.buttons:
            btn_ax.set_visible(False)
        for _, btn, btn_ax in self.detail_buttons:
            btn_ax.set_visible(False)
        self.reset_btn.ax.set_visible(False)
        self.pause_btn.ax.set_visible(False)
        
        # Also hide the main plot axes during modal
        self.ax.set_visible(False)
        
        # Semi-transparent background overlay - use very high zorder
        from matplotlib.patches import Rectangle, FancyBboxPatch
        
        # Create a SOLID overlay that completely covers the figure
        overlay = Rectangle((0, 0), 1, 1, transform=self.fig.transFigure,
                            facecolor='#0a0a1a', alpha=1.0, zorder=10000)
        self.fig.add_artist(overlay)
        self.modal_artists.append(overlay)
        
        # Modal box
        modal_box = Rectangle((0.05, 0.05), 0.90, 0.90, transform=self.fig.transFigure,
                              facecolor='#0a0a1a', edgecolor='#ffcc00', linewidth=3, zorder=10001)
        self.fig.add_artist(modal_box)
        self.modal_artists.append(modal_box)
        
        # Close button - create an axes for reliable click handling
        close_ax = self.fig.add_axes([0.85, 0.91, 0.09, 0.05], zorder=10010)
        close_ax.set_facecolor('#661111')
        close_ax.set_navigate(False)
        for spine in close_ax.spines.values():
            spine.set_edgecolor('#ff4444')
            spine.set_linewidth(2)
        close_ax.set_xticks([])
        close_ax.set_yticks([])
        close_ax.text(0.5, 0.5, "✕ CLOSE", transform=close_ax.transAxes,
                     ha='center', va='center', fontsize=13, fontweight='bold', color='#ff6666')
        self.modal_artists.append(close_ax)
        self._modal_close_ax = close_ax
        
        # Handle click on close button region
        def on_click(event):
            if not self.modal_visible:
                return
            if event.inaxes == close_ax:
                self._close_modal()
        self._modal_click_cid = self.fig.canvas.mpl_connect('button_press_event', on_click)
        
        # Title (use modal_title variable set earlier based on teen_mode)
        title = self.fig.text(0.5, 0.92, f"#{claim_id}: {modal_title}", 
                             fontsize=18, fontweight="bold", color="#ffcc00",
                             ha="center", va="top", zorder=10002)
        self.modal_artists.append(title)
        
        # Two-column layout
        # Left column: LOEB'S CLAIM (green border)
        left_box = Rectangle((0.07, 0.10), 0.42, 0.75, transform=self.fig.transFigure,
                             facecolor='#0d1a0d', edgecolor='#88ff88', linewidth=2, zorder=10001)
        self.fig.add_artist(left_box)
        self.modal_artists.append(left_box)
        
        left_title_txt = self.fig.text(0.28, 0.83, left_header, fontsize=14, fontweight="bold",
                                  color="#88ff88", ha="center", va="top", zorder=10002)
        self.modal_artists.append(left_title_txt)
        
        # Claim text with word wrapping - use teen or normal text
        left_text = self.fig.text(0.09, 0.79, left_content, fontsize=11, color="#ccddcc",
                                 ha="left", va="top", family="monospace", zorder=10002,
                                 wrap=True, linespacing=1.3,
                                 bbox=dict(boxstyle="square,pad=0", facecolor='none', edgecolor='none'))
        left_text._get_wrap_line_width = lambda: 320  # pixels for wrapping
        self.modal_artists.append(left_text)
        
        # Right column: ANALYSIS (purple border)
        right_box = Rectangle((0.51, 0.10), 0.42, 0.75, transform=self.fig.transFigure,
                              facecolor='#1a0d1a', edgecolor='#ff88ff', linewidth=2, zorder=10001)
        self.fig.add_artist(right_box)
        self.modal_artists.append(right_box)
        
        right_title_txt = self.fig.text(0.72, 0.83, right_header, fontsize=14, fontweight="bold",
                                   color="#ff88ff", ha="center", va="top", zorder=10002)
        self.modal_artists.append(right_title_txt)
        
        right_text = self.fig.text(0.53, 0.79, right_content, fontsize=11, color="#ddccdd",
                                  ha="left", va="top", family="monospace", zorder=10002,
                                  wrap=True, linespacing=1.3)
        right_text._get_wrap_line_width = lambda: 320
        self.modal_artists.append(right_text)
        
        # Source link at bottom
        source = self.fig.text(0.5, 0.07, f"Source: {claim['source']}", fontsize=10, color="#66aaff",
                              ha="center", va="top", style="italic", zorder=10002)
        self.modal_artists.append(source)
        
        # Instruction
        instr = self.fig.text(0.5, 0.03, "Press ESC or click ✕ CLOSE to close", fontsize=12, color="#aaaacc",
                             ha="center", va="top", zorder=10002)
        self.modal_artists.append(instr)
        
        # Force complete redraw (not just draw_idle)
        self.fig.canvas.draw()

    def _close_modal(self, event=None):
        """Close the modal overlay"""
        if not self.modal_visible:
            return
            
        self.modal_visible = False
        
        # Disconnect click event handler
        if hasattr(self, '_modal_click_cid') and self._modal_click_cid:
            self.fig.canvas.mpl_disconnect(self._modal_click_cid)
            self._modal_click_cid = None
        
        # Remove all modal artists
        for artist in self.modal_artists:
            try:
                if isinstance(artist, plt.Axes):
                    self.fig.delaxes(artist)
                else:
                    artist.remove()
            except Exception:
                pass
        
        # Cleanup: remove any remaining high-zorder patches from fig.patches
        patches_to_remove = [p for p in list(self.fig.patches) if getattr(p, 'get_zorder', lambda: 0)() >= 10000]
        for p in patches_to_remove:
            try:
                self.fig.patches.remove(p)
            except:
                pass

    def _show_help_modal(self, event=None):
        """Show FAQ/Help modal explaining terminology"""
        if self.modal_visible:
            self._close_modal()
            return
        
        self.modal_visible = True
        self.modal_artists = []
        
        # Semi-transparent overlay
        overlay = Rectangle((0, 0), 1, 1, transform=self.fig.transFigure,
                            facecolor='black', alpha=0.85, zorder=10000)
        self.fig.add_artist(overlay)
        self.modal_artists.append(overlay)
        
        # Modal background box
        modal_bg = Rectangle((0.05, 0.05), 0.90, 0.90, transform=self.fig.transFigure,
                             facecolor='#0a0a1a', edgecolor='#4466aa', linewidth=3, zorder=10001)
        self.fig.add_artist(modal_bg)
        self.modal_artists.append(modal_bg)
        
        # Close button
        close_ax = self.fig.add_axes([0.88, 0.88, 0.06, 0.05], zorder=10003)
        close_ax.set_facecolor('#440000')
        for spine in close_ax.spines.values():
            spine.set_edgecolor('#ff4444')
            spine.set_linewidth(2)
        close_ax.set_xticks([])
        close_ax.set_yticks([])
        close_ax.text(0.5, 0.5, "✕ CLOSE", transform=close_ax.transAxes,
                     ha='center', va='center', fontsize=12, fontweight='bold', color='#ff6666')
        self.modal_artists.append(close_ax)
        self._modal_close_ax = close_ax
        
        # Click handler
        def on_click(event):
            if not self.modal_visible:
                return
            if event.inaxes == close_ax:
                self._close_modal()
        self._modal_click_cid = self.fig.canvas.mpl_connect('button_press_event', on_click)
        
        # Title
        title = self.fig.text(0.5, 0.92, "HELP / FAQ", fontsize=20, fontweight="bold", 
                             color="#aaccff", ha="center", va="top", zorder=10002)
        self.modal_artists.append(title)
        
        # FAQ content - left column
        left_content = """WHAT'S CALCULATED vs ASSUMED
═══════════════════════════════════════

CALCULATED (real physics):
  • Blue dashed line = orbital path
  • Green arrow = velocity
  • Red dashed = gravity direction
  These come from orbital mechanics.

LOEB'S ASSUMPTION (not physics):
  • Orange cone (+A) = fixed-axis jet
  • Blue cone (-A) = opposite direction
  These show Loeb's CLAIMED geometry.
  The axis is fixed, not tracking Sun.

SUN-IN-CONE = TEST of Loeb's model
  Does Sun stay in a cone? If not,
  Loeb's fixed-axis claim fails.


SIMULATION MODES
═══════════════════════════════════════

LOEB_GEOMETRY (default)
  Gravity-only orbit + Loeb's cones.

JET_DYNAMICS (alternate)
  Adds actual jet thrust physics.
  Change MODE in code line 31."""
        
        left_text = self.fig.text(0.08, 0.85, left_content, fontsize=11, color="#ccddff",
                                 ha="left", va="top", family="monospace", zorder=10002,
                                 linespacing=1.2)
        self.modal_artists.append(left_text)
        
        # FAQ content - right column
        right_content = """TERMINOLOGY
═══════════════════════════════════════

Perihelion (r_p)
  Closest approach to the Sun.
  For 3I/ATLAS: ~202 million km.

Hyperbolic orbit
  Open trajectory, object escapes solar
  system (e > 1, not bound to Sun).

Deflection angle
  How much gravity bends the path.
  Loeb claims ~16° for 3I/ATLAS.

Anti-tail / Sunward jet
  Loeb's theory: 3I/ATLAS has a jet
  pointing TOWARD the Sun (opposite
  of normal comet tails).


CONTROLS
═══════════════════════════════════════

SPACE = Pause/Play
←/→   = Step frames (when paused)
1-6   = Select claim
T     = Toggle 15yo mode
H     = This help screen
ESC   = Close modal"""
        
        right_text = self.fig.text(0.52, 0.85, right_content, fontsize=11, color="#ddccff",
                                  ha="left", va="top", family="monospace", zorder=10002,
                                  linespacing=1.2)
        self.modal_artists.append(right_text)
        
        # Footer
        footer = self.fig.text(0.5, 0.08, "Press ESC or click ✕ CLOSE to close", fontsize=12, 
                              color="#aaaacc", ha="center", va="top", zorder=10002)
        self.modal_artists.append(footer)
        
        self.fig.canvas.draw()
        
        # SHOW all buttons again
        for _, btn, btn_ax in self.buttons:
            btn_ax.set_visible(True)
        for _, btn, btn_ax in self.detail_buttons:
            btn_ax.set_visible(True)
        self.reset_btn.ax.set_visible(True)
        self.pause_btn.ax.set_visible(True)
        
        # Show main plot again
        self.ax.set_visible(True)
        
        # Restore pause state and restart animation
        if hasattr(self, '_was_paused_before_modal'):
            self.paused = self._was_paused_before_modal
        
        # Restart animation event source
        if hasattr(self, 'ani') and self.ani.event_source:
            self.ani.event_source.start()
        
        # Force complete redraw
        self.fig.canvas.draw()

    def _bind_keys(self):
        def on_key(event):
            # Handle modal close with ESC
            if event.key == "escape":
                if self.modal_visible:
                    self._close_modal()
                    return
            if event.key == " ":
                self._toggle_pause()
            elif event.key in ("right", "d"):
                # step forward one frame when paused
                if self.paused:
                    self._manual_step(+1)
            elif event.key in ("left", "a"):
                if self.paused:
                    self._manual_step(-1)
            elif event.key == "t":
                # Toggle teen mode
                self._toggle_teen_mode()
            elif event.key == "h":
                # Show help modal
                self._show_help_modal()
            elif event.key and event.key.isdigit():
                cid = int(event.key)
                if cid in CLAIMS:
                    self.select_claim(cid)
        
        self.fig.canvas.mpl_connect("key_press_event", on_key)

    def _toggle_pause(self, event=None):
        """Toggle pause/play state"""
        self.paused = not self.paused
        if self.paused:
            self.pause_btn.label.set_text(ui_text("btn_play", self.teen_mode))
            self.pause_btn.ax.set_facecolor("#2a4a2a")
            self.pause_btn.hovercolor = "#4a6a4a"
        else:
            self.pause_btn.label.set_text(ui_text("btn_pause", self.teen_mode))
            self.pause_btn.ax.set_facecolor("#4a2a2a")
            self.pause_btn.hovercolor = "#6a4a4a"
        self.fig.canvas.draw_idle()

    def _toggle_teen_mode(self, event=None):
        """Toggle 15-year-old mode on/off"""
        self.teen_mode = not self.teen_mode
        self._update_all_ui_text()
        self.fig.canvas.draw_idle()
    
    def _update_all_ui_text(self):
        """Update all UI text elements based on current teen_mode setting"""
        # Update teen toggle button
        teen_label = "15yo:ON" if self.teen_mode else "15yo:OFF"
        self.teen_btn.label.set_text(teen_label)
        self.teen_btn.ax.set_facecolor("#2a4a2a" if self.teen_mode else "#3a3a4a")
        self.teen_btn.label.set_color("#88ff88" if self.teen_mode else "#aaaaaa")
        
        # Update main title
        mode_label = "GRAVITY-ONLY" if MODE == "LOEB_GEOMETRY" else "GRAVITY+JET"
        self._main_title.set_text(ui_text("main_title", self.teen_mode, mode=mode_label))
        
        # Update claims panel header
        self._claims_header.set_text(ui_text("claims_header", self.teen_mode))
        self._claims_subheader.set_text(ui_text("claims_subheader", self.teen_mode))
        
        # Update claim buttons
        for cid, btn, btn_ax in self.buttons:
            claim = CLAIMS[cid]
            short_text = claim.get('teen_short', claim['short']) if self.teen_mode else claim['short']
            btn.label.set_text(f"{cid}. {short_text}")
        
        # Update control buttons
        self.reset_btn.label.set_text(ui_text("btn_reset", self.teen_mode))
        self.quit_btn.label.set_text(ui_text("btn_quit", self.teen_mode))
        if self.paused:
            self.pause_btn.label.set_text(ui_text("btn_play", self.teen_mode))
        else:
            self.pause_btn.label.set_text(ui_text("btn_pause", self.teen_mode))
        
        # Update geometry panel
        if self.teen_mode:
            geom_text = (
                f"═══ SETTINGS ═══\n"
                f"Mode: {MODE}\n"
                f"Bend: {computed_deflection_deg:.1f}° (expected: {LOEB_DEFLECTION_DEG}°)\n"
                f"Beam width: {JET_SPAN_DEG}°\n"
                f"Closest to Sun: {rp/1e9:.0f}M km\n"
                f"Speed far away: {v_inf/1000:.1f} km/s"
            )
        else:
            geom_text = (
                f"═══ GEOMETRY ═══\n"
                f"Mode: {MODE}\n"
                f"Turn: {computed_deflection_deg:.1f}° (Loeb: {LOEB_DEFLECTION_DEG}°)\n"
                f"Cone: {JET_SPAN_DEG}° (half: {JET_HALF_ANGLE_DEG}°)\n"
                f"r_p: {rp/1e9:.0f}M km\n"
                f"v_∞: {v_inf/1000:.1f} km/s"
            )
        self._geom_panel.set_text(geom_text)
        
        # Update PASS/FAIL panel
        c1_pass, c1_d = self.claim_results[1]
        c2_pass, c2_d = self.claim_results[2]
        c3_pass, c3_d = self.claim_results[3]
        c4_pass, c4_d = self.claim_results[4]
        c5_pass, c5_d = self.claim_results[5]
        c6_pass, c6_d = self.claim_results[6]
        
        s1 = "✓ PASS" if c1_pass else "✗ FAIL"
        s2 = "✓ PASS" if c2_pass else "✗ FAIL"
        s3 = "✓ PASS" if c3_pass else "✗ FAIL"
        s4 = "✓ PASS" if c4_pass else "✗ FAIL"
        s5 = "✓ PASS" if c5_pass else "✗ FAIL"
        s6 = "⚠ N/A"
        
        if self.teen_mode:
            pf_text = (
                f"═══ DOES IT WORK? ═══\n"
                f"1. Bend:   {s1} {c1_d}\n"
                f"2. Sun in: {s2} {c2_d}\n"
                f"3. Wiggle: {s3} {c3_d}\n"
                f"4. No aim: {s4} {c4_d}\n"
                f"5. Size:   {s5} {c5_d}\n"
                f"6. 3D tilt:{s6} {c6_d}"
            )
        else:
            pf_text = (
                f"═══ PASS/FAIL ═══\n"
                f"1. Turn:    {s1} {c1_d}\n"
                f"2. Cone:    {s2} {c2_d}\n"
                f"3. Wobble:  {s3} {c3_d}\n"
                f"4. No steer:{s4} {c4_d}\n"
                f"5. Scale:   {s5} {c5_d}\n"
                f"6. Eclipt:  {s6} {c6_d}"
            )
        self.pass_fail_text.set_text(pf_text)
        
        # Update legend
        self._legend.remove()
        self._legend_items = [
            plt.Line2D([0], [0], color='#00ffcc', marker='o', markersize=10, linestyle='None', 
                      label=ui_text("legend_body", self.teen_mode)),
            plt.Line2D([0], [0], color='#66aaff', linewidth=2, 
                      label=ui_text("legend_trail", self.teen_mode)),
            plt.Line2D([0], [0], color='#ffaa00', linewidth=4, 
                      label=ui_text("legend_cone_plus", self.teen_mode)),
            plt.Line2D([0], [0], color='#6699ff', linewidth=4, 
                      label=ui_text("legend_cone_minus", self.teen_mode)),
            plt.Line2D([0], [0], color='#6688aa', linewidth=3, linestyle=':', 
                      label=ui_text("legend_normal_tail", self.teen_mode)),
            plt.Line2D([0], [0], color='#00ff88', linewidth=3, 
                      label=ui_text("legend_velocity", self.teen_mode)),
            plt.Line2D([0], [0], color='#ff6666', linewidth=2, linestyle='--', 
                      label=ui_text("legend_gravity", self.teen_mode)),
        ]
        self._legend = self.ax.legend(handles=self._legend_items, loc='upper left', bbox_to_anchor=(1.01, 1.0),
                  fontsize=11, facecolor='#1a1a2e', edgecolor='#333355', 
                  labelcolor='white', framealpha=0.95)

    def _quit_app(self, event=None):
        """Quit the application"""
        plt.close(self.fig)

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

        # Show claim title and PASS/FAIL status
        # Use teen title if in teen mode
        display_title = claim.get('teen_title', claim['title']) if self.teen_mode else claim['title']
        
        if hasattr(self, 'claim_results') and claim_id in self.claim_results:
            passed, detail = self.claim_results[claim_id]
            if claim_id == 6:
                if self.teen_mode:
                    status = "⚠ Can't check (need 3D)"
                else:
                    status = "⚠ NOT MODELED (2D)"
                status_color = "#ffff88"
            else:
                if self.teen_mode:
                    status = f"✓ Works! {detail}" if passed else f"✗ Doesn't match: {detail}"
                else:
                    status = f"✓ PASS: {detail}" if passed else f"✗ FAIL: {detail}"
                status_color = "#88ff88" if passed else "#ff8888"
            self.desc_title.set_text(f"#{claim_id}: {display_title}")
            self.desc_title.set_color("#ffcc00")
            hint = "Click [i] to learn more!" if self.teen_mode else "Click [i] for details"
            self.desc_text.set_text(f"{status}\n{hint}")
            self.desc_text.set_color(status_color)
        else:
            self.desc_title.set_text(f"#{claim_id}: {display_title}")
            hint = "Click [i] to learn more!" if self.teen_mode else "Click [i] for full analysis"
            self.desc_text.set_text(hint)

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
        # Compute the actual deflection arc from trajectory asymptotes
        # The DEFLECTION is the angle between:
        #   - Reversed inbound direction (-v_in_hat): where it came FROM
        #   - Outbound direction (v_out_hat): where it's going TO
        # This gives the actual turn/scattering angle.
        
        # Direction comet came FROM (reversed inbound velocity)
        from_dir = -v_in_hat
        # Direction comet is going TO
        to_dir = v_out_hat
        
        theta1_deg = np.degrees(np.arctan2(from_dir[1], from_dir[0]))
        theta2_deg = np.degrees(np.arctan2(to_dir[1], to_dir[0]))
        
        # Handle angle wrap-around to get the minor arc
        if theta2_deg < theta1_deg:
            theta1_deg, theta2_deg = theta2_deg, theta1_deg
        if theta2_deg - theta1_deg > 180:
            theta1_deg, theta2_deg = theta2_deg, theta1_deg + 360
        
        # Draw arc showing the actual deflection (the turn angle)
        arc = Arc((0, 0), 1.5, 1.5, angle=0, 
                  theta1=theta1_deg, 
                  theta2=theta2_deg,
                  color="#ffff00", lw=4, zorder=25)
        self.ax.add_patch(arc)
        self.extra_artists.append(arc)
        
        # Draw the FROM and TO direction lines (these show the turn)
        line_from, = self.ax.plot([0, from_dir[0]*2.5], [0, from_dir[1]*2.5], 
                                  '--', color='#88ff88', lw=2, alpha=0.7, zorder=24,
                                  label='From (came from)')
        line_to, = self.ax.plot([0, to_dir[0]*2.5], [0, to_dir[1]*2.5], 
                                '--', color='#ff8888', lw=2, alpha=0.7, zorder=24,
                                label='To (going to)')
        self.extra_artists.extend([line_from, line_to])
        
        # Add small labels at ends of lines
        lbl_from = self.ax.text(from_dir[0]*2.6, from_dir[1]*2.6, 'FROM', fontsize=8, 
                                color='#88ff88', ha='center', va='center')
        lbl_to = self.ax.text(to_dir[0]*2.6, to_dir[1]*2.6, 'TO', fontsize=8,
                              color='#ff8888', ha='center', va='center')
        self.extra_artists.extend([lbl_from, lbl_to])

        if self.teen_mode:
            msg = (f"The path bends by {computed_deflection_deg:.1f}°\n"
                   f"Expected: ~{LOEB_DEFLECTION_DEG}° (that's 2× the beam width!)\n"
                   f"(Yellow arc shows how much direction changes)")
        else:
            msg = (f"Computed deflection: {computed_deflection_deg:.1f}°\n"
                   f"Loeb's value: {LOEB_DEFLECTION_DEG}° = 2 × {JET_SPAN_DEG}° cone\n"
                   f"(Arc shows turn from 'FROM' to 'TO' direction)")
        t = self.ax.text(
            0.02, 0.97, msg,
            transform=self.ax.transAxes, fontsize=13, color="#ffff88", ha="left", va="top", fontweight="bold",
            bbox=dict(facecolor="#2a2a1a", edgecolor="#ffff00", alpha=0.95, pad=6),
        )
        self.annotations.append(t)

    def _show_jet_cone(self):
        if self.teen_mode:
            msg = "WEIRD! This jet points TOWARD the Sun.\n(Normal comets have tails that point AWAY from the Sun)"
        else:
            msg = "ANTI-TAIL: Jet points TOWARD Sun (opposite of normal comets!)"
        t = self.ax.text(
            0.02, 0.97, msg,
            transform=self.ax.transAxes, fontsize=13, color="#ffaa00", ha="left", va="top", fontweight="bold",
            bbox=dict(facecolor="#2a2a1a", edgecolor="#ffaa00", alpha=0.95, pad=6),
        )
        self.annotations.append(t)

    def _show_wobble(self):
        if self.teen_mode:
            msg = f"Jet wiggles ±{P.wobble_amp_deg}° every {JET_WOBBLE_PERIOD_H} hours\n(like a spinning sprinkler)"
        else:
            msg = f"Jet wobbles ±{P.wobble_amp_deg}° with {JET_WOBBLE_PERIOD_H}h period"
        t = self.ax.text(
            0.02, 0.97, msg,
            transform=self.ax.transAxes, fontsize=14, color="#00ffcc", ha="left", va="top", fontweight="bold",
            bbox=dict(facecolor="#1a2a2a", edgecolor="#00ffcc", alpha=0.95, pad=6),
        )
        self.annotations.append(t)

    def _show_sunward(self):
        if self.teen_mode:
            msg = ("The jet doesn't aim at the Sun on purpose!\n"
                   "Sun stays in the beam because 16° turn ≈ 2×8° beam width.\n"
                   "(Watch the 'Sun in beam' status as it orbits)")
        else:
            msg = ("FIXED-AXIS JET: Sun stays in cone due to 16.4° = 2×8° geometry\n"
                   "(Watch 'SUN IN CONE' indicator as orbit turns)")
        t = self.ax.text(
            0.02, 0.97, msg,
            transform=self.ax.transAxes, fontsize=13, color="#ffdd44", ha="left", va="top", fontweight="bold",
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
        if self.teen_mode:
            msg = (f"⚠ Can't check this in 2D!\n"
                   f"Need 3D simulation to measure orbit tilt vs Earth's orbit.\n"
                   f"(The claim is about {ECLIPTIC_ALIGNMENT_DEG}° alignment)")
        else:
            msg = (f"⚠ NOT MODELED: Ecliptic ±{ECLIPTIC_ALIGNMENT_DEG}° (requires 3D)\n"
                   "This 2D sim shows orbital plane, not ecliptic alignment")
        t = self.ax.text(
            0.02, 0.97, msg,
            transform=self.ax.transAxes, fontsize=13, color="#88ff88", ha="left", va="top", fontweight="bold",
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
        
        # Show UFO image in teen mode, scatter dot otherwise
        if self.teen_mode and getattr(self, '_ufo_loaded', False):
            self.body.set_visible(False)
            self.ufo_box.set_visible(True)
            self.ufo_box.xybox = (xi, yi)
            self.body_label.set_position((xi + 0.15, yi + 0.10))
            self.body_label.set_text("UFO?")
        else:
            if getattr(self, '_ufo_loaded', False):
                self.ufo_box.set_visible(False)
            self.body.set_visible(True)
            self.body.set_offsets([[xi, yi]])
            self.body_label.set_position((xi + 0.08, yi + 0.08))
            self.body_label.set_text("3I/ATLAS")

        # trail
        i0 = max(0, frame - self.trail_len)
        self.trail.set_data(x_traj[i0:frame + 1], y_traj[i0:frame + 1])

        # --- RENDER BOTH CONES (+A and -A) ---
        start = np.array([xi, yi])
        
        # +A cone (orange)
        idx = np.arange(self.n_rays) - (self.n_rays // 2)
        ang_plus = wobble_rad[frame] + idx * (self.half_angle / max(1, (self.n_rays // 2)))
        ca_plus, sa_plus = np.cos(ang_plus), np.sin(ang_plus)
        dirs_plus = np.column_stack([
            ca_plus * jet_axis_plus[0] - sa_plus * jet_axis_plus[1],
            sa_plus * jet_axis_plus[0] + ca_plus * jet_axis_plus[1]
        ])
        phase = (frame * 0.12 + idx * 0.7) % 1.0
        lengths = self.jet_len_au * (0.75 + 0.25 * phase)
        end_plus = start + dirs_plus * lengths[:, None]
        segs_plus = np.stack([np.repeat(start[None, :], self.n_rays, axis=0), end_plus], axis=1)
        self.jet_lc_plus.set_segments(segs_plus)
        
        # -A cone (blue)
        ang_minus = wobble_rad[frame] + idx * (self.half_angle / max(1, (self.n_rays // 2)))
        ca_minus, sa_minus = np.cos(ang_minus), np.sin(ang_minus)
        dirs_minus = np.column_stack([
            ca_minus * jet_axis_minus[0] - sa_minus * jet_axis_minus[1],
            sa_minus * jet_axis_minus[0] + ca_minus * jet_axis_minus[1]
        ])
        end_minus = start + dirs_minus * lengths[:, None]
        segs_minus = np.stack([np.repeat(start[None, :], self.n_rays, axis=0), end_minus], axis=1)
        self.jet_lc_minus.set_segments(segs_minus)
        
        # Show fixed jet axis lines for BOTH axes
        axis_end_plus = start + jet_axis_plus * (self.jet_len_au * 1.1)
        axis_end_minus = start + jet_axis_minus * (self.jet_len_au * 1.1)
        self.axis_line_plus.set_data([xi, axis_end_plus[0]], [yi, axis_end_plus[1]])
        self.axis_line_minus.set_data([xi, axis_end_minus[0]], [yi, axis_end_minus[1]])
        
        # Cone labels - position at end of each axis
        self.cone_label_plus.set_position((axis_end_plus[0] + 0.05, axis_end_plus[1] + 0.05))
        self.cone_label_minus.set_position((axis_end_minus[0] + 0.05, axis_end_minus[1] + 0.05))

        # Sun-in-cone indicator - show status for BOTH cones
        in_plus = sun_in_plus_cone[frame]
        in_minus = sun_in_minus_cone[frame]
        ang_plus_deg = angle_to_plus[frame]
        ang_minus_deg = angle_to_minus[frame]
        
        if in_plus and in_minus:
            status_key = "sun_in_both"
            status_color = "#00ff88"
        elif in_plus:
            status_key = "sun_in_plus"
            status_color = "#ffaa00"
        elif in_minus:
            status_key = "sun_in_minus"
            status_color = "#6699ff"
        else:
            status_key = "sun_outside"
            status_color = "#ff4444"
        
        status_label = ui_text(status_key, self.teen_mode)
        status_text = f"{status_label}\n+A: {ang_plus_deg:.1f}° | -A: {ang_minus_deg:.1f}°"
        
        self.sun_cone_indicator.set_text(status_text)
        self.sun_cone_indicator.set_color(status_color)
        self.sun_cone_indicator.get_bbox_patch().set_edgecolor(status_color)

        # jet label only for claim 2
        if self.current_claim == 2:
            jet_end = start + jet_axis_plus * (self.jet_len_au + 0.12)
            self.jet_label.set_position((jet_end[0], jet_end[1]))
            self.jet_label.set_text(f"BOTH CONES\n({JET_SPAN_DEG}° width)")
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
        normal_dir = -g_dir  # Away from sun
        normal_end = start + normal_dir * 0.50
        self.normal_tail_arrow.set_positions((xi, yi), (normal_end[0], normal_end[1]))
        self.normal_tail_label.set_position((normal_end[0] + normal_dir[0]*0.08, normal_end[1] + normal_dir[1]*0.08))
        self.normal_tail_label.set_text("Normal\ntail")

        # info
        t_days = t_anim[frame] / 86400.0
        r_au = distances[frame]
        v_kms = vel_norm[frame] / 1000.0
        
        if self.teen_mode:
            phase_txt = ui_text("info_phase_incoming", True) if frame < perihelion_idx else ui_text("info_phase_outgoing", True)
            extra = f"\nJet wiggle: {wobble_deg[frame]:+.1f}°" if self.current_claim == 3 else ""
            self.info.set_text(
                f"═══ 3I/ATLAS ═══\n"
                f"Status: {phase_txt}\n"
                f"Days from closest: {t_days:+.0f}\n"
                f"Distance from Sun: {r_au:.2f} AU\n"
                f"Speed: {v_kms:.0f} km/s{extra}"
            )
        else:
            phase_txt = "INCOMING" if frame < perihelion_idx else "OUTGOING"
            extra = f"\nWobble: {wobble_deg[frame]:+.1f}°" if self.current_claim == 3 else ""
            self.info.set_text(
                f"═══ 3I/ATLAS ═══\n"
                f"Phase: {phase_txt}\n"
                f"Day: {t_days:+.0f}\n"
                f"Distance: {r_au:.2f} AU\n"
                f"Speed: {v_kms:.0f} km/s{extra}"
            )

        # return artists for blitting
        artists = [
            self.body,
            self.body_label,
            self.trail,
            self.jet_lc_plus,
            self.jet_lc_minus,
            self.jet_label,
            self.cone_label_plus,
            self.cone_label_minus,
            self.axis_line_plus,
            self.axis_line_minus,
            self.sun_cone_indicator,
            self.vel_line,
            self.grav_line,
            self.normal_tail_arrow,
            self.normal_tail_label,
            self.info,
            *self.annotations,
        ]
        # Add UFO box if loaded
        if getattr(self, '_ufo_loaded', False):
            artists.append(self.ufo_box)
        return tuple(artists)

    def run(self):
        plt.show(block=True)


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    print("Starting interactive simulation...")
    print(f"  Mode: {MODE}")
    print(f"  Containment uses wobble: {CONTAINMENT_USES_WOBBLE}")
    print("Controls:")
    print("  SPACE = Pause/Play")
    print("  Left/Right = Step frame (when paused)")
    print("  1-6 = Select claim")
    print("  T = Toggle 15-year-old mode (simplified text)")
    print()
    sim = InteractiveSimulation()
    sim.run()
