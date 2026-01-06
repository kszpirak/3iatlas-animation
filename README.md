# 3I/ATLAS Interactive Orbital Simulation

An interactive Python simulation visualizing the orbital mechanics and jet geometry claims from Avi Loeb's analysis of interstellar object 3I/ATLAS (December 2025).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## What Is This?

This is an educational visualization tool that:

1. **Computes the orbital trajectory** of 3I/ATLAS using real physics (2-body gravitational dynamics)
2. **Overlays Loeb's claimed jet geometry** (fixed-axis cones) to test whether his geometric model works
3. **Provides interactive claim-by-claim analysis** with pass/fail verification

### Key Distinction

| Element                            | Source                                |
| ---------------------------------- | ------------------------------------- |
| **Orbital path** (blue dashed)     | Calculated from gravitational physics |
| **Velocity vector** (green arrow)  | Calculated                            |
| **Gravity direction** (red dashed) | Calculated                            |
| **Jet cones (+A/-A)**              | Loeb's _assumed_ fixed-axis geometry  |
| **Sun-in-cone test**               | Verification of Loeb's assumption     |

The simulation does NOT assume Loeb is correct — it tests whether his geometric claims hold up against real orbital mechanics.

## Loeb's Claims (Tested)

Based on Avi Loeb's December 17, 2025 Medium post ["3I/ATLAS Maintained a Sunward Jet After Its Gravitational Deflection by 16 Degrees at Perihelion"](https://avi-loeb.medium.com/3i-atlas-maintained-a-sunward-jet-after-its-gravitational-deflection-by-16-degrees-at-perihelion-e6810be9b3d8):

| #   | Claim                         | Simulation Status            |
| --- | ----------------------------- | ---------------------------- |
| 1   | **Gravitational turn ~16.4°** | ✅ Computed & verified       |
| 2   | **Jet cone width ~8°**        | ✅ Rendered & tested         |
| 3   | **Wobble period 7.74h, ±4°**  | ✅ Animated                  |
| 4   | **No active steering**        | ✅ Both cones always shown   |
| 5   | **Jet length ~10⁶ km**        | ✅ Scaled (60× visual)       |
| 6   | **Ecliptic alignment ±5°**    | ⚠️ Requires 3D (not modeled) |

## Screenshots

The simulation shows:

- Real-time orbital animation around the Sun
- Dual jet cones (+A orange, -A blue) representing Loeb's geometry
- Sun-in-cone status indicator (TEST result)
- Interactive claims panel with detailed explanations
- "15-year-old mode" for simplified explanations

## Installation

### Requirements

- Python 3.10 or higher
- numpy, scipy, matplotlib

### Quick Start

```bash
git clone https://github.com/kszpirak/3iatlas-animation.git
cd 3iatlas-animation
./install.sh   # Creates venv, installs dependencies
./run.sh       # Launch simulation
```

### Manual Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Usage

### Controls

| Key       | Action                    |
| --------- | ------------------------- |
| **Space** | Pause / Play              |
| **←/→**   | Step frames (when paused) |
| **1-6**   | Select claim to highlight |
| **T**     | Toggle "15-year-old mode" |
| **H**     | Show help / FAQ           |
| **ESC**   | Close modal               |

### UI Elements

- **Claim buttons (1-6)**: Click to highlight specific claims
- **[i] buttons**: Show detailed claim analysis modal
- **? Help**: FAQ explaining terminology and what's calculated vs assumed
- **15yo toggle**: Simplified explanations for younger audiences

## Simulation Modes

The code supports two physics modes (set via `MODE` constant in `main.py`):

### LOEB_GEOMETRY (default)

- Pure gravity physics (Keplerian hyperbolic orbit)
- Jet cones are visualization only
- Tests whether Loeb's fixed-axis geometry works

### JET_DYNAMICS

- Gravity + actual jet thrust physics
- Jet actively affects trajectory
- More realistic but deviates from Loeb's specific claims

## Physics Background

### Gravitational Deflection Formula

The deflection angle for an object passing a massive body is given by a well-established formula from classical mechanics:

$$\theta = \frac{2GM}{b \cdot v^2}$$

| Symbol | Meaning                                                 |
| ------ | ------------------------------------------------------- |
| θ      | Deflection angle (radians)                              |
| G      | Gravitational constant (6.674 × 10⁻¹¹ m³/kg·s²)         |
| M      | Mass of deflecting body (Sun = 1.989 × 10³⁰ kg)         |
| b      | Impact parameter (closest approach to undeflected path) |
| v      | Velocity at infinity                                    |

This is the **Newtonian** version, valid for objects moving much slower than light. For comparison, Einstein's General Relativity predicts **twice** this deflection for light (θ = 4GM/bc²), famously confirmed during the 1919 solar eclipse.

For 3I/ATLAS at ~68 km/s (0.02% of light speed), the Newtonian formula applies. The formula itself is textbook physics — Loeb's application of it is what this simulation tests.

## Technical Details

### Orbital Parameters (from Loeb)

- Perihelion distance: 202 million km (~1.35 AU)
- Velocity at perihelion: 68 km/s
- Gravitational deflection: ~16.4° (claimed)
- Jet cone half-angle: 4° (8° full width)
- Wobble period: 7.74 hours
- Wobble amplitude: ±4°

### ⚠️ Note on Deflection Angle

[Loeb's article](https://avi-loeb.medium.com/3i-atlas-maintained-a-sunward-jet-after-its-gravitational-deflection-by-16-degrees-at-perihelion-e6810be9b3d8) states:

> _"The direction of motion of 3I/ATLAS was shifted by the following angle (in radians) at perihelion: 2GM/(b·v²) = 0.286 = 16.4 degrees, where G is Newton's constant, M is the mass of the Sun, b=202 million kilometers is the perihelion distance and v=68 kilometers per second is the perihelion speed."_

However, in classical scattering physics, `b` is the **impact parameter** (perpendicular distance from Sun to the undeflected asymptotic trajectory), not the perihelion distance.

The simulation computes both:

- **Perihelion distance (rp):** 202 million km (Loeb's value)
- **Impact parameter (b):** ~239 million km (calculated)
- **Calculated deflection:** ~19° (using correct impact parameter)
- **Loeb's claimed deflection:** 16.4° (using rp as "b")

This discrepancy is shown in the simulation output. It may indicate an error in Loeb's formula application, or a different definition of parameters. The simulation exposes this for users to evaluate.

### Physics

- 2-body gravitational dynamics via `scipy.integrate.solve_ivp`
- Hyperbolic trajectory (eccentricity > 1)
- Standard gravitational parameter μ☉ = 1.327×10²⁰ m³/s²

### Visualization

- matplotlib with FuncAnimation
- Jet length exaggerated 60× for visibility
- All angle calculations use true geometry

## File Structure

```
3iatlas-animation/
├── main.py           # Main simulation code
├── run.sh            # Launch script
├── install.sh        # Setup script
├── requirements.txt  # Python dependencies
├── 3i-atlas.png      # Custom icon (optional)
└── README.md         # This file
```

## Acknowledgments

- Orbital mechanics based on standard 2-body problem
- Claims and parameters from [Avi Loeb's Medium post](https://avi-loeb.medium.com/3i-atlas-maintained-a-sunward-jet-after-its-gravitational-deflection-by-16-degrees-at-perihelion-e6810be9b3d8) (December 17, 2025)
- Wobble period from cited preprint
- Built with assistance from ChatGPT 5.2 and Claude (AI pair programmers)

## Disclaimer

This is an independent educational visualization. It does not endorse or refute Loeb's claims — it provides a tool for visualizing and testing the geometry he describes against standard orbital mechanics.

## License

MIT License - See LICENSE file for details.
