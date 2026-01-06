# 3I/ATLAS Animation

An illustrative 2D simulation of the interstellar object **3I/ATLAS**, combining solar gravity with a time-varying sunward jet model featuring periodic wobble.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)

## Overview

This project simulates the trajectory of 3I/ATLAS using:

- **Two-body orbital mechanics** (solar gravity)
- **Sunward jet acceleration** with periodic wobble (~7.74 hour period)
- **Factual parameters** from Avi Loeb's analysis (Dec 17, 2025)

The animation displays the object's path around perihelion, showing velocity, gravitational acceleration, and jet thrust vectors in real-time.

## Key Parameters (from Loeb's Article)

| Parameter                         | Value           |
| --------------------------------- | --------------- |
| Perihelion distance               | ~202 million km |
| Perihelion speed                  | ~68 km/s        |
| Gravitational deflection estimate | ~16.4°          |
| Jet span                          | ~8°             |
| Jet wobble period                 | ~7.74 hours     |

## Requirements

### Python Dependencies

```
numpy
scipy
matplotlib
```

### Optional (for MP4 export)

- **ffmpeg** installed and available in PATH

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/3iatlas-animation.git
   cd 3iatlas-animation
   ```

2. Install dependencies:
   ```bash
   pip install numpy scipy matplotlib
   ```

## Usage

Run the simulation:

```bash
python main.py
```

This will open an interactive matplotlib window showing the animated trajectory.

### Saving to MP4

To save the animation as a video file, uncomment the following lines at the end of `main.py`:

```python
writer = FFMpegWriter(fps=30, bitrate=1800)
ani.save("3i_atlas_sim.mp4", writer=writer)
```

## Animation Features

- **Yellow sun** at the center (origin)
- **Blue trajectory** showing the full orbital path
- **Velocity vector** (visual scale)
- **Gravity acceleration vector**
- **Jet acceleration vector** with wobble
- **Jet cone edges** showing the ~8° emission span
- **Real-time info overlay** with factual values and simulation time

## Model Parameters

The simulation includes adjustable parameters in the `ModelParams` dataclass:

- `wobble_amp_deg`: Jet wobble amplitude (default: 4°)
- `a0`: Peak jet acceleration near perihelion (default: 5×10⁻⁵ m/s²)
- `r_ref`: Reference distance for jet scaling (1 AU)
- `r_cut`: Cutoff distance where jet fades (2 AU)
- `t_span_days`: Simulation duration around perihelion (±120 days)

## Physics Notes

The simulation integrates the equations of motion using `scipy.integrate.solve_ivp` with:

- Forward and backward integration from perihelion
- High precision tolerances (rtol=1e-9, atol=1e-9)
- 6-hour maximum time steps

The displayed deflection angles include both:

1. **Loeb's estimate**: Using the small-angle formula 2GM/(bv²)
2. **Exact two-body**: Computed from the full Keplerian hyperbola

## License

MIT License

## References

- Avi Loeb's analysis of 3I/ATLAS (December 17, 2025)
