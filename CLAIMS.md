# Analysis of Loeb's 3I/ATLAS Claims

A systematic examination of claims from Avi Loeb's December 17, 2025 Medium post: ["3I/ATLAS Maintained a Sunward Jet After Its Gravitational Deflection by 16 Degrees at Perihelion"](https://avi-loeb.medium.com/3i-atlas-maintained-a-sunward-jet-after-its-gravitational-deflection-by-16-degrees-at-perihelion-e6810be9b3d8)

---

## Claim 1: Gravitational Deflection of 16.4°

### Loeb's Statement

> _"The direction of motion of 3I/ATLAS was shifted by the following angle (in radians) at perihelion: 2GM/(b·v²) = 0.286 = 16.4 degrees, where G is Newton's constant, M is the mass of the Sun, b=202 million kilometers is the perihelion distance and v=68 kilometers per second is the perihelion speed."_

### Analysis

**The Formula is Correct, But the Variable is Wrong**

The deflection formula θ = 2GM/(b·v²) is a standard result from classical scattering theory. However, in this formula:

| Variable | Correct Definition                                                       | Loeb's Usage                           |
| -------- | ------------------------------------------------------------------------ | -------------------------------------- |
| **b**    | Impact parameter (perpendicular distance to undeflected asymptotic path) | Perihelion distance (closest approach) |

**Impact parameter ≠ Perihelion distance**

```
                    ← b (impact parameter) →
                    ┌─────────────────────→  undeflected path
                    │                        (if no gravity)
                    │
                  ☀ Sun

                      ╲
                       ╲  actual curved path
                        ╲
                         • perihelion (rₚ)
```

The impact parameter is **larger** than perihelion because gravity bends the trajectory inward:

$$b = r_p \sqrt{1 + \frac{2GM}{r_p \cdot v^2}}$$

**Calculated Values:**

- Perihelion distance (rₚ): 202 million km (Loeb's value) ✓
- Impact parameter (b): ~239 million km (calculated)
- **Loeb's deflection (using rₚ):** 16.4°
- **Correct deflection (using b):** ~19°

### Verdict

⚠️ **Methodological Issue** — The formula requires impact parameter, not perihelion distance. The actual deflection is approximately 19°, not 16.4°.

---

## Claim 2: Jet Cone Width of 8° Matches Deflection Geometry

### Loeb's Statement

> _"The orientation of the jet varied within a cone with a half opening angle of 4 degrees..."_

And the implicit argument that:

- Deflection angle (~16°) ≈ 2 × jet cone full width (2 × 8°)
- Therefore the Sun stays within the cone throughout the flyby

### Analysis

**The Suspicious Coincidence**

| Value | Source                                  |
| ----- | --------------------------------------- |
| 16.4° | Loeb's calculated deflection            |
| 16°   | 2 × 8° cone width                       |
| ~19°  | Actual deflection (correct calculation) |

If using the correct impact parameter:

- Actual deflection (~19°) > 2 × cone width (16°)
- The geometry is **worse** than Loeb claims
- The Sun exits the cone faster

**Does This Look Like Working Backwards?**

1. Desired conclusion: deflection = 2 × cone angle (geometry "works")
2. Need deflection ≈ 16° to match 2 × 8°
3. Using rₚ instead of b conveniently yields ~16°

This may be coincidence or an alternative convention, but the numerical convenience is notable.

### Verdict

⚠️ **Questionable** — The 16° ≈ 2×8° relationship relies on the incorrect deflection calculation. With correct physics, the match disappears.

---

## Claim 3: No Active Steering Required (Fixed-Axis Jet)

### Loeb's Statement

> _"Even though the jet wobbles by ±4 degrees every 7.74 hours, its mean orientation relative to the spin axis of 3I/ATLAS is fixed."_

The core argument: A simple fixed-axis jet (no Sun-tracking) maintains sunward orientation throughout the flyby due to the geometric relationship between deflection angle and cone width.

### Analysis

**Simulation Results**

Running the full orbital trajectory with Loeb's parameters:

| Metric                    | Result                  |
| ------------------------- | ----------------------- |
| Sun inside +A cone        | **13.6%** of trajectory |
| Sun inside -A cone        | **0%** of trajectory    |
| Sun outside BOTH cones    | **~86%** of trajectory  |
| Longest gap (Sun outside) | 642 frames (~156 days)  |

**Why the Geometry Fails**

Loeb's argument only works at **one moment** (perihelion):

```
Before perihelion:        At perihelion:         After perihelion:

    ↘ velocity              → velocity              ↗ velocity
     ╲                       ─                       ╱
      ╲                      │                      ╱
       ☀                     ☀                     ☀

   Sun NOT in cone      Sun briefly in cone    Sun NOT in cone
```

During an orbit spanning months:

- Velocity direction changes continuously (~19° total turn)
- Fixed jet axis stays fixed
- Sun spends most of the time outside the cone

### Verdict

❌ **Refuted** — The simulation shows the fixed-axis geometry fails for 86% of the trajectory. If a jet theory were true, it would require active Sun-tracking (steering), contradicting Loeb's "simple fixed jet" claim.

---

## Claim 4: Lightsail Propulsion Mechanism

### Loeb's Statement

The broader hypothesis (from this and previous articles): 3I/ATLAS and 'Oumuamua exhibit anomalous acceleration consistent with radiation pressure on a lightsail — an artificial structure pushed by sunlight.

### Analysis

**How a Lightsail Would Work**

```
    Sunlight (photons)
    ☀ ══════════════▶ 💨 ════▶
    Sun              Sail    Push

    Photons hit reflective surface → bounce off → transfer momentum
```

For this to work:

1. The reflective surface must **face the Sun**
2. The "jet" direction (push) would always be **anti-sunward**
3. Orientation must be maintained throughout the flyby

**The Problem**

If the lightsail has a fixed orientation (Claim 3):

- It only faces the Sun ~14% of the time
- The other 86% it would receive reduced/no thrust
- Acceleration would be highly variable, not consistent

If the lightsail actively tracks the Sun:

- This requires steering mechanisms
- Far more complex than Loeb's "simple fixed geometry" suggests
- Raises question: why would aliens use such a system?

### Verdict

⚠️ **Unsupported** — The geometric claims that would make a "dumb" lightsail plausible don't hold up. A functional lightsail would need active steering, which is a much stronger claim about alien technology.

---

## Claim 5: The 7.74-Hour Wobble Period

### Loeb's Statement

> _"The period of the wobble is 7.74 hours..."_

### Analysis

This value is taken from observational data and cited preprints. The simulation implements this wobble.

**What the Wobble Means**

The jet cone oscillates ±4° around its mean axis every 7.74 hours:

```
         ╱ +4° max
        ╱
    ───●─── mean axis
        ╲
         ╲ -4° max

    One cycle = 7.74 hours
```

**Impact on Geometry**

The wobble provides a small additional angular range (±4°), but:

- It doesn't change the fundamental problem
- Sun is still outside the wobble range 86% of the time
- The 8° total cone width (4° half-angle) is already insufficient

### Verdict

✅ **Implemented** — The simulation correctly animates the wobble. It doesn't save the geometric argument.

---

## Claim 6: Jet Length of ~10⁶ km

### Loeb's Statement

> _"...the jet of 3I/ATLAS spans a characteristic scale of at least a million kilometers"_

### Analysis

This is an observational claim about the visible extent of outgassing/jet activity. The simulation visualizes this (scaled 60× for visibility).

This claim is independent of the geometric arguments and doesn't affect the core analysis.

### Verdict

✅ **Visualized** — Rendered in simulation at appropriate scale.

---

## Summary Table

| #   | Claim                | Verdict         | Issue                                  |
| --- | -------------------- | --------------- | -------------------------------------- |
| 1   | Deflection = 16.4°   | ⚠️ Questionable | Wrong variable in formula; actual ~19° |
| 2   | 16° = 2 × 8° cone    | ⚠️ Questionable | Only works with incorrect deflection   |
| 3   | No steering needed   | ❌ Refuted      | Sun outside cones 86% of trajectory    |
| 4   | Lightsail propulsion | ⚠️ Unsupported  | Would require active tracking          |
| 5   | 7.74h wobble period  | ✅ Implemented  | Correct, but doesn't fix geometry      |
| 6   | Jet length ~10⁶ km   | ✅ Visualized   | Observational claim, not disputed      |

---

## Conclusion

The central geometric argument — that a fixed-axis jet naturally maintains sunward orientation throughout 3I/ATLAS's flyby — does not survive quantitative analysis.

**Key Findings:**

1. The deflection angle calculation appears to use perihelion distance where impact parameter is required, yielding 16.4° instead of ~19°

2. The convenient match between deflection (~16°) and twice the cone width (2×8°) disappears when using correct physics

3. Even using Loeb's own parameters, a fixed-axis jet would point away from the Sun for **86% of the trajectory**

4. A functional lightsail would require active Sun-tracking, which is a far stronger (and unsupported) claim about alien technology

The simulation provides a tool for anyone to verify these findings by visualizing the actual orbital geometry with Loeb's stated parameters.

---

## References

- Loeb, A. (2025). ["3I/ATLAS Maintained a Sunward Jet After Its Gravitational Deflection by 16 Degrees at Perihelion"](https://avi-loeb.medium.com/3i-atlas-maintained-a-sunward-jet-after-its-gravitational-deflection-by-16-degrees-at-perihelion-e6810be9b3d8). Medium, December 17, 2025.

- Simulation source code: [github.com/kszpirak/3iatlas-animation](https://github.com/kszpirak/3iatlas-animation)
