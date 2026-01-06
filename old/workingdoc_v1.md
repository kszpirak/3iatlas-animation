# 3I/ATLAS Animation v1 - Working Document

**File:** `main-gpt5.2_v1.py`  
**Created:** 2026-01-05  
**Source:** Loeb's Dec 17, 2025 Medium post on "Six Anomalies of 3I/ATLAS"

---

## Key Correction: Loeb's Actual Geometric Claim

The current animation **forces the jet direction to be sunward at all times**. This is NOT what Loeb claims.

### Loeb's Actual Argument (Dec 17, 2025)

1. **Trajectory deflected by ~16.4°** at perihelion using `2GM/(bv²)` with b = 202 million km and v = 68 km/s
2. **Anti-tail jet spans ~8°** and extends to ~1 million km (Dec 15 imagery)
3. **The 16.4° ≈ 2 × 8° coincidence** enables a fixed rotation/jet geometry to still overlap the sunward direction before AND after perihelion:
   - Pre-perihelion: sunward direction lies on one edge of the cone
   - Post-perihelion: sunward direction lies on the opposite-pole cone edge

**The "always sunward" is NOT "the jet actively steers itself to the Sun"** — it's "given a fixed axis + cone geometry, the Sun stays inside the cone before and after because the orbit turned by ~2× the cone angle."

---

## Required Changes

### 1. CLAIMS Text - Rewrite to Match Loeb Accurately

#### Claim 1 — Deflection and 2× Cone Geometry

**Loeb-accurate phrasing:**

- "At perihelion, the direction of motion of 3I/ATLAS is shifted by 2GM/(bv²) = 0.286 rad = 16.4°, using b = 202 million km and v = 68 km/s."
- "This deflection angle is twice the opening angle of the anti-tail jet (~8°)."
- "If one edge of the jet cone overlapped the sunward direction before perihelion, then an identical cone on the opposite pole can overlap the sunward direction after perihelion."

**Visualization needed:**

- Two cones: one centered on fixed axis +A (pre), one centered on −A (post)
- Show Sun direction vector and whether it falls inside active cone
- Show measured/assumed cone angle (8°) and gravitational turn (16.4°)

#### Claim 2 — Jet Width + Physical Scale

**Loeb-accurate phrasing:**

- Dec 15 image "shows a prominent tightly-collimated anti-tail jet… field of view spans 1.6 × 0.7 million km… the jet spans about 8° out to a distance of order a million km."

**Visualization needed:**

- Jet length should correspond to ~1,000,000 km ≈ 0.0067 AU (NOT 0.60 AU!)
- If keeping longer jet for visibility: label as "exaggerated length (×N)" and show real scale

#### Claim 3 — Wobble Constraints

**Important:** The 7.74h and ±4° values are NOT stated in the Dec 17 post — Loeb references other sources. Label as "observational report (per cited preprint)" and keep as parameters.

**Loeb's Dec 17 phrasing:**

- The observed wobble of the pre-perihelion sunward jet "requires the base of the jet to be within 8° from the sun-facing pole."

#### Claim 4 — Maintaining Sunward Anti-Tail Before & After Perihelion

**Loeb-accurate phrasing:**

- A prominent tightly collimated sunward anti-tail appears before AND after perihelion
- Persistence is part of "geometric coincidences"
- The "mechanism" is that a physical sunward jet stays tightly collimated to ~million km scales

**Visualization needed:**

- Show pre and post phases separately
- Show same fixed axis and how cone overlap switches to opposite pole

#### Claim 5 — Perihelion Parameters

Keep current orbital parameters, they're accurate.

#### Claim 6 — Ecliptic Alignment

**Problem:** Current sim is 2D in orbital plane. Drawing y=0 line has no physical link to real ecliptic unless coordinate system is explicitly defined.

**Options:**

1. Promote to 3D with z-coordinate and ~5° inclination
2. Rename to "(Not modeled in 2D) Ecliptic alignment ±5°"

---

### 2. Core Physics Changes

#### 2.1 Stop Forcing Jet to Be Sunward (HIGH PRIORITY)

**Current code (WRONG):**

```python
sunward = -r_vec / r
# then rotate a bit → always sunward by construction
```

**Needed: Fixed body axis + jet cone**

Implementation:

1. Choose fixed inertial axis `A_hat` (unit vector), e.g., aligned with sunward direction at far pre-perihelion time
2. Jet direction at time t generated around `A_hat` with wobble:
   - Pre-perihelion (t < 0): cone centered on +A_hat
   - Post-perihelion (t > 0): cone centered on −A_hat (opposite pole)
3. Compute whether Sun direction lies inside cone:
   ```python
   angle_to_sun = acos(dot(cone_axis, sunward_hat))
   inside_cone = angle_to_sun <= half_angle
   ```

This demonstrates Loeb's "16.4° is twice 8° so sunward direction can remain within one cone edge pre and opposite cone edge post."

#### 2.2 Make Jet Length Physically Consistent

**Current:** `self.jet_len_au = 0.60` (way too long!)

**Loeb references:** ~1 million km scales

```python
JET_LEN_KM = 1_000_000
JET_LEN_AU_REAL = JET_LEN_KM * 1000 / AU  # ≈ 0.0067 AU
VIS_SCALE = 90  # exaggeration factor for visibility
JET_LEN_AU_VIS = JET_LEN_AU_REAL * VIS_SCALE
```

Annotate `VIS_SCALE` in corner of visualization.

#### 2.3 Make Deflection Arc Match Actual Trajectory

**Current:** Arc drawn at fixed angles (theta1=70...) — NOT tied to trajectory!

**Needed:**

1. Compute inbound velocity direction at far pre-perihelion (first frame)
2. Compute outbound velocity direction at far post-perihelion (last frame)
3. Deflection angle = `acos(dot(v_in_hat, v_out_hat))` in degrees
4. Draw arc between those actual directions

This shows whether chosen `RP_KM` and `VP_KMS` actually produce ~16.4°.

#### 2.4 Separate Visual Jet from Dynamical Acceleration

**Current:** Implicitly asserts jet direction = acceleration direction (extra assumption!)

**Loeb describes:** Observed feature direction, not necessarily force direction.

**Options:**

- Mode A: "Visual-only jet (no force)" — trajectory purely gravitational
- Mode B: "Jet-reaction acceleration" — force opposite displayed jet

For accuracy, keep trajectory primarily gravitational (2-body) and add toggle.

#### 2.5 Ecliptic Alignment in 2D

**Current:** Drawing y=0 line is meaningless in 2D orbital plane sim.

**Options:**

1. Promote to 3D: state = (x, y, z, vx, vy, vz), define ecliptic (z=0), set initial inclination ~5°
2. Rename claim 6 to "(Not modeled) Ecliptic alignment ±5°"

---

## Implementation Order

1. **[CRITICAL]** Refactor jet to fixed axis + opposite pole switch at perihelion + sun-in-cone indicator
2. Update CLAIMS text to match Loeb accurately
3. Fix jet length to physical scale with exaggeration label
4. Fix deflection arc to match actual computed trajectory
5. Update claim 6 to note 2D limitation
6. (Optional) Add toggle for jet force vs visual-only mode

---

## Work Log

### 2026-01-05 — Session Start

**Reading complete.** Current code analysis:

- Lines 380-405: `jet_hat` computed by rotating `sunward_hat` — WRONG, forces sunward
- Line 460: `self.jet_len_au = 0.60` — should be ~0.0067 AU (×90 exaggerated)
- Lines 827-830: Deflection arc at fixed theta1=70 — should use actual trajectory directions

**Starting implementation...**

---

### Implementation Complete — Key Results

**Changes Made:**
1. ✅ Fixed-axis jet geometry implemented (not always-sunward)
2. ✅ Axis switches from +A to -A at perihelion
3. ✅ Sun-in-cone indicator added
4. ✅ Computed deflection shown vs Loeb's claim
5. ✅ Jet length scaled to physical value (~1M km) with ×60 exaggeration
6. ✅ Claims text updated to match Loeb's Dec 17, 2025 post accurately
7. ✅ Claim 6 (ecliptic) marked as "NOT MODELED" in 2D
8. ✅ Deflection arc now shows actual trajectory asymptotes

**Critical Finding:**
```
Computed deflection angle: 17.33° (Loeb claims: 16.4°)
Sun in cone: 57/925 frames (6.2%)
```

**Interpretation:**
With a simple fixed-axis model where the jet axis is set to the far pre-perihelion sunward direction:
- The computed deflection (17.33°) is close to but not exactly 16.4°
- **The Sun is only in the cone for 6.2% of frames!**

This reveals an important subtlety in Loeb's argument:
- Loeb's claim requires that the fixed axis be specifically tuned
- The axis must be set such that the Sun just grazes the cone edges before AND after perihelion
- Our simple model uses far-pre-perihelion sunward direction, which isn't optimal

**Possible next steps:**
1. Optimize the fixed axis angle to maximize Sun-in-cone time
2. Add cone boundary visualization to show Sun position relative to cone edges
3. Show the "two cones" (±A) simultaneously to illustrate Loeb's geometry

---

### Code Test Output
```
Computing trajectory...
Trajectory computed: 925 frames, perihelion frame 458
Computed deflection angle: 17.33° (Loeb claims: 16.4°)
Sun in cone: 57/925 frames (6.2%)
```

---

### Final Status: Ready for Use

**Simulation runs successfully.** All major changes implemented:

1. **Fixed-axis jet geometry** — jet axis is now fixed in inertial space, not always-sunward
2. **Pole switching at perihelion** — +A axis pre-perihelion, -A post-perihelion
3. **Sun-in-cone indicator** — shows whether geometry actually keeps Sun in cone
4. **Physical jet scale** — 1M km real, ×60 exaggerated for visibility (0.40 AU displayed)
5. **Computed deflection** — 17.33° actual vs 16.4° Loeb claim
6. **Trajectory-based deflection arc** — shows actual asymptote directions
7. **Updated claims text** — matches Loeb's Dec 17, 2025 Medium post
8. **Ecliptic warning** — Claim 6 marked as "NOT MODELED" in 2D

**Key insight from simulation:**
The simple fixed-axis model shows Sun in cone only 6.2% of the time. Loeb's geometry requires a specifically tuned axis orientation — not just "sunward at far pre-perihelion" — to maximize Sun-in-cone time. This could be a future enhancement.

**Run with:**
```bash
cd /Users/konrad/src/3iatlas-animation
source .venv/bin/activate
python3 main-gpt5.2_v1.py
```
