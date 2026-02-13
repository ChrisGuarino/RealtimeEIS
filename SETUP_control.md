# Setup Guide - Real-Time Electrodeposition Control

Complete installation and configuration guide for the real-time control system.

---

## Prerequisites

### Required Hardware
- ✅ Gamry Potentiostat (Reference 3000 or equivalent)
- ✅ Windows computer (COM interface requirement)
- ✅ Electrochemical cell setup

### Required Software
- ✅ Windows 10 or 11
- ✅ Python 3.10 or 3.11
- ✅ Gamry Framework (from Gamry Instruments)

---

## Step-by-Step Installation

### 1. Install Gamry Framework

**Download from Gamry:**
1. Go to https://www.gamry.com/support/software-downloads/
2. Download latest Framework version
3. Install with Administrator privileges
4. Restart computer

**Verify installation:**
```bash
# Should see Gamry folder
dir "C:\Program Files\Gamry Instruments\Framework"
```

### 2. Install Python

**Option A: From python.org (Recommended)**
```bash
# Download Python 3.11 from python.org
# During install: CHECK "Add Python to PATH"
# Verify:
python --version  # Should show 3.11.x
```

**Option B: Using Anaconda**
```bash
# Download from anaconda.com
conda create -n control_env python=3.11
conda activate control_env
```

### 3. Create Project Directory

```bash
# Create and navigate to project folder
mkdir realtime-control
cd realtime-control

# Create virtual environment
python -m venv control_env

# Activate (Windows)
control_env\Scripts\activate
```

### 4. Install Python Dependencies

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# This will install:
# - JAX (CPU version)
# - NumPy, SciPy
# - Matplotlib, Pandas
# - pywin32 (for Gamry COM)
# - And all other dependencies
```

**If pywin32 fails:**
```bash
# Manual pywin32 installation
pip install --upgrade setuptools wheel
pip install pywin32

# Post-install script (run as Administrator)
python Scripts\pywin32_postinstall.py -install
```

### 5. Register Gamry COM Interface

```bash
# Open Command Prompt as Administrator
# Navigate to Gamry Framework folder
cd "C:\Program Files\Gamry Instruments\Framework"

# Register COM library
regsvr32 GamryCOM.dll

# Should see "DllRegisterServer succeeded" message
```

### 6. Test Installation

**Quick test script:**
```python
# test_installation.py
import sys

print("Testing installation...")
print("-" * 60)

# Test 1: Python version
print(f"Python version: {sys.version}")
assert sys.version_info >= (3, 10), "Python 3.10+ required"
print("✓ Python version OK")

# Test 2: JAX
try:
    import jax
    import jax.numpy as jnp
    print(f"✓ JAX installed: {jax.__version__}")
    
    # Test JIT compilation
    @jax.jit
    def test_jit(x):
        return jnp.sum(x ** 2)
    
    result = test_jit(jnp.arange(100))
    print(f"✓ JAX JIT working (result: {result})")
except Exception as e:
    print(f"✗ JAX failed: {e}")
    sys.exit(1)

# Test 3: NumPy/SciPy
try:
    import numpy as np
    import scipy
    print(f"✓ NumPy: {np.__version__}")
    print(f"✓ SciPy: {scipy.__version__}")
except Exception as e:
    print(f"✗ Scientific libraries failed: {e}")
    sys.exit(1)

# Test 4: Gamry COM
try:
    import win32com.client
    print("✓ win32com installed")
    
    # Try to dispatch Gamry COM
    try:
        gamry = win32com.client.Dispatch("GamryCOM.GamryPstat")
        print("✓ Gamry COM interface available")
        print(f"  Found {gamry.Count()} potentiostat(s)")
    except Exception as e:
        print(f"⚠ Gamry COM registered but no devices found")
        print(f"  (This is OK if Gamry is not connected)")
except Exception as e:
    print(f"✗ Gamry COM failed: {e}")
    print("  → Run as Administrator and register DLL")
    sys.exit(1)

# Test 5: Other dependencies
try:
    import matplotlib
    import pandas
    import scipy.optimize
    print("✓ Matplotlib, Pandas installed")
except Exception as e:
    print(f"✗ Dependencies missing: {e}")
    sys.exit(1)

print("-" * 60)
print("✓✓✓ ALL TESTS PASSED ✓✓✓")
print("Ready to run real-time control system!")
```

**Run test:**
```bash
python test_installation.py
```

Expected output:
```
Testing installation...
------------------------------------------------------------
Python version: 3.11.5 ...
✓ Python version OK
✓ JAX installed: 0.4.20
✓ JAX JIT working (result: 328350)
✓ NumPy: 1.24.3
✓ SciPy: 1.11.2
✓ win32com installed
✓ Gamry COM interface available
  Found 1 potentiostat(s)
✓ Matplotlib, Pandas installed
------------------------------------------------------------
✓✓✓ ALL TESTS PASSED ✓✓✓
Ready to run real-time control system!
```

---

## Hardware Setup

### Electrochemical Cell Configuration

**Components:**
```
┌─────────────────────────────────────┐
│     Electrochemical Cell            │
│                                     │
│  Counter (Pt mesh)                  │
│       │                             │
│  Working (Cu substrate)             │
│       │                             │
│  Reference (Hg/Hg₂SO₄)             │
└─────────────────────────────────────┘
        │
        ↓
   Gamry Potentiostat
        │
        ↓
   Computer (USB)
```

**Electrolyte preparation:**
1. 1.0 M CuSO₄·5H₂O in DI water
2. 0.5 M H₂SO₄ (for conductivity)
3. Add PEG (typically 50-200 ppm)
4. Add Cl⁻ (typically 50-100 ppm as NaCl or HCl)

**Safety:**
- Wear gloves and safety glasses
- Work in ventilated area
- Have acid spill kit available
- Know location of eyewash station

### Gamry Connection

1. Connect electrodes to Gamry:
   - **Working (green)** → Copper substrate
   - **Counter (red)** → Platinum mesh
   - **Reference (white)** → Hg/Hg₂SO₄ or Ag/AgCl
   - **Ground (black)** → Cell ground (optional)

2. Connect Gamry to computer (USB)

3. Power on Gamry

4. Verify in Gamry Framework software:
   - Open "Framework"
   - Click "Experiment" → "Physical Electrochemistry" → "Open Circuit Potential"
   - Should see stable OCP (~0.3V for Cu²⁺/Cu vs. SHE)

---

## First Run - Baseline Deposition

Before using real-time control, run a baseline deposition to verify everything works.

### 1. Prepare Electrolyte

```python
# Electrolyte composition
CuSO4 = 1.0 M      # 249.7 g/L CuSO₄·5H₂O
H2SO4 = 0.5 M      # 24.5 mL/L concentrated H₂SO₄
PEG = 100 ppm      # ~0.1 g/L PEG-3350
Cl = 50 ppm        # ~0.05 g/L NaCl
```

### 2. Check Open Circuit Potential (OCP)

```python
from gamry_interface import GamryRealtimeController

# Initialize
gamry = GamryRealtimeController(device_id=0)

# Measure OCP
ocp = gamry.measure_ocp(duration=30)  # 30 seconds
print(f"Open Circuit Potential: {ocp:.3f} V")

# Should be around -0.3V vs. Hg/Hg₂SO₄
# (or +0.34V vs. SHE for Cu²⁺/Cu)
```

### 3. Run Test EIS

```python
# Quick EIS to verify cell is working
frequencies, Z_data = gamry.measure_eis_fast()

# Plot Nyquist
import matplotlib.pyplot as plt
plt.plot(Z_data.real, -Z_data.imag, 'o-')
plt.xlabel('Z_real (Ω·cm²)')
plt.ylabel('-Z_imag (Ω·cm²)')
plt.axis('equal')
plt.title('Test EIS - Verify Cell')
plt.show()

# Should see semicircle (Randles circuit)
```

### 4. Run Baseline Deposition (No Control)

```python
from scripts.run_baseline import run_baseline_deposition

log = run_baseline_deposition(
    voltage=-0.5,          # Fixed voltage
    duration_seconds=60,   # Short test (1 minute)
    PEG_ppm=100.0,
    Cl_ppm=50.0
)

# Check the log
print(f"Average current: {np.mean(log['current']):.2f} mA")
print(f"Total charge: {np.trapz(log['current'], log['time']):.2f} mA·s")
```

### 5. Measure Uniformity

After deposition:
1. Remove sample, rinse with DI water
2. Dry with N₂
3. Measure thickness at 5-10 points (profilometry or XRF)
4. Calculate uniformity: `1 - (σ / μ)`

**Record in lab notebook:**
- Voltage, current, time
- PEG/Cl concentrations
- Measured uniformity
- Any observations (bubbles, color, etc.)

---

## Running Controlled Deposition

Once baseline works, try real-time control:

### 1. Train Initial Model (Optional)

If you have 5-10 baseline experiments:
```python
from models.train_model import train_uniformity_predictor

model_params = train_uniformity_predictor(
    data_file='data/baseline_experiments.csv',
    n_epochs=1000
)

# Save
import pickle
with open('models/uniformity_predictor.pkl', 'wb') as f:
    pickle.dump(model_params, f)
```

If you don't have training data yet, use physics-only model (built-in).

### 2. Run Controlled Deposition

```python
from scripts.run_controlled import run_controlled_deposition

log = run_controlled_deposition(
    initial_voltage=-0.5,
    target_uniformity=0.95,
    duration_seconds=300,    # 5 minutes
    control_interval=10.0,   # Adjust every 10s
    PEG_ppm=100.0,
    Cl_ppm=50.0
)

# Log contains time series of:
# - voltage (adjusted by controller)
# - current
# - R_ct (from EIS)
# - predicted_uniformity
# - control_action (delta_voltage)
```

### 3. Analyze Results

```python
from scripts.analyze_results import compare_baseline_vs_controlled

compare_baseline_vs_controlled(
    baseline_file='data/raw/baseline_001.json',
    controlled_file='data/raw/controlled_001.json'
)

# Generates plots:
# - Voltage profiles
# - Current profiles  
# - R_ct evolution
# - Uniformity comparison
```

---

## Troubleshooting

### Common Issues

#### Issue: "No potentiostats found"

**Cause:** Gamry not connected or COM not registered

**Solution:**
1. Check USB connection
2. Verify Gamry is powered on
3. Register COM as Administrator:
   ```bash
   cd "C:\Program Files\Gamry Instruments\Framework"
   regsvr32 GamryCOM.dll
   ```

#### Issue: "Control loop too slow (>10s)"

**Cause:** JAX not using JIT, or too many EIS frequencies

**Solution:**
1. Reduce EIS frequencies from 10 to 5:
   ```python
   gamry.measure_eis_fast(n_points=5)  # Faster
   ```
2. Verify JAX JIT is working:
   ```python
   import jax
   @jax.jit
   def test(x):
       return x ** 2
   
   # First call: slow (compile)
   test(1.0)
   
   # Second call: fast (compiled)
   %timeit test(1.0)  # Should be <1µs
   ```

#### Issue: "Predicted uniformity is nonsense"

**Cause:** Model not trained on your data

**Solution:**
1. Run 5-10 baseline experiments
2. Measure actual uniformity for each
3. Train model on this data
4. Alternatively, use physics-only predictor (no ML)

#### Issue: "Cell voltage unstable during control"

**Cause:** PID gains too aggressive, or electrical noise

**Solution:**
1. Reduce Kp, Ki, Kd gains:
   ```python
   controller = UniformityController(
       Kp=0.05,  # Reduce from 0.1
       Ki=0.005, # Reduce from 0.01  
       Kd=0.02   # Reduce from 0.05
   )
   ```
2. Add voltage change rate limit (already implemented)
3. Check ground connections

#### Issue: "Import error: No module named 'win32com'"

**Cause:** pywin32 not properly installed

**Solution:**
```bash
pip install --upgrade pywin32
python Scripts\pywin32_postinstall.py -install
```

---

## Performance Tuning

### JAX Optimization

**Enable 64-bit precision (if needed):**
```python
import jax
jax.config.update("jax_enable_x64", True)
```

**Check compilation:**
```python
from jax import jit

@jit
def predict_uniformity(features):
    # Your model
    pass

# First call: compile (slow)
result = predict_uniformity(test_features)

# Subsequent calls: fast
%timeit predict_uniformity(test_features)
# Should be <5ms
```

### Control Loop Optimization

**Reduce EIS overhead:**
```python
# Instead of 10 frequencies (100ms):
frequencies = np.logspace(3, 0, 10)  # 1000 Hz to 1 Hz

# Use 5 frequencies (50ms):
frequencies = np.logspace(3, 0, 5)   # Faster, still works
```

**Increase control interval if needed:**
```python
# 10s intervals (standard)
run_controlled_deposition(control_interval=10.0)

# 15s intervals (more margin, less frequent updates)
run_controlled_deposition(control_interval=15.0)
```

---

## Next Steps

### After Successful Installation

1. **Run baseline depositions** (5-10 experiments)
2. **Measure uniformity** for each
3. **Train predictive model** on your data
4. **Run controlled depositions** (5-10 experiments)
5. **Compare results** (baseline vs. controlled)
6. **Tune controller** if needed (adjust Kp, Ki, Kd)
7. **Publish results!**

### Recommended Experimental Design

**Week 1: Baseline**
- 10 depositions at different voltages (-0.4V to -0.6V)
- Fixed PEG (100 ppm), varied voltage
- Measure uniformity for all

**Week 2: Model Training**
- Train JAX predictor on Week 1 data
- Cross-validate (80/20 split)
- Target: RMSE < 0.05

**Week 3: Controlled**
- 10 controlled depositions (same voltage range)
- Compare to baseline
- Statistical analysis (t-test)

**Week 4: Publication**
- Write up results
- Create visualizations
- Submit to conference or journal

---

## Resources

### Documentation
- `README.md` - Project overview
- `docs/THEORY.md` - Electrochemistry background
- `docs/TUNING.md` - Controller tuning guide
- `docs/API.md` - API reference

### Example Scripts
- `scripts/run_baseline.py` - Open-loop deposition
- `scripts/run_controlled.py` - Closed-loop deposition
- `scripts/analyze_results.py` - Data analysis
- `scripts/tune_controller.py` - PID tuning utility

### Jupyter Notebooks
- `notebooks/01_baseline_analysis.ipynb` - Baseline results
- `notebooks/02_control_performance.ipynb` - Control evaluation
- `notebooks/03_model_validation.ipynb` - Model accuracy

### External Resources
- Gamry Application Notes: https://www.gamry.com/application-notes/
- JAX Documentation: https://jax.readthedocs.io/
- Electrochemistry: Bard & Faulkner textbook

---

## Getting Help

**If you're stuck:**
1. Check troubleshooting section above
2. Review example scripts
3. Run test_installation.py
4. Check Gamry Framework logs
5. Open issue on GitHub

**Common mistakes:**
- ❌ Not running as Administrator (COM registration)
- ❌ Forgetting to activate virtual environment
- ❌ Using wrong Gamry device ID
- ❌ Not training model on YOUR data

---

**Ready to start?** Run `python test_installation.py` and verify all tests pass!
