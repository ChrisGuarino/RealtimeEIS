"""
JAX BASICS FOR ELECTROCHEMISTRY
=================================

Start here to understand the key JAX concepts before diving into the full simulator.
"""

import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
import time

print("="*70)
print("JAX BASICS TUTORIAL")
print("="*70)
print()

# =============================================================================
# CONCEPT 1: JIT COMPILATION (Just-In-Time)
# =============================================================================
print("1. JIT COMPILATION - Making Functions Fast")
print("-" * 70)

def slow_function(x):
    """Regular Python function - runs on CPU, not optimized"""
    return jnp.exp(x) * jnp.sin(x) + jnp.cos(x)**2

@jit
def fast_function(x):
    """JIT-compiled - compiles to optimized machine code"""
    return jnp.exp(x) * jnp.sin(x) + jnp.cos(x)**2

# Time comparison
x = jnp.linspace(0, 10, 10000)

start = time.time()
result_slow = slow_function(x)
time_slow = time.time() - start

start = time.time()
result_fast = fast_function(x)  # First call: compiles
time_fast_first = time.time() - start

start = time.time()
result_fast = fast_function(x)  # Second call: uses compiled version
time_fast_second = time.time() - start

print(f"Regular function: {time_slow*1000:.2f} ms")
print(f"JIT first call (with compilation): {time_fast_first*1000:.2f} ms")
print(f"JIT second call (compiled): {time_fast_second*1000:.2f} ms")
print(f"Speedup: {time_slow/time_fast_second:.1f}x faster!")
print()
print("KEY INSIGHT: JIT compiles once, then runs MUCH faster")
print()

# =============================================================================
# CONCEPT 2: VMAP (Vectorization)
# =============================================================================
print("2. VMAP - Automatic Vectorization")
print("-" * 70)

@jit
def compute_current_single(voltage):
    """Compute current for a SINGLE voltage"""
    i0 = 10.0
    alpha = 0.5
    F = 96485.0
    R = 8.314
    T = 298.0
    E_eq = -0.3
    
    eta = voltage - E_eq
    current = i0 * (jnp.exp(alpha * F * eta / (R * T)) - 
                    jnp.exp(-(1-alpha) * F * eta / (R * T)))
    return current

# Method 1: Loop (SLOW - don't do this)
voltages = jnp.linspace(-0.8, 0.2, 1000)
start = time.time()
currents_loop = jnp.array([compute_current_single(v) for v in voltages])
time_loop = time.time() - start

# Method 2: vmap (FAST - do this!)
compute_current_batch = jit(vmap(compute_current_single))
start = time.time()
currents_vmap = compute_current_batch(voltages)
time_vmap = time.time() - start

print(f"Loop method: {time_loop*1000:.2f} ms")
print(f"vmap method: {time_vmap*1000:.2f} ms")
print(f"Speedup: {time_loop/time_vmap:.1f}x faster!")
print()
print("KEY INSIGHT: vmap automatically vectorizes your function")
print("It's like numpy broadcasting but works with ANY function")
print()

# =============================================================================
# CONCEPT 3: GRAD (Automatic Differentiation)
# =============================================================================
print("3. GRAD - Automatic Differentiation")
print("-" * 70)

@jit
def loss_function(params):
    """
    Example loss function: predicting deposition uniformity
    params = [voltage, PEG_concentration]
    """
    voltage, PEG = params
    
    # Fake model: uniformity = 0.95 - (voltage + 0.5)² - 0.001*PEG
    target_voltage = -0.5
    target_PEG = 100.0
    
    uniformity = 0.95 - 0.5*(voltage - target_voltage)**2 - 0.0001*(PEG - target_PEG)**2
    
    # Loss: how far are we from perfect uniformity (1.0)?
    loss = (uniformity - 1.0)**2
    return loss

# Compute gradient automatically!
params = jnp.array([-0.4, 80.0])  # [voltage, PEG]

loss_value = loss_function(params)
gradient = grad(loss_function)(params)

print(f"Current parameters: voltage={params[0]:.2f}V, PEG={params[1]:.0f}ppm")
print(f"Loss: {loss_value:.6f}")
print(f"Gradient: dL/dV={gradient[0]:.4f}, dL/dPEG={gradient[1]:.6f}")
print()
print("The gradient tells us:")
if gradient[0] > 0:
    print("  - Decrease voltage to improve uniformity")
else:
    print("  - Increase voltage to improve uniformity")
if gradient[1] > 0:
    print("  - Decrease PEG to improve uniformity")
else:
    print("  - Increase PEG to improve uniformity")
print()
print("KEY INSIGHT: JAX automatically computes derivatives")
print("This is the foundation of optimization and ML training")
print()

# =============================================================================
# CONCEPT 4: JAX.LAX.SCAN (Replacing Loops)
# =============================================================================
print("4. JAX.LAX.SCAN - Efficient Loops")
print("-" * 70)

def update_concentration(carry, t):
    """
    Simulate concentration change over time during deposition.
    
    carry: (concentration, total_deposited)
    t: time step
    
    Returns: (new_carry, stored_value)
    """
    concentration, total_deposited = carry
    
    # Simple model: concentration decreases as copper deposits
    deposition_rate = 0.001 * concentration
    new_concentration = concentration - deposition_rate
    new_total = total_deposited + deposition_rate
    
    new_carry = (new_concentration, new_total)
    stored_value = new_concentration  # Store for later
    
    return new_carry, stored_value

# Initial conditions
initial_concentration = 1.0  # 1 M
initial_deposited = 0.0
initial_carry = (initial_concentration, initial_deposited)

# Run 100 time steps
time_steps = jnp.arange(100)

final_carry, concentration_history = jax.lax.scan(
    update_concentration,
    initial_carry,
    time_steps
)

final_concentration, total_deposited = final_carry

print(f"Initial concentration: {initial_concentration:.3f} M")
print(f"Final concentration: {final_concentration:.3f} M")
print(f"Total deposited: {total_deposited:.3f} M")
print(f"Concentration history shape: {concentration_history.shape}")
print()
print("KEY INSIGHT: scan is like a for-loop but:")
print("  - Compiles to fast code")
print("  - Returns all intermediate values")
print("  - Essential for time-series simulations")
print()

# =============================================================================
# CONCEPT 5: PUTTING IT ALL TOGETHER
# =============================================================================
print("5. COMPLETE EXAMPLE: Optimizing Deposition Parameters")
print("-" * 70)

@jit
def predict_uniformity(voltage, PEG_ppm):
    """
    Physics-informed model: predict deposition uniformity
    """
    # Target conditions for best uniformity
    optimal_voltage = -0.5
    optimal_PEG = 100.0
    
    # Physics: deviations from optimal reduce uniformity
    voltage_penalty = 0.5 * (voltage - optimal_voltage)**2
    PEG_penalty = 0.0001 * (PEG_ppm - optimal_PEG)**2
    
    uniformity = 0.95 - voltage_penalty - PEG_penalty
    return uniformity

@jit  
def loss_fn(params):
    """Loss function for optimization"""
    voltage, PEG = params
    uniformity = predict_uniformity(voltage, PEG)
    target_uniformity = 0.95
    return (uniformity - target_uniformity)**2

# Gradient descent optimization
params = jnp.array([-0.3, 50.0])  # Starting guess
learning_rate = 0.1

print("Optimizing parameters to achieve 95% uniformity...")
print()
for step in range(10):
    loss = loss_fn(params)
    grads = grad(loss_fn)(params)
    params = params - learning_rate * grads
    
    uniformity = predict_uniformity(params[0], params[1])
    
    if step % 2 == 0:
        print(f"Step {step}: V={params[0]:.3f}, PEG={params[1]:.1f}ppm, "
              f"uniformity={uniformity:.4f}, loss={loss:.6f}")

print()
print(f"Final optimized parameters:")
print(f"  Voltage: {params[0]:.3f} V")
print(f"  PEG: {params[1]:.1f} ppm")
print(f"  Predicted uniformity: {predict_uniformity(params[0], params[1]):.4f}")
print()

print("="*70)
print("SUMMARY: KEY JAX CONCEPTS")
print("="*70)
print("1. @jit      → Compile functions for speed")
print("2. vmap      → Vectorize over batch dimension")
print("3. grad      → Automatic differentiation")
print("4. lax.scan  → Efficient loops that compile")
print("5. Combine   → Fast physics-informed ML models")
print()
print("NEXT: Run electrochemistry_simulator.py for full Butler-Volmer simulation")
print("="*70)
