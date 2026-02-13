"""
JAX-Based Electrochemistry Simulator
====================================

Starting simple with Butler-Volmer, then building up to realistic copper deposition.

Author: Chris Guarino
"""

import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict

# =============================================================================
# PART 1: FUNDAMENTAL CONSTANTS
# =============================================================================

# Physical constants
F = 96485.0      # Faraday constant (C/mol)
R = 8.314        # Gas constant (J/(mol·K))
T_ROOM = 298.15  # Room temperature (K)


# =============================================================================
# PART 2: BASIC BUTLER-VOLMER EQUATION
# =============================================================================

@jit
def butler_volmer_simple(
    voltage: float,
    i0: float = 10.0,          # Exchange current density (mA/cm²)
    alpha: float = 0.5,        # Transfer coefficient
    E_eq: float = 0.0,         # Equilibrium potential (V)
    T: float = T_ROOM          # Temperature (K)
) -> float:
    """
    Basic Butler-Volmer equation for electrode kinetics.
    
    The Butler-Volmer equation describes how current density depends on 
    overpotential (voltage deviation from equilibrium).
    
    i = i0 * [exp(αₐFη/RT) - exp(-αᵤFη/RT)]
    
    Where:
    - i0: exchange current density (kinetic facility of the reaction)
    - α: transfer coefficient (symmetry factor, usually ~0.5)
    - F: Faraday constant
    - η: overpotential (V - E_eq)
    - R: gas constant
    - T: temperature
    
    Parameters:
    -----------
    voltage : Applied voltage (V)
    i0 : Exchange current density (mA/cm²)
    alpha : Anodic transfer coefficient
    E_eq : Equilibrium potential (V)
    T : Temperature (K)
    
    Returns:
    --------
    current_density : Current density (mA/cm²)
    """
    # Overpotential
    eta = voltage - E_eq
    
    # Anodic and cathodic contributions
    alpha_c = 1.0 - alpha  # Cathodic transfer coefficient
    
    # Butler-Volmer equation
    i_anodic = jnp.exp(alpha * F * eta / (R * T))
    i_cathodic = jnp.exp(-alpha_c * F * eta / (R * T))
    
    current_density = i0 * (i_anodic - i_cathodic)
    
    return current_density


# =============================================================================
# PART 3: CYCLIC VOLTAMMETRY (CV) SIMULATION
# =============================================================================

@jit
def generate_cv_waveform(
    v_start: float,
    v_end: float,
    scan_rate: float,  # V/s
    n_points: int = 1000
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate triangular voltage waveform for cyclic voltammetry.
    
    CV sweeps voltage linearly from v_start to v_end and back.
    
    Parameters:
    -----------
    v_start : Starting voltage (V)
    v_end : End voltage (V)
    scan_rate : Scan rate (V/s)
    n_points : Number of data points
    
    Returns:
    --------
    time, voltage : Arrays of time and voltage
    """
    # Calculate sweep duration
    delta_v = jnp.abs(v_end - v_start)
    t_sweep = delta_v / scan_rate
    t_total = 2 * t_sweep  # Forward + reverse
    
    # Time array
    time = jnp.linspace(0, t_total, n_points)
    
    # Voltage waveform (triangle wave)
    # Forward sweep: v_start → v_end
    # Reverse sweep: v_end → v_start
    voltage = jnp.where(
        time <= t_sweep,
        v_start + scan_rate * time,                    # Forward
        v_end - scan_rate * (time - t_sweep)          # Reverse
    )
    
    return time, voltage


def simulate_cv_simple(
    v_start: float = -0.8,
    v_end: float = 0.2,
    scan_rate: float = 0.05,   # V/s
    i0: float = 10.0,          # mA/cm²
    alpha: float = 0.5,
    E_eq: float = -0.3,        # Cu²⁺/Cu equilibrium at pH ~0
    n_points: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a simple cyclic voltammogram.
    
    Returns:
    --------
    time, voltage, current : Arrays for plotting
    """
    # Generate voltage waveform
    time, voltage = generate_cv_waveform(v_start, v_end, scan_rate, n_points)
    
    # Compute current at each voltage (vectorized with vmap)
    current_fn = lambda v: butler_volmer_simple(v, i0, alpha, E_eq)
    current = vmap(current_fn)(voltage)
    
    return np.array(time), np.array(voltage), np.array(current)


# =============================================================================
# PART 4: ADDING MASS TRANSPORT (More Realistic)
# =============================================================================

@jit
def butler_volmer_with_mass_transport(
    voltage: float,
    C_bulk: float = 1.0,       # Bulk concentration (M)
    i0: float = 10.0,
    alpha: float = 0.5,
    E_eq: float = -0.3,
    n_electrons: int = 2,      # Cu²⁺ → Cu (2 electrons)
    D: float = 7e-6,           # Diffusion coefficient (cm²/s)
    delta: float = 0.01,       # Diffusion layer thickness (cm)
    T: float = T_ROOM
) -> float:
    """
    Butler-Volmer with mass transport limitation.
    
    At high current densities, the reaction becomes limited by how fast
    Cu²⁺ ions can diffuse to the electrode surface.
    
    The limiting current is:
    i_lim = n * F * D * C_bulk / δ
    
    The actual current is limited by min(i_kinetic, i_lim)
    """
    # Limiting current (mass transport controlled)
    i_lim = n_electrons * F * D * C_bulk / delta
    
    # Kinetic current (from Butler-Volmer)
    eta = voltage - E_eq
    i_kinetic = i0 * (
        jnp.exp(alpha * F * eta / (R * T)) - 
        jnp.exp(-(1-alpha) * F * eta / (R * T))
    )
    
    # Actual current is limited by mass transport
    # Use smooth approximation to avoid discontinuity
    # i = i_kinetic * tanh(i_lim / i_kinetic)
    current = i_kinetic * jnp.tanh(i_lim / (jnp.abs(i_kinetic) + 1e-6))
    
    return current


def simulate_cv_with_mass_transport(
    v_start: float = -0.8,
    v_end: float = 0.2,
    scan_rate: float = 0.05,
    C_bulk: float = 1.0,       # 1 M CuSO₄
    n_points: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate CV with mass transport effects.
    
    You'll see the current plateau at high overpotentials due to 
    diffusion limitation.
    """
    time, voltage = generate_cv_waveform(v_start, v_end, scan_rate, n_points)
    
    current_fn = lambda v: butler_volmer_with_mass_transport(v, C_bulk)
    current = vmap(current_fn)(voltage)
    
    return np.array(time), np.array(voltage), np.array(current)


# =============================================================================
# PART 5: ADDING ADDITIVES (PEG SUPPRESSION)
# =============================================================================

@jit
def langmuir_adsorption(
    concentration: float,      # ppm or mM
    K_ads: float = 0.1        # Adsorption equilibrium constant
) -> float:
    """
    Langmuir adsorption isotherm.
    
    Describes surface coverage (θ) as a function of concentration:
    θ = (K * C) / (1 + K * C)
    
    θ = 0: no coverage
    θ = 1: complete monolayer coverage
    """
    theta = (K_ads * concentration) / (1.0 + K_ads * concentration)
    return theta


@jit
def butler_volmer_with_PEG(
    voltage: float,
    PEG_ppm: float = 0.0,      # PEG concentration (ppm)
    Cl_ppm: float = 50.0,      # Chloride concentration (ppm)
    C_bulk: float = 1.0,
    i0_base: float = 10.0,
    alpha: float = 0.5,
    E_eq: float = -0.3,
    T: float = T_ROOM
) -> float:
    """
    Butler-Volmer with PEG suppressor effect.
    
    PEG + Cl⁻ forms a complex that adsorbs on the copper surface,
    blocking active sites and reducing the exchange current density.
    
    The suppression effect is modeled as:
    i0_effective = i0_base * (1 - suppression_strength * θ_PEG)
    
    Where θ_PEG is the PEG surface coverage from Langmuir isotherm.
    """
    # PEG-Cl complex adsorption (requires both PEG and Cl)
    effective_PEG = PEG_ppm * (Cl_ppm / 50.0)  # Cl enhances PEG adsorption
    theta_PEG = langmuir_adsorption(effective_PEG, K_ads=0.01)
    
    # Suppress exchange current density
    suppression_strength = 0.95  # PEG can suppress up to 95%
    i0_effective = i0_base * (1.0 - suppression_strength * theta_PEG)
    
    # Standard Butler-Volmer with modified i0
    eta = voltage - E_eq
    current = i0_effective * (
        jnp.exp(alpha * F * eta / (R * T)) - 
        jnp.exp(-(1-alpha) * F * eta / (R * T))
    )
    
    return current


def simulate_cv_with_PEG(
    v_start: float = -0.8,
    v_end: float = 0.2,
    scan_rate: float = 0.05,
    PEG_ppm: float = 100.0,
    n_points: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate CV with PEG suppressor.
    
    Compare with and without PEG to see the suppression effect.
    """
    time, voltage = generate_cv_waveform(v_start, v_end, scan_rate, n_points)
    
    current_fn = lambda v: butler_volmer_with_PEG(v, PEG_ppm=PEG_ppm)
    current = vmap(current_fn)(voltage)
    
    return np.array(time), np.array(voltage), np.array(current)


# =============================================================================
# PART 6: ELECTROCHEMICAL IMPEDANCE SPECTROSCOPY (EIS)
# =============================================================================

@jit
def randles_circuit_impedance(
    frequency: float,          # Hz
    R_s: float = 10.0,        # Solution resistance (Ω·cm²)
    R_ct: float = 50.0,       # Charge transfer resistance (Ω·cm²)
    C_dl: float = 20e-6,      # Double layer capacitance (F/cm²)
    sigma: float = 10.0       # Warburg coefficient (Ω·cm²·s^-0.5)
) -> jnp.ndarray:
    """
    Randles equivalent circuit for EIS.
    
    Circuit: R_s + [R_ct + W] || C_dl
    
    Where:
    - R_s: Solution resistance (electrolyte)
    - R_ct: Charge transfer resistance (kinetics)
    - C_dl: Double layer capacitance (interface)
    - W: Warburg impedance (diffusion)
    
    Returns complex impedance Z = Z_real + j*Z_imag
    """
    omega = 2.0 * jnp.pi * frequency
    
    # Warburg impedance (diffusion)
    sqrt_omega = jnp.sqrt(omega)
    Z_w_real = sigma / sqrt_omega
    Z_w_imag = -sigma / sqrt_omega  # -j component
    
    # Charge transfer + Warburg
    Z_ct_w_real = R_ct + Z_w_real
    Z_ct_w_imag = Z_w_imag
    
    # Parallel with capacitance
    Y_cdl_real = 0.0
    Y_cdl_imag = omega * C_dl
    
    # Z_parallel = 1 / (1/Z_ct_w + Y_cdl)
    denom_real = 1.0 / Z_ct_w_real + Y_cdl_real
    denom_imag = -1.0 / Z_ct_w_imag + Y_cdl_imag
    
    # Complex division
    denom_mag_sq = denom_real**2 + denom_imag**2
    Z_par_real = denom_real / denom_mag_sq
    Z_par_imag = -denom_imag / denom_mag_sq
    
    # Add solution resistance
    Z_total_real = R_s + Z_par_real
    Z_total_imag = Z_par_imag
    
    return jnp.array([Z_total_real, Z_total_imag])


def simulate_EIS(
    freq_min: float = 0.1,     # Hz
    freq_max: float = 10000.0, # Hz
    n_points: int = 50,
    R_ct: float = 50.0,        # This is what changes with PEG!
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate electrochemical impedance spectroscopy.
    
    Returns:
    --------
    frequencies, Z_real, Z_imag : For Nyquist and Bode plots
    """
    # Logarithmic frequency sweep
    frequencies = jnp.logspace(
        jnp.log10(freq_min), 
        jnp.log10(freq_max), 
        n_points
    )
    
    # Compute impedance at each frequency
    impedance_fn = lambda f: randles_circuit_impedance(f, R_ct=R_ct, **kwargs)
    impedances = vmap(impedance_fn)(frequencies)
    
    Z_real = impedances[:, 0]
    Z_imag = impedances[:, 1]
    
    return np.array(frequencies), np.array(Z_real), np.array(Z_imag)


# =============================================================================
# PART 7: PUTTING IT ALL TOGETHER - COPPER DEPOSITION SIMULATOR
# =============================================================================

class CopperDepositionSimulator:
    """
    Complete copper electrodeposition simulator.
    
    Combines:
    - Butler-Volmer kinetics
    - Mass transport
    - Additive effects (PEG, SPS, JGB)
    - Multiple techniques (CV, CA, EIS)
    """
    
    def __init__(
        self,
        Cu_conc: float = 1.0,      # M
        H2SO4_conc: float = 0.5,   # M
        PEG_ppm: float = 0.0,
        Cl_ppm: float = 50.0,
        SPS_ppm: float = 0.0,
        temperature: float = 298.15
    ):
        self.Cu_conc = Cu_conc
        self.H2SO4_conc = H2SO4_conc
        self.PEG_ppm = PEG_ppm
        self.Cl_ppm = Cl_ppm
        self.SPS_ppm = SPS_ppm
        self.temperature = temperature
        
        # Derived parameters
        self.i0_base = 10.0  # mA/cm²
        self.E_eq = self._calculate_equilibrium_potential()
        
    def _calculate_equilibrium_potential(self) -> float:
        """
        Nernst equation for Cu²⁺/Cu
        E = E° + (RT/nF) * ln([Cu²⁺])
        """
        E_standard = 0.34  # V vs SHE (use Hg/Hg2SO4 as reference: ~0.64 V)
        E = E_standard - 0.64  # Convert to Hg/Hg2SO4 reference
        # Concentration correction (usually small)
        E += (R * self.temperature / (2 * F)) * jnp.log(self.Cu_conc)
        return float(E)
    
    def get_effective_i0(self) -> float:
        """
        Calculate effective exchange current with additive effects.
        """
        # PEG suppression
        theta_PEG = langmuir_adsorption(self.PEG_ppm, K_ads=0.01)
        suppression = 1.0 - 0.95 * theta_PEG
        
        # SPS acceleration (breaks through PEG)
        theta_SPS = langmuir_adsorption(self.SPS_ppm, K_ads=0.1)
        acceleration = 1.0 + 5.0 * theta_SPS
        
        i0_eff = self.i0_base * suppression * acceleration
        return float(i0_eff)
    
    def simulate_cv(
        self,
        v_start: float = -0.8,
        v_end: float = 0.2,
        scan_rate: float = 0.05,
        n_points: int = 1000
    ) -> Dict[str, np.ndarray]:
        """Run cyclic voltammetry simulation."""
        time, voltage = generate_cv_waveform(v_start, v_end, scan_rate, n_points)
        
        i0_eff = self.get_effective_i0()
        current_fn = lambda v: butler_volmer_with_mass_transport(
            v, self.Cu_conc, i0_eff, 0.5, self.E_eq, T=self.temperature
        )
        current = vmap(current_fn)(voltage)
        
        return {
            'time': np.array(time),
            'voltage': np.array(voltage),
            'current': np.array(current),
            'i0_effective': i0_eff,
            'E_eq': self.E_eq
        }
    
    def simulate_eis(
        self,
        dc_voltage: float = -0.5,
        freq_min: float = 0.1,
        freq_max: float = 10000.0,
        n_points: int = 50
    ) -> Dict[str, np.ndarray]:
        """Run EIS simulation."""
        # R_ct depends on dc_voltage and additive coverage
        i0_eff = self.get_effective_i0()
        eta = dc_voltage - self.E_eq
        
        # Calculate charge transfer resistance
        # R_ct = RT / (nFi0) for small overpotentials
        R_ct = (R * self.temperature) / (2 * F * i0_eff * 1e-3)  # Convert to Ω·cm²
        
        freq, Z_real, Z_imag = simulate_EIS(
            freq_min, freq_max, n_points, R_ct=R_ct
        )
        
        return {
            'frequency': freq,
            'Z_real': Z_real,
            'Z_imag': Z_imag,
            'R_ct': R_ct
        }


# =============================================================================
# PART 8: VISUALIZATION & DEMO
# =============================================================================

def plot_cv_comparison():
    """
    Compare CV with and without PEG.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Without PEG
    t1, v1, i1 = simulate_cv_simple(i0=10.0)
    ax1.plot(v1, i1, 'b-', linewidth=2, label='No PEG')
    
    # With PEG
    t2, v2, i2 = simulate_cv_with_PEG(PEG_ppm=100.0)
    ax1.plot(v2, i2, 'r--', linewidth=2, label='100 ppm PEG')
    
    ax1.set_xlabel('Voltage (V vs Hg/Hg₂SO₄)', fontsize=12)
    ax1.set_ylabel('Current Density (mA/cm²)', fontsize=12)
    ax1.set_title('Cyclic Voltammetry: PEG Effect', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.axvline(x=-0.3, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Mass transport effect
    scan_rates = [0.01, 0.05, 0.1, 0.2]
    for sr in scan_rates:
        t, v, i = simulate_cv_with_mass_transport(scan_rate=sr)
        ax2.plot(v, i, linewidth=2, label=f'{sr} V/s')
    
    ax2.set_xlabel('Voltage (V vs Hg/Hg₂SO₄)', fontsize=12)
    ax2.set_ylabel('Current Density (mA/cm²)', fontsize=12)
    ax2.set_title('Scan Rate Effect (Mass Transport)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    return fig


def plot_eis_comparison():
    """
    Compare EIS with and without PEG (Nyquist plot).
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Without PEG
    freq1, Zr1, Zi1 = simulate_EIS(R_ct=20.0)
    ax.plot(Zr1, -Zi1, 'bo-', linewidth=2, markersize=6, label='No PEG (R_ct=20Ω)')
    
    # With PEG
    freq2, Zr2, Zi2 = simulate_EIS(R_ct=100.0)
    ax.plot(Zr2, -Zi2, 'rs-', linewidth=2, markersize=6, label='PEG (R_ct=100Ω)')
    
    ax.set_xlabel('Z_real (Ω·cm²)', fontsize=12)
    ax.set_ylabel('-Z_imag (Ω·cm²)', fontsize=12)
    ax.set_title('Nyquist Plot: PEG Increases R_ct', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig


def demo_simulator():
    """
    Full demonstration of the simulator.
    """
    print("="*70)
    print("COPPER ELECTRODEPOSITION SIMULATOR")
    print("="*70)
    print()
    
    # Create simulator
    print("Creating simulator with:")
    print("  - 1.0 M CuSO₄")
    print("  - 0.5 M H₂SO₄")
    print("  - 100 ppm PEG")
    print("  - 50 ppm Cl⁻")
    print()
    
    sim = CopperDepositionSimulator(
        Cu_conc=1.0,
        PEG_ppm=100.0,
        Cl_ppm=50.0
    )
    
    print(f"Equilibrium potential: {sim.E_eq:.3f} V")
    print(f"Effective i0: {sim.get_effective_i0():.2f} mA/cm²")
    print()
    
    # Run CV
    print("Running cyclic voltammetry...")
    cv_data = sim.simulate_cv(scan_rate=0.05)
    print(f"  Peak current: {np.max(np.abs(cv_data['current'])):.2f} mA/cm²")
    print()
    
    # Run EIS
    print("Running impedance spectroscopy...")
    eis_data = sim.simulate_eis(dc_voltage=-0.5)
    print(f"  Charge transfer resistance: {eis_data['R_ct']:.1f} Ω·cm²")
    print()
    
    print("Generating plots...")
    fig1 = plot_cv_comparison()
    fig2 = plot_eis_comparison()
    
    plt.show()
    
    print()
    print("="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Compare these simulations to your Gamry data")
    print("2. Fit model parameters to match real experiments")
    print("3. Add SPS accelerator effects")
    print("4. Implement chronoamperometry")
    print("5. Build optimization loop for recipe tuning")
    print("="*70)


if __name__ == "__main__":
    # Run the demo
    demo_simulator()
