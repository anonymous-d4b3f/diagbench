"""
Piezoelectric cantilever analytical physics oracle.

Based on Erturk & Inman (2011), "Piezoelectric Energy Harvesting", Chapter 3:
  - Single-mode Euler-Bernoulli cantilever beam model
  - Unimorph configuration (substrate + single piezo layer)
  - Tip mass correction using first-mode shape factor

Positioning:
  This is a fast analytical physics oracle (~0.5ms/call) for use in:
    1. Reference solver portfolio (RBKF generation)
    2. Agent loop feedback (replacing v1 fixture_midpoint_reference)
    3. Constraint validation and slack computation

  It is NOT:
    - FEM truth (no mesh, no 3D effects)
    - Experimental truth (no damping identification, no fabrication variation)
    - A fully validated production simulator

  Calibration status:
    - Resonant frequency: consistent with Euler-Bernoulli first-mode prediction
    - Power output: single-mode electromechanical approximation
    - Stress: root bending stress under base excitation (simplified)
    - Validated against: published benchmark cases in Erturk & Inman (2011) Table 3.1

Material parameters:
  - PZT-5A, PZT-5H: from Erturk & Inman (2011) Appendix C, standard IEEE values
  - MFC-M8528: from Smart Material Corp. datasheet (typical values)
  - Substrate: standard structural material reference values

Units:
  Input: mm, μm, g, Ω, Hz, g_accel
  Output: Hz, μW, MPa, mm, %
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Material databases
# ──────────────────────────────────────────────────────────────────────────────

# Piezoelectric materials: d31 [m/V], eps33_r (relative permittivity),
# E_p [Pa] (Young's modulus), rho_p [kg/m³]
# Source: Erturk & Inman (2011) Appendix C; MFC from Smart Material datasheet
_PIEZO_MATERIALS: dict[str, dict[str, float]] = {
    "PZT-5A": {
        "d31": -171e-12,   # m/V  (IEEE Std 176)
        "eps33_r": 1700.0,  # relative permittivity (T_33 / eps_0)
        "E_p": 61.0e9,      # Pa
        "rho_p": 7750.0,    # kg/m³
    },
    "PZT-5H": {
        "d31": -274e-12,
        "eps33_r": 3800.0,
        "E_p": 60.0e9,
        "rho_p": 7500.0,
    },
    "MFC-M8528": {
        # Macro-Fiber Composite, d33-mode approximated as equivalent d31
        "d31": -170e-12,
        "eps33_r": 800.0,
        "E_p": 30.0e9,
        "rho_p": 5400.0,
    },
    # Legacy alias used in v1 task bank
    "PVDF": {
        "d31": -23e-12,
        "eps33_r": 13.0,
        "E_p": 2.0e9,
        "rho_p": 1780.0,
    },
}

# Substrate materials: E_s [Pa], rho_s [kg/m³]
# Source: Matweb reference values (typical/representative)
_SUBSTRATE_MATERIALS: dict[str, dict[str, float]] = {
    "stainless_steel": {"E_s": 200.0e9, "rho_s": 7900.0},
    "aluminum":        {"E_s": 70.0e9,  "rho_s": 2700.0},
    "brass":           {"E_s": 100.0e9, "rho_s": 8500.0},
    "titanium":        {"E_s": 114.0e9, "rho_s": 4510.0},
}

# Default values when not specified in task
_DEFAULT_PIEZO = "PZT-5A"
_DEFAULT_SUBSTRATE = "stainless_steel"
_DEFAULT_ZETA = 0.01          # mechanical damping ratio (1%)
_EPS_0 = 8.854187817e-12      # vacuum permittivity [F/m]
_PI = math.pi
_G = 9.80665                  # standard gravity [m/s²]

# Erturk first-mode equivalent mass factor for uniform cantilever without tip mass
# m_eff = phi_factor * m_beam + m_tip  where phi_factor ≈ 0.2357
# See Erturk & Inman (2011) Eq. (3.81)
_PHI_FACTOR = 0.2357

# First clamped-free bending mode root slope for a tip-normalized mode shape:
#   -∫_0^L phi''(x) dx = phi'(0) ≈ 1.3765 / L
# This converts distributed piezoelectric bending strain into the single-mode
# electromechanical coupling coefficient θ [N/V = C/m].
_FIRST_MODE_ROOT_SLOPE_TIP_NORM = 1.3765

# ──────────────────────────────────────────────────────────────────────────────
# Material alias tables — map common user-supplied names → canonical DB keys
# ──────────────────────────────────────────────────────────────────────────────

_PIEZO_ALIASES: dict[str, str] = {
    # PZT-5A variants
    "pzt":       "PZT-5A",
    "pzt5a":     "PZT-5A",
    "pzt-5a":    "PZT-5A",
    "pzt_5a":    "PZT-5A",
    "pzt 5a":    "PZT-5A",
    # PZT-5H variants
    "pzt5h":     "PZT-5H",
    "pzt-5h":    "PZT-5H",
    "pzt_5h":    "PZT-5H",
    "pzt 5h":    "PZT-5H",
    # MFC variants
    "mfc":          "MFC-M8528",
    "mfc-m8528":    "MFC-M8528",
    "mfc_m8528":    "MFC-M8528",
    "mfcm8528":     "MFC-M8528",
    # PVDF
    "pvdf":      "PVDF",
}

_SUBSTRATE_ALIASES: dict[str, str] = {
    # stainless steel variants
    "steel":              "stainless_steel",
    "stainless steel":    "stainless_steel",
    "stainless-steel":    "stainless_steel",
    "stainlesssteel":     "stainless_steel",
    "ss":                 "stainless_steel",
    "ss304":              "stainless_steel",
    "ss316":              "stainless_steel",
    # aluminum variants
    "al":         "aluminum",
    "aluminium":  "aluminum",
    # titanium
    "ti":         "titanium",
    # brass
    "cu-zn":      "brass",
}


def _canonicalize_material(name: str, alias_map: dict[str, str], db_keys: list[str]) -> str:
    """Return canonical DB key for a material name, trying alias lookup then case-fold."""
    if name in db_keys:
        return name
    # Try alias map (lowercased, spaces normalized to underscores)
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    # also try without underscores
    compact = normalized.replace("_", "")
    for key in (normalized, compact, name.lower()):
        if key in alias_map:
            return alias_map[key]
    return name   # unchanged — will raise ValueError in caller


def normalize_environment_context(environment: Optional[dict]) -> dict[str, object]:
    """Normalize legacy and v2 environment context keys, and canonicalize material names."""
    env = dict(environment or {})
    # Key aliasing: material ↔ piezo_material
    if "piezo_material" not in env and "material" in env:
        env["piezo_material"] = env["material"]
    if "material" not in env and "piezo_material" in env:
        env["material"] = env["piezo_material"]
    env.setdefault("substrate_material", _DEFAULT_SUBSTRATE)
    # Material name canonicalization
    if "piezo_material" in env and env["piezo_material"] is not None:
        env["piezo_material"] = _canonicalize_material(
            str(env["piezo_material"]), _PIEZO_ALIASES, list(_PIEZO_MATERIALS.keys())
        )
        env["material"] = env["piezo_material"]
    if "substrate_material" in env and env["substrate_material"] is not None:
        env["substrate_material"] = _canonicalize_material(
            str(env["substrate_material"]), _SUBSTRATE_ALIASES, list(_SUBSTRATE_MATERIALS.keys())
        )
    return env


def normalize_constraint_limits(constraints: Optional[dict[str, float]]) -> dict[str, float]:
    """Normalize legacy and v2 constraint names."""
    normalized = dict(constraints or {})
    if "freq_error_pct_limit" not in normalized and "freq_error_pct" in normalized:
        normalized["freq_error_pct_limit"] = normalized["freq_error_pct"]
    if "freq_error_pct" not in normalized and "freq_error_pct_limit" in normalized:
        normalized["freq_error_pct"] = normalized["freq_error_pct_limit"]
    return normalized


@dataclass
class OracleResult:
    """Full output from PiezoelectricOracle.evaluate().

    constraint_slack keys are the full constraint names used in task schema:
      "stress_limit_mpa", "disp_limit_mm", "freq_error_pct_limit", "power_target_uw"
    Positive slack = constraint satisfied; negative = violated.
    """
    resonant_freq_hz: float
    load_power_uw: float
    tip_stress_mpa: float
    tip_disp_mm: float
    freq_error_pct: float
    is_feasible: bool
    constraint_slack: dict[str, float]
    # Provenance
    oracle_tier: str = "analytical"
    damping_ratio: float = _DEFAULT_ZETA   # actual zeta used in this evaluation
    # Intermediate values for diagnostics
    effective_mass_kg: Optional[float] = None
    bending_stiffness_nm2: Optional[float] = None
    coupling_coefficient: Optional[float] = None
    internal_capacitance_f: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "resonant_freq_hz": round(self.resonant_freq_hz, 4),
            "load_power_uw": round(self.load_power_uw, 6),
            "tip_stress_mpa": round(self.tip_stress_mpa, 4),
            "tip_disp_mm": round(self.tip_disp_mm, 4),
            "freq_error_pct": round(self.freq_error_pct, 4),
            "is_feasible": self.is_feasible,
            "constraint_slack": {k: round(v, 6) for k, v in self.constraint_slack.items()},
            "oracle_tier": self.oracle_tier,
            "damping_ratio": self.damping_ratio,
        }


class PiezoelectricOracle:
    """
    Fast analytical physics oracle for piezoelectric cantilever VEH.

    Model: Erturk-Inman single-mode, unimorph, Euler-Bernoulli beam.
    Tier: "analytical" — calibrated approximation, NOT FEM/experimental truth.
    Compute time: ~0.1-0.5 ms per call (pure Python, no FEM solver).

    Supported design variables (all 6 required unless optional noted):
      beam_length_mm          [10, 200]    mm
      beam_width_mm           [1, 50]      mm
      substrate_thickness_um  [50, 2000]   μm
      piezo_thickness_um      [5, 500]     μm
      tip_mass_g              [0, 20]      g
      load_resistance_ohm     [100, 1e7]   Ω

    Supported excitation fields:
      frequency_hz            Hz  (excitation frequency)
      acceleration_g          g   (base acceleration amplitude)

    Environment context (optional):
      piezo_material          str  ("PZT-5A", "PZT-5H", "MFC-M8528", "PVDF")
      substrate_material      str  ("stainless_steel", "aluminum", "brass", "titanium")
      damping_ratio           float  (default: 0.01)

    Constraint fields recognized in task (optional):
      stress_limit_mpa        upper bound on tip_stress_mpa
      disp_limit_mm           upper bound on tip_disp_mm
      freq_error_pct_limit    upper bound on freq_error_pct
      power_target_uw         lower bound on load_power_uw
    """

    def __init__(
        self,
        damping_ratio: float = _DEFAULT_ZETA,
    ) -> None:
        self._default_zeta = damping_ratio

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        params: dict[str, float],
        excitation: dict[str, float],
        constraints: Optional[dict[str, float]] = None,
        environment: Optional[dict] = None,
    ) -> OracleResult:
        """
        Evaluate design candidate under given excitation.

        Args:
            params: Design variable values (keys = design variable names).
            excitation: {"frequency_hz": float, "acceleration_g": float}
            constraints: Optional constraint limits override.
                         Keys: stress_limit_mpa, disp_limit_mm,
                               freq_error_pct_limit, power_target_uw
            environment: Optional {"piezo_material": str, "substrate_material": str,
                                   "damping_ratio": float}

        Returns:
            OracleResult dataclass (call .to_dict() for JSON-compatible output).

        Raises:
            ValueError: If required params are missing or material is unknown.
        """
        env = normalize_environment_context(environment)

        # --- Parse material parameters ---
        piezo_key = env.get("piezo_material", _DEFAULT_PIEZO)
        sub_key = env.get("substrate_material", _DEFAULT_SUBSTRATE)
        zeta = float(env.get("damping_ratio", self._default_zeta))

        if piezo_key not in _PIEZO_MATERIALS:
            raise ValueError(
                f"Unknown piezo material: {piezo_key!r}. "
                f"Available: {list(_PIEZO_MATERIALS.keys())}"
            )
        if sub_key not in _SUBSTRATE_MATERIALS:
            raise ValueError(
                f"Unknown substrate material: {sub_key!r}. "
                f"Available: {list(_SUBSTRATE_MATERIALS.keys())}"
            )

        mp = _PIEZO_MATERIALS[piezo_key]
        ms = _SUBSTRATE_MATERIALS[sub_key]

        # --- Validate numeric input domain before entering closed-form model ---
        raw_inputs = {
            "beam_length_mm": float(params["beam_length_mm"]),
            "beam_width_mm": float(params["beam_width_mm"]),
            "substrate_thickness_um": float(params["substrate_thickness_um"]),
            "piezo_thickness_um": float(params["piezo_thickness_um"]),
            "tip_mass_g": float(params["tip_mass_g"]),
            "load_resistance_ohm": float(params["load_resistance_ohm"]),
            "frequency_hz": float(excitation["frequency_hz"]),
            "acceleration_g": float(excitation["acceleration_g"]),
        }
        if raw_inputs["beam_length_mm"] <= 0:
            raise ValueError("beam_length_mm must be > 0")
        if raw_inputs["beam_width_mm"] <= 0:
            raise ValueError("beam_width_mm must be > 0")
        if raw_inputs["substrate_thickness_um"] <= 0:
            raise ValueError("substrate_thickness_um must be > 0")
        if raw_inputs["piezo_thickness_um"] <= 0:
            raise ValueError("piezo_thickness_um must be > 0")
        if raw_inputs["tip_mass_g"] < 0:
            raise ValueError("tip_mass_g must be >= 0")
        if raw_inputs["load_resistance_ohm"] <= 0:
            raise ValueError("load_resistance_ohm must be > 0")
        if raw_inputs["frequency_hz"] <= 0:
            raise ValueError("frequency_hz must be > 0")
        if raw_inputs["acceleration_g"] < 0:
            raise ValueError("acceleration_g must be >= 0")
        if zeta < 0:
            raise ValueError("damping_ratio must be >= 0")

        # --- Convert units to SI ---
        L = raw_inputs["beam_length_mm"] * 1e-3         # m
        b = raw_inputs["beam_width_mm"] * 1e-3          # m
        h_s = raw_inputs["substrate_thickness_um"] * 1e-6  # m
        h_p = raw_inputs["piezo_thickness_um"] * 1e-6      # m
        m_t = raw_inputs["tip_mass_g"] * 1e-3           # kg
        R_L = raw_inputs["load_resistance_ohm"]         # Ω

        f_exc = raw_inputs["frequency_hz"]           # Hz
        a_exc = raw_inputs["acceleration_g"] * _G   # m/s²

        # --- Composite beam bending stiffness (EI) [N·m²] ---
        # Neutral axis correction for bending stiffness of composite cross-section
        # Using parallel-axis theorem for the two-layer unimorph
        E_s, rho_s = ms["E_s"], ms["rho_s"]
        E_p, rho_p = mp["E_p"], mp["rho_p"]

        # Centroid of each layer from bottom of substrate
        y_s = h_s / 2.0
        y_p = h_s + h_p / 2.0

        # Neutral axis location
        num_na = E_s * b * h_s * y_s + E_p * b * h_p * y_p
        den_na = E_s * b * h_s + E_p * b * h_p
        y_na = num_na / den_na  # distance from bottom to neutral axis

        # EI via parallel axis theorem
        I_s = b * h_s**3 / 12.0 + b * h_s * (y_s - y_na)**2
        I_p = b * h_p**3 / 12.0 + b * h_p * (y_p - y_na)**2
        EI = E_s * I_s + E_p * I_p  # N·m²

        # --- Equivalent mass ---
        m_beam = (rho_s * h_s + rho_p * h_p) * b * L   # kg (distributed mass)
        m_eff = _PHI_FACTOR * m_beam + m_t               # kg (modal mass)

        # --- Resonant frequency ---
        # f_r = (1/2π) * sqrt(3 * EI / (m_eff * L³))
        # From Erturk & Inman (2011) Eq. (3.23) for clamped-free beam first mode
        omega_r = math.sqrt(3.0 * EI / (m_eff * L**3))  # rad/s
        f_r = omega_r / (2.0 * _PI)                      # Hz

        # --- Electromechanical coupling coefficient θ [N/V = C/m] ---
        # Erturk & Inman (2011) Eq. 3.30 for unimorph with full piezo coverage:
        #   θ_1 = ẽ31 * b * h̃_pc * [dφ_1/dx]_{x=0}^{x=L}
        # where:
        #   ẽ31 = e31 = d31 × E_p  (piezoelectric stress constant, C/m²)
        #   h̃_pc = (y_p - y_na)    distance from neutral axis to piezo centroid (m)
        #   [dφ_1/dx]_{x=0}^{x=L} = 1.3765 / L  (first-mode slope integral, tip-normalized)
        #
        # NOTE: h̃_pc is NOT multiplied by h_p.  h̃_pc alone carries the moment arm.
        # An earlier version of this code incorrectly included an extra h_p factor,
        # which suppressed θ by ~1/h_p (typically ~1000×), making power ~1e6× too low.
        d31 = mp["d31"]          # m/V
        e31 = d31 * E_p          # C/m² (piezo stress constant)
        mode_factor = _FIRST_MODE_ROOT_SLOPE_TIP_NORM / L
        theta = -e31 * b * (y_p - y_na) * mode_factor   # N/V ≡ C/m

        # --- Internal capacitance C_p [F] ---
        # C_p = eps_33^T * b * L / h_p  (clamped permittivity)
        eps33_T = mp["eps33_r"] * _EPS_0  # F/m (permittivity at const stress)
        C_p = eps33_T * b * L / h_p       # F

        # --- Steady-state response under harmonic base excitation ---
        # Correct Erturk-Inman single-mode FRF (Erturk & Inman 2011, Eq. 3.55)
        #
        # Complex denominator:
        #   D(ω) = m_eff*(ωr² - ω²) + j*2ζ*m_eff*ωr*ω + θ²*(jω*Z_e)
        # where Z_e = R_L / (1 + jωR_L*C_p) is the electrical load impedance
        #
        # W amplitude: |W| = m_eff * a_exc / |D(ω)|
        # Voltage amplitude: |V| = |θ| * ω * |Z_e| * |W|
        # Average power: P = |V|² / (2 R_L)  [W]

        omega_exc = 2.0 * _PI * f_exc       # rad/s
        omega_r = math.sqrt(3.0 * EI / (m_eff * L**3))   # rad/s (recomputed here)
        f_r = omega_r / (2.0 * _PI)

        # Electrical load impedance Z_e = R_L / (1 + jω C_p R_L)
        # Real and imaginary parts:
        tau_e = omega_exc * R_L * C_p          # dimensionless (RC time constant × ω)
        tau_e_sq = tau_e**2
        Z_e_re = R_L / (1.0 + tau_e_sq)
        Z_e_im = -R_L * tau_e / (1.0 + tau_e_sq)
        Z_e_abs = math.sqrt(Z_e_re**2 + Z_e_im**2)

        # Complex coupling term: θ² × jω × Z_e
        # j×ω × (Z_e_re + j×Z_e_im) = (-ω×Z_e_im) + j(ω×Z_e_re)
        coupling_re = theta**2 * (-omega_exc * Z_e_im)
        coupling_im = theta**2 * (omega_exc * Z_e_re)

        # Full complex denominator
        denom_re = m_eff * (omega_r**2 - omega_exc**2) + coupling_re
        denom_im = 2.0 * zeta * m_eff * omega_r * omega_exc + coupling_im
        denom_abs = math.sqrt(denom_re**2 + denom_im**2)

        # Tip displacement amplitude [m]
        W_amp = m_eff * a_exc / denom_abs

        # Voltage and power
        V_amp = abs(theta) * omega_exc * Z_e_abs * W_amp
        P_avg_uw = (V_amp**2 / (2.0 * R_L)) * 1.0e6     # μW
        W_tip_mm = W_amp * 1.0e3                           # mm

        # --- Root bending stress (substrate, tensile face) [MPa] ---
        # σ_max = E_s * (h_s - y_na) * 3 M / (E_I) at root (x=0)
        # M_root = m_eff * a_exc * L  (static equivalent moment)
        # For beam under base excitation at resonance, simplified:
        #   σ = E_s * (h_s - y_na) * W_tip / L²  * C_beam
        # Using root-bending approximation (Erturk & Inman Eq. 3.65):
        #   σ_root = E_s * c_s * (3 W_tip / L²)
        # where c_s = distance from neutral axis to outer substrate fiber
        c_s = abs(y_na - 0.0)  # = y_na (distance from NA to bottom = tensile face)
        sigma_mpa = E_s * c_s * 3.0 * W_amp / (L**2) / 1e6  # MPa

        # --- Frequency error ---
        freq_err_pct = abs(f_r - f_exc) / f_exc * 100.0  # %

        # --- Constraint evaluation ---
        normalized_constraints = normalize_constraint_limits(constraints)
        stress_limit = float(normalized_constraints.get("stress_limit_mpa", 50.0))
        disp_limit = float(normalized_constraints.get("disp_limit_mm", 5.0))
        freq_err_limit = float(normalized_constraints.get("freq_error_pct_limit", 5.0))
        power_target = float(normalized_constraints.get("power_target_uw", 1.0))

        slack = {
            "stress_limit_mpa":     round(stress_limit - sigma_mpa, 6),
            "disp_limit_mm":        round(disp_limit - W_tip_mm, 6),
            "freq_error_pct_limit": round(freq_err_limit - freq_err_pct, 6),
            "power_target_uw":      round(P_avg_uw - power_target, 6),
        }
        is_feasible = (
            sigma_mpa <= stress_limit
            and W_tip_mm <= disp_limit
            and freq_err_pct <= freq_err_limit
            and P_avg_uw >= power_target
        )

        return OracleResult(
            resonant_freq_hz=round(f_r, 4),
            load_power_uw=round(P_avg_uw, 6),
            tip_stress_mpa=round(sigma_mpa, 4),
            tip_disp_mm=round(W_tip_mm, 4),
            freq_error_pct=round(freq_err_pct, 4),
            is_feasible=is_feasible,
            constraint_slack=slack,
            oracle_tier="analytical",
            damping_ratio=zeta,
            effective_mass_kg=round(m_eff, 9),
            bending_stiffness_nm2=round(EI, 9),
            coupling_coefficient=round(theta, 12),
            internal_capacitance_f=round(C_p, 12),
        )

    # ── Convenience helpers ───────────────────────────────────────────────────

    def evaluate_from_task(self, params: dict, task: dict) -> OracleResult:
        """Evaluate using task's excitation_context and constraint definitions."""
        excitation = task["excitation_context"]
        environment = normalize_environment_context(task.get("environment_context", {}))
        # Build constraint dict from task constraint list
        constraints: dict[str, float] = {}
        for c in task.get("constraints", []):
            constraints[c["name"]] = c["limit"]
        return self.evaluate(
            params,
            excitation,
            constraints=normalize_constraint_limits(constraints),
            environment=environment,
        )

    @staticmethod
    def list_materials() -> dict[str, list[str]]:
        return {
            "piezo": list(_PIEZO_MATERIALS.keys()),
            "substrate": list(_SUBSTRATE_MATERIALS.keys()),
        }

    @staticmethod
    def get_material_params(material_type: str, name: str) -> dict[str, float]:
        db = _PIEZO_MATERIALS if material_type == "piezo" else _SUBSTRATE_MATERIALS
        if name not in db:
            raise ValueError(f"Unknown {material_type} material: {name!r}")
        return dict(db[name])
