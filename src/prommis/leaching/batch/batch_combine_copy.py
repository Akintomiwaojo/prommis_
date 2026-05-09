from pyomo.environ import (
    ConcreteModel,
    Expression,
    NonNegativeReals,
    Param,
    Set,
    Suffix,
    TransformationFactory,
    Var,
    assert_optimal_termination,
    value,
    units,
)
from pyomo.dae import ContinuousSet, DerivativeVar
from idaes.core.solvers import get_solver
from idaes.core.scaling import CustomScalerBase
import matplotlib.pyplot as plt


N_FE = 20  # finite elements for backward-difference discretisation

V_L = 0.9959  # liquid volume [L]
M_S0 = 9.96e-3  # total solid feed [kg]

mass_frac_fs = {
    "Sc2O3": 0.025,
    "Y2O3": 0.0832,
    "La2O3": 0.1830,
    "Ce2O3": 0.4293,
    "Pr2O3": 0.0416,
    "Nd2O3": 0.1531,
    "Sm2O3": 0.0399,
    "Gd2O3": 0.0283,
    "Dy2O3": 0.0166,
}

MW_OX = {
    # "Al2O3": (26.982 * 2 + 3 * 15.999) * 1e-3,
    # "Fe2O3": (55.845 * 2 + 3 * 15.999) * 1e-3,
    # "CaO": (40.078 + 15.999) * 1e-3,
    "Sc2O3": (44.946 * 2 + 3 * 15.999) * 1e-3,
    "Y2O3": (88.905 * 2 + 3 * 15.999) * 1e-3,
    "La2O3": (138.905 * 2 + 3 * 15.999) * 1e-3,
    "Ce2O3": (140.116 * 2 + 3 * 15.999) * 1e-3,
    "Pr2O3": (140.907 * 2 + 3 * 15.999) * 1e-3,
    "Nd2O3": (144.242 * 2 + 3 * 15.999) * 1e-3,
    "Sm2O3": (150.36 * 2 + 3 * 15.999) * 1e-3,
    "Gd2O3": (157.25 * 2 + 3 * 15.999) * 1e-3,
    "Dy2O3": (162.50 * 2 + 3 * 15.999) * 1e-3,
}

A_ox = {
    "Sc2O3": 1.060953,
    "Y2O3": 0.847822,
    "La2O3": 0.588942,
    "Ce2O3": 0.712741,
    "Pr2O3": 0.693478,
    "Nd2O3": 1.002756,
    "Sm2O3": 0.779691,
    "Gd2O3": 1.692433,
    "Dy2O3": 0.705743,
    # "Al2O3": 1.778202,
    # "CaO":   0.351398,
    # "Fe2O3": 2.000000,
}


k_prime = {
    "Sc2O3": 1.885814e-05,
    "Y2O3": 1.069003e-05,
    "La2O3": 9.680611e-06,
    "Ce2O3": 7.161427e-05,
    "Pr2O3": 6.519568e-06,
    "Nd2O3": 1.617839e-04,
    "Sm2O3": 3.341028e-06,
    "Gd2O3": 2.603209e-03,
    "Dy2O3": 1.000000e-06,
    # "Al2O3": 7.358337e+00,
    # "CaO":   1.117390e-03,
    # "Fe2O3": 9.999999e+01,
}

K_film = {
    "Sc2O3": 5.000109e01,
    "Y2O3": 4.956089e01,
    "La2O3": 4.695036e01,
    "Ce2O3": 1.000001e-03,
    "Pr2O3": 4.830578e01,
    "Nd2O3": 4.755947e01,
    "Sm2O3": 4.957167e01,
    "Gd2O3": 4.927297e01,
    "Dy2O3": 5.130148e01,
    # "Al2O3": 9.921235e+01,
    # "CaO":   9.997876e+01,
    # "Fe2O3": 9.999913e+01,
}


D_e = {
    "Sc2O3": 1.000030e-10,
    "Y2O3": 9.407288e-06,
    "La2O3": 9.922580e-06,
    "Ce2O3": 9.961020e-06,
    "Pr2O3": 9.616805e-06,
    "Nd2O3": 9.799733e-06,
    "Sm2O3": 9.255063e-06,
    "Gd2O3": 9.532976e-06,
    "Dy2O3": 1.000000e-10,
    # "Al2O3": 9.999908e-06,
    # "CaO":   9.999657e-06,
    # "Fe2O3": 1.354318e-07,
}

# Liquid stoichiometry (oxide, liquid species) -> coefficient
# M2O3 + 6H+ -> 2M^3+ + 3H2O  ;  CaO + 2H+ -> Ca^2+ + H2O
NU_LIQUID = {
    # ("Al2O3", "Al"): 2,
    # ("Al2O3", "H"): -6,
    # ("Al2O3", "H2O"): 3,
    # ("Fe2O3", "Fe"): 2,
    # ("Fe2O3", "H"): -6,
    # ("Fe2O3", "H2O"): 3,
    # ("CaO", "Ca"): 1,
    # ("CaO", "H"): -2,
    # ("CaO", "H2O"): 1,
    ("Sc2O3", "Sc"): 2,
    ("Sc2O3", "H"): -6,
    ("Sc2O3", "H2O"): 3,
    ("Y2O3", "Y"): 2,
    ("Y2O3", "H"): -6,
    ("Y2O3", "H2O"): 3,
    ("La2O3", "La"): 2,
    ("La2O3", "H"): -6,
    ("La2O3", "H2O"): 3,
    ("Ce2O3", "Ce"): 2,
    ("Ce2O3", "H"): -6,
    ("Ce2O3", "H2O"): 3,
    ("Pr2O3", "Pr"): 2,
    ("Pr2O3", "H"): -6,
    ("Pr2O3", "H2O"): 3,
    ("Nd2O3", "Nd"): 2,
    ("Nd2O3", "H"): -6,
    ("Nd2O3", "H2O"): 3,
    ("Sm2O3", "Sm"): 2,
    ("Sm2O3", "H"): -6,
    ("Sm2O3", "H2O"): 3,
    ("Gd2O3", "Gd"): 2,
    ("Gd2O3", "H"): -6,
    ("Gd2O3", "H2O"): 3,
    ("Dy2O3", "Dy"): 2,
    ("Dy2O3", "H"): -6,
    ("Dy2O3", "H2O"): 3,
}


C0 = {
    "H2O": 1e6 / 18e3,  # ~55.56 mol/L
    "H": 0.10,  # 2 * 0.05 mol/L
    "HSO4": 1e-13,
    "SO4": 0.05,
    "Sc": 2.224e-15,
    "Y": 1.125e-15,
    "La": 7.199e-16,
    "Ce": 7.137e-16,
    "Pr": 7.097e-16,
    "Nd": 6.933e-16,
    "Sm": 6.651e-16,
    "Gd": 6.359e-16,
    "Dy": 6.154e-16,
}

OXIDES = list(MW_OX.keys())
LIQ_REE_IMP = ["Sc", "Y", "La", "Ce", "Pr", "Nd", "Sm", "Gd", "Dy"]
ALL_LIQ = ["H2O", "H", "HSO4", "SO4"] + LIQ_REE_IMP


# Maps oxide name <-> dissolved REE species name
OX_TO_REE = {
    "Sc2O3": "Sc",
    "Y2O3": "Y",
    "La2O3": "La",
    "Ce2O3": "Ce",
    "Pr2O3": "Pr",
    "Nd2O3": "Nd",
    "Sm2O3": "Sm",
    "Gd2O3": "Gd",
    "Dy2O3": "Dy",
}
REE_TO_OX = {ree: ox for ox, ree in OX_TO_REE.items()}
NU_REE = {ox: 2 for ox in OX_TO_REE}  # moles of REE per mole of M2O3 oxide


def build_model(batch_duration=2.0):

    m = ConcreteModel()

    # --- Sets ---
    m.t = ContinuousSet(bounds=(0, batch_duration))
    m.oxides = Set(initialize=OXIDES, ordered=True)
    m.liq = Set(initialize=ALL_LIQ, ordered=True)

    # --- Parameters ---
    m.V_L = Param(initialize=V_L, units=units.L, doc="Liquid volume [L]")

    m.Ka2 = Param(
        initialize=10**-1.99,
        units=units.mol / units.L,
    )
    m.A_ox = Param(m.oxides, initialize=A_ox, units=units.dimensionless)
    m.k_prime = Param(
        m.oxides, initialize=k_prime, units=units.mol * units.m**-2 * units.hr**-1
    )
    m.K_film = Param(m.oxides, initialize=K_film, units=units.m * units.hr**-1)
    m.D_e = Param(m.oxides, initialize=D_e, units=units.m**2 * units.hr**-1)

    m.MW_ox = Param(
        m.oxides,
        initialize=MW_OX,
        units=units.kg / units.mol,
    )
    m.mass_solid_feed = Param(
        initialize=M_S0, units=units.kg, doc="Total solid mass fed to reactor [kg]"
    )
    m.rho_solid = Param(initialize=2.4, units=units.kg / units.L, doc="Coal density")
    m.rho_liquid = Param(
        initialize=1.0,
        units=units.kg / units.litre,
        mutable=True,
        doc="Liquid phase density",
    )
    m.mass_frac_fs_ox = Param(
        m.oxides,
        initialize={ox: mass_frac_fs[ox] for ox in OXIDES},
        doc="Initial oxide mass fractions [-]",
    )

    m.phi_s_inlet = Param(
        initialize=0.95,
        units=units.dimensionless,
        mutable=True,
        doc="Volume fraction of solid in the solid inlet slurry stream",
    )
    m.R_p = Param(
        initialize=4.35e-6,
        units=units.m,
        mutable=True,
        doc="Mean particle radius",
    )

    # --- Differential state variables ---

    # Liquid concentrations [mol/L]
    m.C = Var(
        m.liq,
        m.t,
        bounds=(1e-12, None),
        initialize=lambda _, j, t: C0[j],
        units=units.mol / units.L,
        doc="Liquid molar concentration [mol/L]",
    )
    m.dC_dt = DerivativeVar(
        m.C,
        wrt=m.t,
        units=units.mol / units.L / units.hr,
    )

    # Solid oxide masses [kg]
    # Lower bound is 1e-10 (not 0) so IPOPT cannot step into m_s < 0,
    # which would cause pow(frac, 1/3) and pow(frac, 2/3) to raise a domain
    # error in R_ash / R_rxn when frac = m_s / m_s_feed < 0.
    m.m_s = Var(
        m.oxides,
        m.t,
        bounds=(1e-12, None),
        initialize=lambda m, ox, t: M_S0 * mass_frac_fs[ox],
        units=units.kg,
        doc="Solid oxide mass in vessel [kg]",
    )
    m.dm_s_dt = DerivativeVar(
        m.m_s,
        wrt=m.t,
        units=units.kg / units.hr,
    )

    # --- Algebraic variables ---

    m.r_ext = Var(
        m.oxides,
        m.t,
        bounds=(1e-12, None),
        initialize=0,
        units=units.mol / units.hr,
        doc="Reaction extent rate [mol/hr]",
    )
    m.r_inher = Var(
        m.t,
        initialize=1e-10,
        bounds=(1e-12, None),
        units=units.mol / units.hr,
        doc="Inherent Ka2 reaction rate [mol/hr]",
    )
    m.X = Var(
        m.oxides,
        m.t,
        bounds=(0, 1),
        initialize=0.01,
        doc="Solid-phase fractional conversion",
    )

    # --- constraints ---

    # REE and impurity liquid species
    ## MODIFY
    @m.Constraint(LIQ_REE_IMP, m.t)
    def liq_ree_imp(m, j, t):
        rhs = sum(
            NU_LIQUID[ox, j] * m.r_ext[ox, t] for ox in OXIDES if (ox, j) in NU_LIQUID
        )
        return m.V_L * m.dC_dt[j, t] == rhs

    # H2O
    @m.Constraint(m.t)
    def liq_h2o(m, t):
        return m.V_L * m.dC_dt["H2O", t] == sum(
            NU_LIQUID[ox, "H2O"] * m.r_ext[ox, t] for ox in OXIDES
        )

    # H+
    @m.Constraint(m.t)
    def liq_H(m, t):
        r_L_H = sum(NU_LIQUID[ox, "H"] * m.r_ext[ox, t] for ox in OXIDES)
        return m.V_L * m.dC_dt["H", t] == r_L_H + m.r_inher[t]

    # HSO4-
    @m.Constraint(m.t)
    def liq_HSO4(m, t):
        return m.V_L * m.dC_dt["HSO4", t] == -m.r_inher[t]

    # SO4^2-
    @m.Constraint(m.t)
    def liq_SO4(m, t):
        return m.V_L * m.dC_dt["SO4", t] == m.r_inher[t]

    # (nu_s = -1)
    @m.Constraint(m.oxides, m.t)
    def solid(m, ox, t):
        return m.dm_s_dt[ox, t] == -m.r_ext[ox, t] * m.MW_ox[ox]

    # Ka2 equilibrium
    @m.Constraint(m.t)
    def ka2_equilibrium(m, t):
        return m.C["HSO4", t] * m.Ka2 == m.C["H", t] * m.C["SO4", t]

    # conversion definition: X[ox] = (m_s_feed[ox] - m_s[ox, t]) / m_s_feed[ox]
    @m.Constraint(m.oxides, m.t)
    def conversion_def(m, ox, t):
        return (1 - m.X[ox, t]) * m.mass_solid_feed * m.mass_frac_fs_ox[ox] == m.m_s[
            ox, t
        ]

    m.volume_solid = Expression(expr=m.mass_solid_feed / m.rho_solid)
    m.total_volume = Expression(expr=m.volume_solid + m.V_L)
    m.phi_s_reactor = Expression(expr=m.volume_solid / m.total_volume)
    m.rho_pulp = Expression(
        expr=1
        / ((m.phi_s_reactor / m.rho_solid) + ((1 - m.phi_s_reactor) / m.rho_liquid))
    )

    # liquid volume per solid mass [L/kg]
    v_L_per_m_s = units.convert(
        m.V_L / m.mass_solid_feed,
        to_units=units.litre / units.kg,
    )

    m.outer_factor = Expression(
        expr=m.R_p * (1 / m.rho_pulp + v_L_per_m_s) * m.rho_solid
    )

    # --- Three series resistances, all in m²·hour/mol ---
    # Concentrations must be in mol/m³ (K_film and D_e use SI length units)

    @m.Expression(m.oxides, m.t)
    def R_film(m, ox, t):
        c_h = units.convert(m.C["H", t], to_units=units.mol / units.m**3)
        return 1 / (m.K_film[ox] * c_h)

    @m.Expression(m.oxides, m.t)
    def R_ash(m, ox, t):
        c_h = units.convert(m.C["H", t], to_units=units.mol / units.m**3)
        frac = m.m_s[ox, t] / (m.mass_solid_feed * m.mass_frac_fs_ox[ox])
        # IPOPT's interior-point line-search can evaluate expressions slightly
        # outside variable bounds; adding a small epsilon keeps the argument of
        # pow(·, 1/3) strictly positive.  1e-4 is negligible vs frac ~ O(1)
        # (< 0.4 % error at 99 % conversion) but larger than any observed
        # bound violation (~3e-5).
        frac_safe = frac + 1e-4
        return (
            (1 - frac_safe ** (1 / 3))
            * m.R_p
            / (m.D_e[ox] * c_h * frac_safe ** (1 / 3))
        )

    @m.Expression(m.oxides, m.t)
    def R_rxn(m, ox, t):
        # strip units from c_h so the exponent A_ox (a Param) is dimensionless
        c_h = units.convert(m.C["H", t], to_units=units.mol / units.m**3) / (
            units.mol / units.m**3
        )
        frac = m.m_s[ox, t] / (m.mass_solid_feed * m.mass_frac_fs_ox[ox])
        frac_safe = frac + 1e-4  # same regularisation as R_ash

        return 1 / (m.k_prime[ox] * c_h ** m.A_ox[ox] * frac_safe ** (2 / 3))

    # Combined shrinking-core rate expression [mol/hr]
    @m.Constraint(m.oxides, m.t)
    def rate_eqn(m, ox, t):
        rate_vol = units.convert(
            3
            * m.phi_s_inlet
            / (m.outer_factor * (m.R_film[ox, t] + m.R_ash[ox, t] + m.R_rxn[ox, t])),
            to_units=units.mol / units.m**3 / units.hour,
        )
        return m.r_ext[ox, t] == m.V_L * rate_vol

    for j in ALL_LIQ:
        if j != "HSO4":
            m.C[j, 0].fix(value(C0[j]))

    for ox in OXIDES:
        m.m_s[ox, 0].fix(M_S0 * mass_frac_fs[ox])

    # --- Recovery expression ---

    # need to modify

    m.ree = Set(initialize=LIQ_REE_IMP, ordered=True)

    @m.Expression(m.ree)
    def recovery(m, ree):
        t_f = m.t.last()
        ox = REE_TO_OX[ree]
        c_L_V_L = m.C[ree, t_f] * m.V_L  # mol in liquid
        c_SR_m_SR = m.m_s[ox, t_f] / m.MW_ox[ox] * NU_REE[ox]  # mol in solid residue
        return (c_L_V_L / (c_L_V_L + c_SR_m_SR)) * 100

    # Suffix required for nlp_scaling_method = "user-scaling" in IPOPT
    m.scaling_factor = Suffix(direction=Suffix.EXPORT)

    return m


def _initialize(m):
    for t in m.t:
        for j in ALL_LIQ:
            if m.C[j, t].value is None:
                m.C[j, t].set_value(C0[j])
            if m.dC_dt[j, t].value is None:
                m.dC_dt[j, t].set_value(0)

        for ox in OXIDES:
            if m.m_s[ox, t].value is None:
                m.m_s[ox, t].set_value(M_S0 * mass_frac_fs[ox])
            if m.dm_s_dt[ox, t].value is None:
                m.dm_s_dt[ox, t].set_value(0)
            if m.r_ext[ox, t].value is None:
                m.r_ext[ox, t].set_value(0)
            if m.X[ox, t].value is None:
                m.X[ox, t].set_value(0.01)

        if m.r_inher[t].value is None:
            m.r_inher[t].set_value(0)

    # R_film, R_ash and R_rxn all have C["H"] in a denominator or raised to
    # A_ox power.  IPOPT's interior-point method can step slightly below the
    # lb=0 bound during iterations, making c_h negative and causing a domain
    # error in pow(c_h, A_ox).  A small strictly-positive lower bound (1e-8
    # mol/L ≈ pH 8) prevents this; it is physically valid since leaching is
    # always strongly acidic.  Fixed values at t=0 are unaffected.
    for t in m.t:
        if not m.C["H", t].is_fixed():
            m.C["H", t].setlb(1e-8)


def set_scaling(m):
    """Populate m.scaling_factor using CustomScalerBase (sf = 1 / expected_magnitude).

    Sets scaling for both variables AND constraints.  With
    nlp_scaling_method = "user-scaling", IPOPT scales Jacobian columns by
    variable factors and rows by constraint factors.  Without constraint
    scaling the rows are unbalanced and IPOPT converges to a spuriously
    infeasible point.

    Call after discretisation so every indexed time point exists.
    """
    batch_duration = m.t.last()
    csb = CustomScalerBase()

    # ------------------------------------------------------------------
    # Reference magnitudes for liquid concentrations [mol/L]
    # ------------------------------------------------------------------
    # REEs: upper bound is fully dissolved into V_L
    C_ref = {
        "H2O": 55.0,
        "H": 0.10,
        "HSO4": 0.50,  # near-equilibrium value at ~0.1 M H2SO4
        "SO4": 0.05,
    }
    for ree in LIQ_REE_IMP:
        ox = REE_TO_OX[ree]
        c_max = M_S0 * mass_frac_fs[ox] / MW_OX[ox] * 2.0 / V_L  # mol/L
        C_ref[ree] = max(c_max, 1e-6)

    # ------------------------------------------------------------------
    # Variable scaling
    # ------------------------------------------------------------------

    # C and dC_dt
    for j in ALL_LIQ:
        sf_c = 1.0 / C_ref[j]
        sf_dcdt = sf_c / batch_duration
        for t in m.t:
            csb.set_variable_scaling_factor(m.C[j, t], sf_c)
            csb.set_variable_scaling_factor(m.dC_dt[j, t], sf_dcdt)

    # m_s and dm_s_dt
    for ox in OXIDES:
        m_s_ref = M_S0 * mass_frac_fs[ox]  # initial (largest) mass [kg]
        sf_ms = 1.0 / max(m_s_ref, 1e-10)
        sf_dmdt = sf_ms / batch_duration
        for t in m.t:
            csb.set_variable_scaling_factor(m.m_s[ox, t], sf_ms)
            csb.set_variable_scaling_factor(m.dm_s_dt[ox, t], sf_dmdt)

    # r_ext: expected dissolution rate = moles of oxide / batch_duration [mol/hr]
    for ox in OXIDES:
        r_ref = M_S0 * mass_frac_fs[ox] / MW_OX[ox] / batch_duration
        sf_r = 1.0 / max(r_ref, 1e-12)
        for t in m.t:
            csb.set_variable_scaling_factor(m.r_ext[ox, t], sf_r)

    # r_inher: HSO4 equilibration rate ~ V_L * C_SO4_ref / batch_duration [mol/hr]
    r_inher_ref = V_L * C_ref["SO4"] / batch_duration
    sf_ri = 1.0 / max(r_inher_ref, 1e-12)
    for t in m.t:
        csb.set_variable_scaling_factor(m.r_inher[t], sf_ri)

    # X: dimensionless conversion, already O(1)
    for ox in OXIDES:
        for t in m.t:
            csb.set_variable_scaling_factor(m.X[ox, t], 1.0)

    # ------------------------------------------------------------------
    # Constraint scaling  (sf = 1 / expected residual magnitude)
    # Written directly to m.scaling_factor because csb.set_variable_scaling_factor
    # only accepts VarData; for ConstraintData use the suffix directly.
    # IPOPT reads these as row scaling factors for the Jacobian.
    # ------------------------------------------------------------------

    # liq_ree_imp: V_L * dC_dt[j] = nu * r_ext  ~  V_L * C_ref[j] / tau
    for j in LIQ_REE_IMP:
        mag = V_L * C_ref[j] / batch_duration
        sf = 1.0 / max(mag, 1e-20)
        for t in m.t:
            m.scaling_factor[m.liq_ree_imp[j, t]] = sf

    # liq_h2o
    mag_h2o = V_L * C_ref["H2O"] / batch_duration
    for t in m.t:
        m.scaling_factor[m.liq_h2o[t]] = 1.0 / max(mag_h2o, 1e-20)

    # liq_H
    mag_H = V_L * C_ref["H"] / batch_duration
    for t in m.t:
        m.scaling_factor[m.liq_H[t]] = 1.0 / max(mag_H, 1e-20)

    # liq_HSO4
    mag_HSO4 = V_L * C_ref["HSO4"] / batch_duration
    for t in m.t:
        m.scaling_factor[m.liq_HSO4[t]] = 1.0 / max(mag_HSO4, 1e-20)

    # liq_SO4
    mag_SO4 = V_L * C_ref["SO4"] / batch_duration
    for t in m.t:
        m.scaling_factor[m.liq_SO4[t]] = 1.0 / max(mag_SO4, 1e-20)

    # solid: dm_s_dt = -r_ext * MW_ox  ~  M_S0 * frac / tau [kg/hr]
    for ox in OXIDES:
        mag_solid = M_S0 * mass_frac_fs[ox] / batch_duration
        sf = 1.0 / max(mag_solid, 1e-20)
        for t in m.t:
            m.scaling_factor[m.solid[ox, t]] = sf

    # ka2_equilibrium: C_HSO4 * Ka2 == C_H * C_SO4  ~  C_H_ref * C_SO4_ref
    mag_ka2 = C_ref["H"] * C_ref["SO4"]
    for t in m.t:
        m.scaling_factor[m.ka2_equilibrium[t]] = 1.0 / max(mag_ka2, 1e-20)

    # conversion_def: (1-X) * M_S0 * frac == m_s  ~  M_S0 * frac [kg]
    for ox in OXIDES:
        mag_conv = M_S0 * mass_frac_fs[ox]
        sf = 1.0 / max(mag_conv, 1e-20)
        for t in m.t:
            m.scaling_factor[m.conversion_def[ox, t]] = sf

    # rate_eqn: r_ext == V_L * rate_vol  ~  r_ref [mol/hr]
    for ox in OXIDES:
        r_ref = M_S0 * mass_frac_fs[ox] / MW_OX[ox] / batch_duration
        sf = 1.0 / max(r_ref, 1e-12)
        for t in m.t:
            m.scaling_factor[m.rate_eqn[ox, t]] = sf


def build_for_diagnostics(batch_duration=2.0):
    """Build, discretise, and initialise the model without solving.

    Use this as the input to DiagnosticsToolbox so that all variables
    (including DerivativeVars created during discretisation) have numeric
    values rather than None.
    """
    m = build_model(batch_duration=batch_duration)
    TransformationFactory("dae.finite_difference").apply_to(
        m, nfe=N_FE, wrt=m.t, scheme="BACKWARD"
    )
    m.r_inher[m.t.first()].fix(0)
    _initialize(m)
    set_scaling(m)
    return m


def discretise_and_solve(m):
    """Apply backward-difference discretisation and solve with IPOPT.

    ``core.scale_model`` is not used here because it does not cover the
    finite-difference constraints added by the DAE transformation (those
    rows have no entry in the scaling_factor suffix), which distorts the
    Jacobian and causes false infeasibility.

    Instead, ``nlp_scaling_method = "user-scaling"`` tells IPOPT to read
    the scaling_factor suffix directly for both variables and constraints,
    operating entirely in scaled space without rebuilding the model.
    """
    TransformationFactory("dae.finite_difference").apply_to(
        m, nfe=N_FE, wrt=m.t, scheme="BACKWARD"
    )

    m.r_inher[m.t.first()].fix(0)

    _initialize(m)
    set_scaling(m)

    solver = get_solver()
    solver.options["max_iter"] = 5000
    solver.options["halt_on_ampl_error"] = "yes"
    solver.options["nlp_scaling_method"] = "user-scaling"
    results = solver.solve(m, tee=True)
    assert_optimal_termination(results)
    return results


def print_results(m):
    """Print final solid recovery and liquid concentrations."""
    t_f = sorted(m.t)[-1]

    print("\n" + "=" * 60)
    print("Batch Leaching Results")
    print("=" * 60)
    print(f"Liquid volume  : {V_L:.1f} L")
    print(f"Total solid    : {M_S0:.2f} kg")

    print(
        f"\n  {'Oxide':<10}  {'Init mass (kg)':>15}  {'Final mass (kg)':>16}  {'Recovery (%)':>13}"
    )
    print("  " + "-" * 57)
    for ree in LIQ_REE_IMP:
        ox = REE_TO_OX[ree]
        m_in = M_S0 * mass_frac_fs[ox]
        m_out = value(m.m_s[ox, t_f])
        rec = value(m.recovery[ree])
        print(f"  {ox:<10}  {m_in:>15.6f}  {m_out:>16.6f}  {rec:>13.4f}")

    print("\nFinal liquid concentrations [mol/L]:")
    for j in ALL_LIQ:
        print(f"  {j:<8}  {value(m.C[j, t_f]):.6e}")
    print("=" * 60)


def plot_recovery(recovery_data: dict, rees: list, durations: list):
    """
    Parameters
    ----------
    recovery_data : {ree: [rec_at_1hr, rec_at_2hr, ...]}
    rees          : list of REE names to plot (~5)
    durations     : list of batch durations in hours
    """
    _, ax = plt.subplots(figsize=(9, 5))
    for ree in rees:
        ax.plot(durations, recovery_data[ree], marker="o", label=ree)
    ax.set_xlabel("Batch duration [hr]")
    ax.set_ylabel("Leach recovery [%]")
    ax.set_title("REE leach recovery vs batch duration (0.1 M H$_2$SO$_4$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def batch(time=2.0):
    DURATIONS = list(range(1, int(time) + 1))
    recovery_data = {ree: [] for ree in LIQ_REE_IMP}
    REES_to_plot = ["Sc", "Gd", "La", "Ce", "Nd"]  # choose ~5 REEs to visualise

    for t in DURATIONS:
        m = build_model(batch_duration=t)
        discretise_and_solve(m)
        print_results(m)
        for ree in LIQ_REE_IMP:
            recovery_data[ree].append(value(m.recovery[ree]))
    plot_recovery(recovery_data, REES_to_plot, DURATIONS)


if __name__ == "__main__":
    batch(time=2)
