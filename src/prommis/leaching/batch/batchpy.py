r"""
(1)  V_L * dC[i]/dt   = r_L[i]                           REE/impurity liquid species
(2)  V_L * dC[H2O]/dt = r_L[H2O]                         water
(3)  V_L * dC[H]/dt   = r_L[H] + r_inher                 H+ balance
(4)  V_L * dC[HSO4]/dt = -r_inher                        HSO4- balance
(5)  V_L * dC[SO4]/dt  = r_inher                         SO4^2- balance
(6)  dm_s[ox]/dt = -r_ext[ox] * MW[ox]                   solid oxide mass (nu_s = -1)
(7)  C[HSO4] * Ka2 = C[H] * C[SO4]                       Ka2 equilibrium (K_eq = 10^-1.99)
(9)  r_L[i]   = sum_ox( nu[ox,i]   * r_ext[ox] )
(10) r_L[H]   = sum_ox( nu[ox,H]   * r_ext[ox] )
(11) r_L[H2O] = sum_ox( nu[ox,H2O] * r_ext[ox] )
(13) r_ext[ox] = V_L * r_ext_rate[ox]
(14) r_ext_rate[ox] = pulp_density * B[ox] * C[H]^A[ox] * (1-X[ox])^(2/3)
(15) 1 - X[ox] = (m_s[ox] * mass_frac_fs_I) / (initial_mass_inert * mass_frac_fs[ox])
"""

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Expression,
    NonNegativeReals,
    Param,
    Set,
    SolverFactory,
    TransformationFactory,
    Var,
    assert_optimal_termination,
    value,
    units,
)
from pyomo.util.check_units import assert_units_consistent
from pyomo.dae import ContinuousSet, DerivativeVar
from idaes.core.solvers import get_solver
import idaes.core.util.scaling as iscale


duration = 1.0  # batch duration [hr]
N_FE = 20  # finite elements for backward-difference discretisation

V_L = 224.3  # liquid volume [L]
M_S0 = 22.68  # total solid feed [kg]

mass_frac_fs = {
    "inerts": 0.6952,
    "Al2O3": 0.2370,
    "Fe2O3": 0.0642,
    "CaO": 3.31e-3,
    "Sc2O3": 2.77966e-05,
    "Y2O3": 3.28653e-05,
    "La2O3": 6.77769e-05,
    "Ce2O3": 1.56161e-04,
    "Pr2O3": 1.71438e-05,
    "Nd2O3": 6.76618e-05,
    "Sm2O3": 1.47926e-05,
    "Gd2O3": 1.04050e-05,
    "Dy2O3": 7.54827e-06,
}

MW_OX = {
    "Al2O3": (26.982 * 2 + 3 * 15.999) * 1e-3,
    "Fe2O3": (55.845 * 2 + 3 * 15.999) * 1e-3,
    "CaO": (40.078 + 15.999) * 1e-3,
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

A_PARAM = {
    "Al2O3": 1.496716606,
    "Fe2O3": 0.902948175,
    "CaO": 0.159744406,
    "Sc2O3": 0.763763942,
    "Y2O3": 0.580700988,
    "La2O3": 0.443101432,
    "Ce2O3": 0.601391182,
    "Pr2O3": 0.501916124,
    "Nd2O3": 0.702951111,
    "Sm2O3": 0.578717372,
    "Gd2O3": 1.063666638,
    "Dy2O3": 0.428087853,
}

B_PARAM = {
    "Al2O3": 405.5050676,
    "Fe2O3": 11.34710708,
    "CaO": 0.081844698,
    "Sc2O3": 0.000755073,
    "Y2O3": 0.000528612,
    "La2O3": 0.001028295,
    "Ce2O3": 0.007050046,
    "Pr2O3": 0.000522858,
    "Nd2O3": 0.006289942,
    "Sm2O3": 0.000252479,
    "Gd2O3": 0.013408467,
    "Dy2O3": 4.56708e-05,
}

# Liquid stoichiometry (oxide, liquid species) -> coefficient
# M2O3 + 6H+ -> 2M^3+ + 3H2O  ;  CaO + 2H+ -> Ca^2+ + H2O
NU_LIQUID = {
    ("Al2O3", "Al"): 2,
    ("Al2O3", "H"): -6,
    ("Al2O3", "H2O"): 3,
    ("Fe2O3", "Fe"): 2,
    ("Fe2O3", "H"): -6,
    ("Fe2O3", "H2O"): 3,
    ("CaO", "Ca"): 1,
    ("CaO", "H"): -2,
    ("CaO", "H2O"): 1,
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

# Initial liquid concentrations [mol/L]  (0.05 M H2SO4, metals near zero)
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
    "Al": 3.706e-15,
    "Ca": 2.495e-15,
    "Fe": 1.791e-15,
}

OXIDES = list(MW_OX.keys())
LIQ_REE_IMP = ["Sc", "Y", "La", "Ce", "Pr", "Nd", "Sm", "Gd", "Dy", "Al", "Ca", "Fe"]
ALL_LIQ = ["H2O", "H", "HSO4", "SO4"] + LIQ_REE_IMP


def build_model():

    m = ConcreteModel()

    # --- Sets ---
    m.t = ContinuousSet(bounds=(0, duration))
    m.oxides = Set(initialize=OXIDES, ordered=True)
    m.liq = Set(initialize=ALL_LIQ, ordered=True)

    # --- Parameters ---
    m.V_L = Param(initialize=V_L, units=units.L, doc="Liquid volume [L]")

    m.Ka2 = Param(
        initialize=10**-1.99,
        units=units.mol / units.L,
    )
    m.A = Param(m.oxides, initialize=A_PARAM)
    m.B = Param(
        m.oxides,
        initialize=B_PARAM,
        units=units.mol / units.kg / units.hr,
    )
    m.MW_ox = Param(
        m.oxides,
        initialize=MW_OX,
        units=units.kg / units.mol,
    )
    m.mass_solid_feed = Param(
        initialize=M_S0, units=units.kg, doc="Total solid mass fed to reactor [kg]"
    )
    m.coal_density = Param(initialize=2.4, units=units.kg / units.L, doc="Coal density")
    m.initial_mass_inert = Param(
        initialize=M_S0 * mass_frac_fs["inerts"],
        units=units.kg,
    )
    m.mass_frac_fs_I = Param(
        initialize=mass_frac_fs["inerts"], doc="Initial inert mass fraction"
    )
    m.mass_frac_fs_ox = Param(
        m.oxides,
        initialize={ox: mass_frac_fs[ox] for ox in OXIDES},
        doc="Initial oxide mass fractions [-]",
    )

    # --- Differential state variables ---

    # Liquid concentrations [mol/L]
    m.C = Var(
        m.liq,
        m.t,
        within=NonNegativeReals,
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
    m.m_s = Var(
        m.oxides,
        m.t,
        within=NonNegativeReals,
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
        within=NonNegativeReals,
        initialize=0,
        units=units.mol / units.hr,
        doc="Reaction extent rate [mol/hr]",
    )
    m.r_inher = Var(
        m.t,
        initialize=1e-10,
        units=units.mol / units.hr,
        doc="Inherent Ka2 reaction rate [mol/hr]",
    )
    m.X = Var(
        m.oxides,
        m.t,
        bounds=(0, 1 - 1e-8),
        initialize=0.01,
        doc="Solid-phase fractional conversion",
    )

    # --- constraints ---

    # REE and impurity liquid species
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

    # conversion definition
    # 1 - X[ox] = (m_s[ox] * mass_frac_fs_I) / (initial_mass_inert * mass_frac_fs[ox])
    @m.Constraint(m.oxides, m.t)
    def conversion_def(m, ox, t):
        return (1 - m.X[ox, t]) * m.initial_mass_inert * m.mass_frac_fs_ox[ox] == (
            m.m_s[ox, t] * m.mass_frac_fs_I
        )

    @m.Expression(m.t)
    def m_s_total(m, t):
        return m.initial_mass_inert + sum(m.m_s[ox, t] for ox in OXIDES)

    @m.Expression(m.t)
    def pulp_density(m, t):
        return m.m_s_total[t] / (m.m_s_total[t] / m.coal_density + m.V_L)

    # shrinking-core rate expression
    # r_ext[ox] = V_L * pulp_density(t) * B[ox] * C[H]^A[ox] * (1 - X[ox])^(2/3)
    @m.Constraint(m.oxides, m.t)
    def rate_eqn(m, ox, t):

        return m.r_ext[ox, t] == (
            (m.V_L + (m.mass_solid_feed / m.coal_density))
            * m.pulp_density[t]
            * m.B[ox]
            * (m.C["H", t] / (units.mol / units.L)) ** m.A[ox]
            * (1 - m.X[ox, t]) ** (2 / 3)
        )

    for j in ALL_LIQ:
        if j != "HSO4":
            m.C[j, 0].fix(value(C0[j]))

    for ox in OXIDES:
        m.m_s[ox, 0].fix(M_S0 * mass_frac_fs[ox])

    # --- Recovery expression ---

    @m.Expression(m.oxides)
    def recovery(m, ox):
        t_f = sorted(m.t)[-1]
        return (1 - m.m_s[ox, t_f] / (m.mass_solid_feed * m.mass_frac_fs_ox[ox])) * 100

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


def scale_model(m):
    """
    Set scaling factors for all variables and constraints.

    Liquid concentration scaling factors are derived from expected (guess) final
    concentrations (in mol/L).  Solid mass and rate scaling factors are
    derived from initial masses and an estimate of the initial reaction rate.
    """
    # ------------------------------------------------------------------
    # Liquid concentration scaling [mol/L]
    # sf ~ 1 / (expected/guess magnitude in mol/L)
    # ------------------------------------------------------------------
    C_sf = {
        "H2O": 1 / 55.56,  # ~55.56 mol/L
        "H": 10.0,  # ~0.10  mol/L
        "HSO4": 2.0,  # ~0.5   mol/L (Ka2 equilibrium)
        "SO4": 20.0,  # ~0.05  mol/L
        # REEs: expected (guess) final concentrations converted from mg/L via MW
        "Sc": 9e3,  # ~1.1e-4 mol/L
        "Y": 1.8e4,  # ~5.6e-5 mol/L
        "La": 1.4e5,  # ~7.2e-6 mol/L
        "Ce": 2.8e4,  # ~3.6e-5 mol/L
        "Pr": 2.8e5,  # ~3.5e-6 mol/L
        "Nd": 7.2e4,  # ~1.4e-5 mol/L
        "Sm": 1.5e6,  # ~6.7e-7 mol/L
        "Gd": 1.6e5,  # ~6.4e-6 mol/L
        "Dy": 3.2e6,  # ~3.1e-7 mol/L
        # Contaminants
        "Al": 27.0,  # ~0.037 mol/L
        "Ca": 8e2,  # ~1.2e-3 mol/L
        "Fe": 5.6e2,  # ~1.8e-3 mol/L
    }

    for j in ALL_LIQ:
        sf = C_sf[j]
        for t in m.t:
            iscale.set_scaling_factor(m.C[j, t], sf)
            iscale.set_scaling_factor(m.dC_dt[j, t], sf)

    # ------------------------------------------------------------------
    # Solid oxide mass scaling [kg]
    # sf = 1 / initial_mass  (initial mass is the largest it will ever be)
    # ------------------------------------------------------------------
    sf_ms = {}
    for ox in OXIDES:
        m_s0 = M_S0 * mass_frac_fs[ox]
        sf_ms[ox] = 1.0 / m_s0
        for t in m.t:
            iscale.set_scaling_factor(m.m_s[ox, t], sf_ms[ox])
            iscale.set_scaling_factor(m.dm_s_dt[ox, t], sf_ms[ox])

    # ------------------------------------------------------------------
    # Reaction extent rate scaling [mol/hr]
    # Estimate from initial rate: r_ext ~ V_L * eps * B * C_H^A
    # ------------------------------------------------------------------
    eps0 = value(m.pulp_density[m.t.first()])
    C_H0 = C0["H"]

    sf_r = {}
    for ox in OXIDES:
        r0 = V_L * eps0 * B_PARAM[ox] * C_H0 ** A_PARAM[ox]
        sf_r[ox] = 1.0 / max(r0, 1e-12)
        for t in m.t:
            iscale.set_scaling_factor(m.r_ext[ox, t], sf_r[ox])

    # ------------------------------------------------------------------
    # Inherent Ka2 rate scaling [mol/hr]
    # Magnitude ~ total H+ consumption rate from all leaching reactions
    # ------------------------------------------------------------------
    r_inher_mag = sum(
        abs(NU_LIQUID.get((ox, "H"), 0))
        * V_L
        * eps0
        * B_PARAM[ox]
        * C_H0 ** A_PARAM[ox]
        for ox in OXIDES
    )
    sf_ri = 1.0 / max(r_inher_mag, 1e-12)
    for t in m.t:
        iscale.set_scaling_factor(m.r_inher[t], sf_ri)

    # ------------------------------------------------------------------
    # Conversion scaling
    # ------------------------------------------------------------------
    for ox in OXIDES:
        for t in m.t:
            iscale.set_scaling_factor(m.X[ox, t], 1.0)

    # ------------------------------------------------------------------
    # Constraint scaling
    # ------------------------------------------------------------------
    # Liquid ODE constraints: scale by the corresponding C scaling factor
    for j in LIQ_REE_IMP:
        for t in m.t:
            iscale.constraint_scaling_transform(
                m.liq_ree_imp[j, t], C_sf[j], overwrite=True
            )
    for t in m.t:
        iscale.constraint_scaling_transform(m.liq_h2o[t], C_sf["H2O"], overwrite=True)
        iscale.constraint_scaling_transform(m.liq_H[t], C_sf["H"], overwrite=True)
        iscale.constraint_scaling_transform(m.liq_HSO4[t], C_sf["HSO4"], overwrite=True)
        iscale.constraint_scaling_transform(m.liq_SO4[t], C_sf["SO4"], overwrite=True)

    # Solid ODE constraints: scale by sf_ms
    for ox in OXIDES:
        for t in m.t:
            iscale.constraint_scaling_transform(
                m.solid[ox, t], sf_ms[ox], overwrite=True
            )

    # Ka2 equilibrium: both sides ~ C_H * C_SO4 ~ 0.1 * 0.05 = 5e-3 mol²/L²
    sf_ka2 = C_sf["H"] * C_sf["SO4"]
    for t in m.t:
        iscale.constraint_scaling_transform(
            m.ka2_equilibrium[t], sf_ka2, overwrite=True
        )

    # Rate expression: scale by sf_r[ox]
    for ox in OXIDES:
        for t in m.t:
            iscale.constraint_scaling_transform(
                m.rate_eqn[ox, t], sf_r[ox], overwrite=True
            )

    # Conversion definition: scale by sf_ms[ox]
    for ox in OXIDES:
        for t in m.t:
            iscale.constraint_scaling_transform(
                m.conversion_def[ox, t], sf_ms[ox], overwrite=True
            )


def discretise_and_solve(m):
    """Apply backward-difference discretisation and solve with IPOPT."""
    TransformationFactory("dae.finite_difference").apply_to(
        m, nfe=N_FE, wrt=m.t, scheme="BACKWARD"
    )

    m.r_inher[m.t.first()].fix(0)

    scale_model(m)
    _initialize(m)

    solver = get_solver()
    solver.options["max_iter"] = 500
    results = solver.solve(m, tee=True)
    assert_optimal_termination(results)
    return results


def print_results(m):
    """Print final solid recovery and liquid concentrations."""
    t_f = sorted(m.t)[-1]

    print("\n" + "=" * 60)
    print("Batch Leaching Results  (Pyomo DAE model)")
    print("=" * 60)
    print(f"Liquid volume  : {V_L:.1f} L")
    print(f"Batch time     : {duration:.1f} hr")
    print(f"Total solid    : {M_S0:.2f} kg")

    print(
        f"\n  {'Oxide':<10}  {'Init mass (kg)':>15}  {'Final mass (kg)':>16}  {'Recovery (%)':>13}"
    )
    print("  " + "-" * 57)
    for ox in OXIDES:
        m_in = M_S0 * mass_frac_fs[ox]
        m_out = value(m.m_s[ox, t_f])
        rec = value(m.recovery[ox])
        print(f"  {ox:<10}  {m_in:>15.6f}  {m_out:>16.6f}  {rec:>13.4f}")

    print("\nFinal liquid concentrations [mol/L]:")
    for j in ALL_LIQ:
        print(f"  {j:<8}  {value(m.C[j, t_f]):.6e}")
    print("=" * 60)


if __name__ == "__main__":
    m = build_model()
    discretise_and_solve(m)
    print_results(m)
