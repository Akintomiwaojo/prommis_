from pyomo.environ import (
    ConcreteModel,
    NonNegativeReals,
    Param,
    Set,
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

A_PARAM = {
    # "Al2O3": 1.496716606,
    # "Fe2O3": 0.902948175,
    # "CaO": 0.159744406,
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
    # "Al2O3": 405.5050676,
    # "Fe2O3": 11.34710708,
    # "CaO": 0.081844698,
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

# Initial liquid concentrations [mol/L]  (0.1 M H2SO4, metals near zero)
C0 = {
    "H2O": 1e6 / 18e3,  # ~55.56 mol/L
    "H": 0.2,  # 2 * 0.1 mol/L
    "HSO4": 1e-13,
    "SO4": 0.1,
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

    @m.Expression(m.t)
    def m_s_total(m, t):
        return sum(m.m_s[ox, t] for ox in OXIDES)

    ## modify the denominator to be total reactor volume
    @m.Expression(m.t)
    def pulp_density(m, t):
        return m.m_s_total[t] / (m.mass_solid_feed / m.coal_density + m.V_L)

    # shrinking-core rate expression
    # r_ext[ox] = V_L * pulp_density(t) * B[ox] * C[H]^A[ox] * (1 - X[ox])^(2/3)
    @m.Constraint(m.oxides, m.t)
    def rate_eqn(m, ox, t):

        return m.r_ext[ox, t] == (
            (m.mass_solid_feed / m.coal_density + m.V_L)
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

    # need to modify

    m.ree = Set(initialize=LIQ_REE_IMP, ordered=True)

    @m.Expression(m.ree)
    def recovery(m, ree):
        t_f = m.t.last()
        ox = REE_TO_OX[ree]
        c_L_V_L = m.C[ree, t_f] * m.V_L  # mol in liquid
        c_SR_m_SR = m.m_s[ox, t_f] / m.MW_ox[ox] * NU_REE[ox]  # mol in solid residue
        return (c_L_V_L / (c_L_V_L + c_SR_m_SR)) * 100

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
        "Sc": 9e3,
        "Y": 2e4,
        "La": 1e5,
        "Ce": 3e4,
        "Pr": 3e5,
        "Nd": 7e4,
        "Sm": 2e6,
        "Gd": 2e5,
        "Dy": 3e6,
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
    solver.options["halt_on_ampl_error"] = "yes"
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
    fig, ax = plt.subplots(figsize=(9, 5))
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
    DURATIONS = list(range(1, time))
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
    batch(time=24)
