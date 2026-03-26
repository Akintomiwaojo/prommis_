"""
Single-NLP parameter estimation for Yang-Honaker leaching kinetics.

A and B kinetic parameters are Pyomo Vars. The model runs 4 acid-concentration
scenarios simultaneously (multi-scenario DAE), sharing A and B across all
scenarios. IPOPT minimises the SSE between model and experimental recoveries.

Reference data: yang_data.csv  (Yang & Honaker, 2020)
"""

from pyomo.environ import (
    ConcreteModel,
    NonNegativeReals,
    Objective,
    Param,
    Set,
    TransformationFactory,
    Var,
    assert_optimal_termination,
    minimize,
    value,
    units,
)
from pyomo.dae import ContinuousSet, DerivativeVar
from idaes.core.solvers import get_solver
import pandas as pd
from pathlib import Path


duration = 2.0  # batch duration [hr]
N_FE = 20  # finite elements
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

# M2O3 + 6H+ -> 2M^3+ + 3H2O
NU_LIQUID = {
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
    "H2O": 1e6 / 18e3,
    "H": 0.2,
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
NU_REE = {ox: 2 for ox in OXIDES}


# read data from csv; extract acid concentrations and experimental recoveries
_data = pd.read_csv(Path(__file__).parent / "yang_data.csv", index_col="Element")
ACID_CONCS = [float(c.replace("M H2SO4", "")) for c in _data.columns]
N_SCEN = len(ACID_CONCS)
SCENARIOS = list(range(N_SCEN))

# Keep only REEs present in the model; values are fractional (0–1)
EXP_RECOVERY_FRAC = {ree: _data.loc[ree].tolist() for ree in LIQ_REE_IMP}

# Initial H+ and SO4 concentrations per scenario (complete H2SO4 dissociation)
C0_H = {s: 2.0 * ACID_CONCS[s] for s in SCENARIOS}
C0_SO4 = {s: ACID_CONCS[s] for s in SCENARIOS}


def build_param_estimation_model():
    m = ConcreteModel()

    # --- Sets ---
    m.t = ContinuousSet(bounds=(0, duration))
    m.oxides = Set(initialize=OXIDES, ordered=True)
    m.liq = Set(initialize=ALL_LIQ, ordered=True)
    m.ree = Set(initialize=LIQ_REE_IMP, ordered=True)
    m.scenarios = Set(initialize=SCENARIOS, ordered=True)

    # --- Now Vars to be estimated ---
    m.A = Var(
        m.oxides,
        initialize=A_PARAM,
        bounds=(0.1, 5.0),
        doc="Reaction order wrt [H+]",
    )
    m.B = Var(
        m.oxides,
        initialize=B_PARAM,
        bounds=(1e-10, 1e2),
        units=units.mol / units.kg / units.hr,
    )

    # Fixed physical parameters
    m.V_L = Param(initialize=V_L, units=units.L)
    m.Ka2 = Param(initialize=10**-1.99, units=units.mol / units.L)
    m.MW_ox = Param(m.oxides, initialize=MW_OX, units=units.kg / units.mol)
    m.mass_solid_feed = Param(initialize=M_S0, units=units.kg)
    m.coal_density = Param(initialize=2.4, units=units.kg / units.L)
    m.mass_frac_fs_ox = Param(
        m.oxides, initialize={ox: mass_frac_fs[ox] for ox in OXIDES}
    )

    # Experimental recoveries as Params [%]
    m.exp_recovery = Param(
        m.scenarios,
        m.ree,
        initialize={
            (s, ree): EXP_RECOVERY_FRAC[ree][s] * 100
            for s in SCENARIOS
            for ree in LIQ_REE_IMP
        },
        doc="Experimental leach recovery [%]",
    )

    # --- Scenario-indexed differential variables ---

    def _C_init(m, s, j, t):
        if j == "H":
            return C0_H[s]
        if j == "SO4":
            return C0_SO4[s]
        return C0[j]

    m.C = Var(
        m.scenarios,
        m.liq,
        m.t,
        within=NonNegativeReals,
        initialize=_C_init,
        units=units.mol / units.L,
    )
    m.dC_dt = DerivativeVar(m.C, wrt=m.t, units=units.mol / units.L / units.hr)

    m.m_s = Var(
        m.scenarios,
        m.oxides,
        m.t,
        within=NonNegativeReals,
        initialize=lambda m, s, ox, t: M_S0 * mass_frac_fs[ox],
        units=units.kg,
    )
    m.dm_s_dt = DerivativeVar(m.m_s, wrt=m.t, units=units.kg / units.hr)

    # --- Algebraic variables ---
    m.r_ext = Var(
        m.scenarios,
        m.oxides,
        m.t,
        within=NonNegativeReals,
        initialize=0,
        units=units.mol / units.hr,
    )
    m.r_inher = Var(
        m.scenarios,
        m.t,
        initialize=1e-10,
        units=units.mol / units.hr,
    )
    m.X = Var(
        m.scenarios,
        m.oxides,
        m.t,
        bounds=(0, 1),
        initialize=0.01,
    )

    # --- Constraints ---

    @m.Constraint(m.scenarios, LIQ_REE_IMP, m.t)
    def liq_ree_imp(m, s, j, t):
        rhs = sum(
            NU_LIQUID[ox, j] * m.r_ext[s, ox, t]
            for ox in OXIDES
            if (ox, j) in NU_LIQUID
        )
        return m.V_L * m.dC_dt[s, j, t] == rhs

    @m.Constraint(m.scenarios, m.t)
    def liq_h2o(m, s, t):
        return m.V_L * m.dC_dt[s, "H2O", t] == sum(
            NU_LIQUID[ox, "H2O"] * m.r_ext[s, ox, t] for ox in OXIDES
        )

    @m.Constraint(m.scenarios, m.t)
    def liq_H(m, s, t):
        r_L_H = sum(NU_LIQUID[ox, "H"] * m.r_ext[s, ox, t] for ox in OXIDES)
        return m.V_L * m.dC_dt[s, "H", t] == r_L_H + m.r_inher[s, t]

    @m.Constraint(m.scenarios, m.t)
    def liq_HSO4(m, s, t):
        return m.V_L * m.dC_dt[s, "HSO4", t] == -m.r_inher[s, t]

    @m.Constraint(m.scenarios, m.t)
    def liq_SO4(m, s, t):
        return m.V_L * m.dC_dt[s, "SO4", t] == m.r_inher[s, t]

    @m.Constraint(m.scenarios, m.oxides, m.t)
    def solid(m, s, ox, t):
        return m.dm_s_dt[s, ox, t] == -m.r_ext[s, ox, t] * m.MW_ox[ox]

    @m.Constraint(m.scenarios, m.t)
    def ka2_equilibrium(m, s, t):
        return m.C[s, "HSO4", t] * m.Ka2 == m.C[s, "H", t] * m.C[s, "SO4", t]

    @m.Constraint(m.scenarios, m.oxides, m.t)
    def conversion_def(m, s, ox, t):
        return (1 - m.X[s, ox, t]) * m.mass_solid_feed * m.mass_frac_fs_ox[ox] == m.m_s[
            s, ox, t
        ]

    @m.Expression(m.scenarios, m.t)
    def m_s_total(m, s, t):
        return sum(m.m_s[s, ox, t] for ox in OXIDES)

    @m.Expression(m.scenarios, m.t)
    def pulp_density(m, s, t):
        return m.m_s_total[s, t] / (m.mass_solid_feed / m.coal_density + m.V_L)

    @m.Constraint(m.scenarios, m.oxides, m.t)
    def rate_eqn(m, s, ox, t):
        return m.r_ext[s, ox, t] == (
            (m.V_L + (m.mass_solid_feed / m.coal_density))
            * m.pulp_density[s, t]
            * m.B[ox]
            * (m.C[s, "H", t] / (units.mol / units.L)) ** m.A[ox]
            * (1 - m.X[s, ox, t]) ** (2 / 3)
        )

    # --- Initial conditions (scenario-dependent H and SO4) ---
    for s in SCENARIOS:
        for j in ALL_LIQ:
            if j == "HSO4":
                continue
            ic = C0_H[s] if j == "H" else C0_SO4[s] if j == "SO4" else C0[j]
            m.C[s, j, 0].fix(ic)
        for ox in OXIDES:
            m.m_s[s, ox, 0].fix(M_S0 * mass_frac_fs[ox])

    # --- Recovery expression per scenario ---
    @m.Expression(m.scenarios, m.ree)
    def recovery(m, s, ree):
        t_f = m.t.last()
        ox = REE_TO_OX[ree]
        c_L = m.C[s, ree, t_f] * m.V_L
        c_SR = m.m_s[s, ox, t_f] / m.MW_ox[ox] * NU_REE[ox]
        return (c_L / (c_L + c_SR)) * 100

    # --- Objective: minimise SSE ---

    # TODO(human): define a Pyomo Objective named m.obj that minimises the
    # sum of squared errors between m.recovery[s, ree] and m.exp_recovery[s, ree]
    # over all scenarios in m.scenarios and all REEs in m.ree.

    return m


def discretise_and_solve(m):
    TransformationFactory("dae.finite_difference").apply_to(
        m, nfe=N_FE, wrt=m.t, scheme="BACKWARD"
    )

    for s in SCENARIOS:
        m.r_inher[s, m.t.first()].fix(0)

    solver = get_solver()
    solver.options["max_iter"] = 1000
    solver.options["halt_on_ampl_error"] = "yes"
    results = solver.solve(m, tee=True)
    assert_optimal_termination(results)
    return results


def print_results(m):
    print("\n" + "=" * 70)
    print("Parameter Estimation Results")
    print("=" * 70)
    print(
        f"\n{'Oxide':<10}  {'A (fitted)':>12}  {'A (lit)':>10}  {'B (fitted)':>14}  {'B (lit)':>12}"
    )
    print("-" * 64)
    for ox in OXIDES:
        print(
            f"{ox:<10}  {value(m.A[ox]):>12.4f}  {A_PARAM[ox]:>10.4f}"
            f"  {value(m.B[ox]):>14.6e}  {B_PARAM[ox]:>12.6e}"
        )

    print(f"\n{'':10}  ", end="")
    for c in ACID_CONCS:
        print(f"  {c}M H2SO4", end="")
    print()
    print("-" * 70)
    for ree in LIQ_REE_IMP:
        print(f"{ree:<10}", end="")
        for s in SCENARIOS:
            mod = value(m.recovery[s, ree])
            exp = value(m.exp_recovery[s, ree])
            print(f"  {mod:5.1f}% (exp {exp:4.1f}%)", end="")
        print()


if __name__ == "__main__":
    m = build_param_estimation_model()
    discretise_and_solve(m)
    print_results(m)
