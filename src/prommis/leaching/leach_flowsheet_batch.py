#####################################################################################################
# "PrOMMiS" was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# ("PrOMMiS") initiative, and is copyright (c) 2023-2025 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""
Sample batch flowsheet for BatchLeachingTrain unit model using
parameters and data for West Kentucky No. 13 coal refuse.

The model is a closed-system batch: no continuous inflow or outflow.
The initial state is fully specified by:

  - ``liquid_cv.volume``                          : volume of leach solution [m3]
  - ``batch_time``                                : duration of batch [hr]
  - ``solid_mass_total``                          : total solid charged [kg]
  - ``solid_in.mass_frac_comp``                   : initial solid composition [-]
  - ``liquid_cv.properties_in.conc_mass_comp``    : initial liquid concentrations [mg/L]

Authors: Andrew Lee, Arkoprabho Dasgupta, Akintomiwa Ojo, Douglas Allan
"""

from pyomo.environ import (
    ConcreteModel,
    SolverFactory,
    assert_optimal_termination,
    units,
    value,
)

from idaes.core import FlowsheetBlock
from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    number_total_constraints,
    number_unused_variables,
    number_variables,
)

from prommis.leaching.leach_train_batch import BatchLeachingTrain
from prommis.leaching.leach_reactions import CoalRefuseLeachingReactionParameterBlock
from prommis.properties.coal_refuse_properties import CoalRefuseParameters
from prommis.properties.sulfuric_acid_leaching_properties import (
    SulfuricAcidLeachingParameters,
)

# -------------------------------------------------------------------------------------
# Reference data — West Kentucky No. 13 coal refuse / UKy pilot plant
# -------------------------------------------------------------------------------------

# Batch liquid volume [m3] — equivalent to one hour of 224.3 L/hr continuous feed
LIQUID_VOLUME_M3 = 224.3e-3  # 0.2243 m3 = 224.3 L

# Batch duration [hr]
BATCH_TIME_HR = 1.0

# Total solid mass charged to vessel [kg] — one hour of 22.68 kg/hr continuous feed
SOLID_MASS_TOTAL_KG = 22.68

# Initial solid mass fractions
SOLID_MASS_FRAC = {
    "inerts":  0.6952,
    "Al2O3":   0.237,
    "Fe2O3":   0.0642,
    "CaO":     3.31e-3,
    "Sc2O3":   2.77966e-05,
    "Y2O3":    3.28653e-05,
    "La2O3":   6.77769e-05,
    "Ce2O3":   0.000156161,
    "Pr2O3":   1.71438e-05,
    "Nd2O3":   6.76618e-05,
    "Sm2O3":   1.47926e-05,
    "Gd2O3":   1.0405e-05,
    "Dy2O3":   7.54827e-06,
}

# Initial liquid concentrations [mg/L] — 0.05 M H2SO4
LIQUID_CONC_MG_L = {
    "H2O":  1e6,
    "H":    2 * 0.05 * 1e3,
    "HSO4": 1e-8,
    "SO4":  0.05 * 96e3,
}


def build_batch_leaching_flowsheet():
    """
    Build and return a fully specified batch leaching flowsheet.

    Returns
    -------
    m : ConcreteModel
        Model with ``m.fs`` containing the parameter blocks and ``m.fs.leach``
        (a :class:`BatchLeachingTrain`) with all DOFs fixed and ready to solve.
    """
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    # --- Parameter / property blocks ---
    m.fs.leach_soln = SulfuricAcidLeachingParameters()
    m.fs.coal = CoalRefuseParameters()
    m.fs.leach_rxns = CoalRefuseLeachingReactionParameterBlock()

    # --- Unit model ---
    m.fs.leach = BatchLeachingTrain(
        liquid_phase={
            "property_package": m.fs.leach_soln,
            "has_energy_balance": False,
            "has_pressure_balance": False,
        },
        solid_phase={
            "property_package": m.fs.coal,
            "has_energy_balance": False,
            "has_pressure_balance": False,
        },
        reaction_package=m.fs.leach_rxns,
    )

    leach = m.fs.leach

    # --- Batch geometry and timing ---
    leach.liquid_cv.volume.fix(LIQUID_VOLUME_M3 * units.m**3)
    leach.batch_time.fix(BATCH_TIME_HR * units.hour)

    # --- Initial solid charge ---
    leach.solid_mass_total.fix(SOLID_MASS_TOTAL_KG * units.kg)
    for j, xj in SOLID_MASS_FRAC.items():
        leach.solid_cv.properties_in[0].mass_frac_comp[j].fix(xj * units.kg / units.kg)

    # --- Initial liquid concentrations ---
    # All metal species start at near-zero
    leach.liquid_cv.properties_in[0].conc_mass_comp.fix(1e-10 * units.mg / units.L)
    for j, c in LIQUID_CONC_MG_L.items():
        leach.liquid_cv.properties_in[0].conc_mass_comp[j].fix(c * units.mg / units.L)

    # Temperature and pressure (no energy or pressure balances in this flowsheet)
    leach.liquid_cv.properties_in[0].temperature.fix(298.15 * units.K)
    leach.liquid_cv.properties_in[0].pressure.fix(101325 * units.Pa)
    leach.liquid_cv.properties_out[0].temperature.fix(298.15 * units.K)
    leach.liquid_cv.properties_out[0].pressure.fix(101325 * units.Pa)

    return m


def initialize_and_solve(m):
    """
    Initialize and solve the batch leaching flowsheet with IPOPT.

    Initialization seeds the final state (solid_out, properties_out) from
    the initial state (solid_in, properties_in) so that IPOPT starts near
    a physically reasonable point.

    Parameters
    ----------
    m : ConcreteModel
        Model returned by :func:`build_batch_leaching_flowsheet`.

    Returns
    -------
    results
        Solver termination results.
    """
    leach = m.fs.leach
    t = 0

    # Seed solid_cv.properties_out from known initial values
    flow_mass_ref = SOLID_MASS_TOTAL_KG / BATCH_TIME_HR  # kg/hr
    leach.solid_cv.properties_out[t].flow_mass.set_value(flow_mass_ref)
    for j, xj in SOLID_MASS_FRAC.items():
        leach.solid_cv.properties_out[t].mass_frac_comp[j].set_value(xj)

    # Seed liquid_cv.properties_out concentrations from properties_in
    for j, c in LIQUID_CONC_MG_L.items():
        leach.liquid_cv.properties_out[t].conc_mass_comp[j].set_value(c)

    solver = SolverFactory("ipopt_v2")
    results = solver.solve(m, tee=False)
    assert_optimal_termination(results)
    return results


def print_results(m):
    """
    Print recovery and final solid/liquid state for each component.

    Parameters
    ----------
    m : ConcreteModel
        Solved model.
    """
    leach = m.fs.leach
    t = 0  # steady-state time index

    print("\n" + "=" * 65)
    print("Batch Leaching Results")
    print("=" * 65)
    print(
        f"Liquid volume : "
        f"{value(units.convert(leach.liquid_cv.volume[t], to_units=units.L)):.1f} L"
    )
    print(f"Batch time    : {value(leach.batch_time[t]):.1f} hr")
    print(f"Total solid   : {value(leach.solid_mass_total[t]):.2f} kg")

    # Initial and final solid masses per component (kg = flow_mass [kg/hr] * batch_time [hr])
    bt = value(leach.batch_time[t])
    print()
    print(
        f"{'Component':<12}  {'Init. mass (kg)':>16}  "
        f"{'Final mass (kg)':>16}  {'Recovery (%)':>13}"
    )
    print("-" * 65)
    for j in m.fs.coal.component_list:
        m_in  = value(leach.solid_cv.properties_in[t].flow_mass  * leach.solid_cv.properties_in[t].mass_frac_comp[j])  * bt
        m_out = value(leach.solid_cv.properties_out[t].flow_mass * leach.solid_cv.properties_out[t].mass_frac_comp[j]) * bt
        rec   = value(leach.recovery[t, j])
        print(f"{j:<12}  {m_in:>16.6f}  {m_out:>16.6f}  {rec:>13.4f}")
    print("=" * 65)

    print("\nFinal liquid concentrations [mg/L]:")
    for j in m.fs.leach_soln.component_list:
        c = value(
            units.convert(
                leach.liquid_cv.properties_out[t].conc_mass_comp[j],
                to_units=units.mg / units.L,
            )
        )
        print(f"  {j:<10} : {c:.4e}")


# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("Building batch leaching flowsheet...")
    m = build_batch_leaching_flowsheet()

    print(
        f"Model statistics: "
        f"{number_variables(m.fs.leach)} vars, "
        f"{number_total_constraints(m.fs.leach)} constraints, "
        f"{number_unused_variables(m.fs.leach)} unused vars, "
        f"DOF = {degrees_of_freedom(m.fs.leach)}"
    )

    print("Solving...")
    initialize_and_solve(m)

    print_results(m)
