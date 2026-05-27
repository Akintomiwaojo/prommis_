from pyomo.environ import (
    ComponentMap,
    ConcreteModel,
    SolverFactory,
    Suffix,
    TransformationFactory,
    units,
    Set,
    Objective,
    minimize,
)
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

from idaes.core import FlowsheetBlock
from idaes.core.scaling import CustomScalerBase

from prommis.leaching.leach_train import LeachingTrain, LeachingTrainInitializer
from prommis.leaching.param_est_new_model.leach_reactions_combine import (
    CoalRefuseLeachingCombinedReactionParameterBlock,
)
from prommis.leaching.leach_reactions import CoalRefuseLeachingReactionParameterBlock
from prommis.properties.coal_refuse_properties import (
    CoalRefuseParameters,
    CoalRefusePropertiesScaler,
)
from prommis.properties.sulfuric_acid_leaching_properties import (
    SulfuricAcidLeachingParameters,
    SulfuricAcidLeachingPropertiesScaler,
)

from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# Shared scaling helpers (used by both the parameter estimation and simulation
# models below)
# ---------------------------------------------------------------------------

# Expected outlet concentrations [mg/L] at ~20 % recovery, S/L = 1/10.
# Computed from solid feed composition × stoichiometry / liquid flow rate.
_liq_outlet_sf = {
    "Al": 1 / 2500,
    "Fe": 1 / 900,
    "Ca": 1 / 50,
    "Sc": 1 / 0.35,
    "Y": 1 / 0.52,
    "La": 1 / 1.2,
    "Ce": 1 / 2.7,
    "Pr": 1 / 0.30,
    "Nd": 1 / 1.2,
    "Sm": 1 / 0.26,
    "Gd": 1 / 0.18,
    "Dy": 1 / 0.13,
}


def _scale_leach_train_blocks(model, csb, op_cond, solid_feed, acid_conc):
    """Apply physical-state scaling to all leach train sub-blocks.

    Covers volume, solid inlet/outlet (flow mass, mass fractions, conversions),
    and liquid inlet/outlet (flow rate, acid species, all leached species).
    Delegates ``conc_mol_comp`` and constraint scaling to the property-block
    scalers so nothing is missed.

    Parameters
    ----------
    model : ConcreteModel
        Top-level model whose ``scaling_factor`` Suffix will be populated.
    csb : CustomScalerBase
        Scaler instance used to set variable scaling factors.
    op_cond : Set
        Pyomo Set of operating-condition labels.
    solid_feed : dict
        Map from operating-condition label to solid feed flow rate [kg/hr].
    acid_conc : dict
        Map from operating-condition label to H2SO4 concentration [mol/L].
    """
    liq_scaler = SulfuricAcidLeachingPropertiesScaler()
    solid_scaler = CoalRefusePropertiesScaler()

    for s in op_cond:
        # Volume (100 gal ≈ 378.5 L)
        csb.set_variable_scaling_factor(model.fs.leach[s].volume[0, 1], 1 / 378.5)

        # Solid inlet: flow mass
        for blk in model.fs.leach[s].mscontactor.solid_inlet_state.values():
            csb.set_variable_scaling_factor(blk.flow_mass, 1 / solid_feed[s])
            solid_scaler.variable_scaling_routine(blk, overwrite=False)

        # Solid outlet: flow mass, mass fractions (by feed composition), conversions
        for blk in model.fs.leach[s].mscontactor.solid.values():
            csb.set_variable_scaling_factor(blk.flow_mass, 1 / solid_feed[s])
            for comp in model.fs.coal.component_list:
                x0 = model.fs.coal.mass_frac_comp_initial[comp].value
                csb.set_variable_scaling_factor(blk.mass_frac_comp[comp], 1 / x0)
            if hasattr(blk, "conversion_comp"):
                for comp in model.fs.coal.component_list:
                    csb.set_variable_scaling_factor(blk.conversion_comp[comp], 1)
            solid_scaler.constraint_scaling_routine(blk, overwrite=False)

        # Liquid inlet: condition-specific H+ and SO4; scaler fills conc_mol_comp
        for blk in model.fs.leach[s].mscontactor.liquid_inlet_state.values():
            csb.set_variable_scaling_factor(blk.flow_vol, 1 / 224.3)
            csb.set_variable_scaling_factor(
                blk.conc_mass_comp["H"], 1 / (2 * acid_conc[s] * 1e3)
            )
            csb.set_variable_scaling_factor(
                blk.conc_mass_comp["SO4"], 1 / (acid_conc[s] * 96e3)
            )
            liq_scaler.variable_scaling_routine(blk, overwrite=False)

        # Liquid outlet: all leached species; scaler handles conc_mol_comp
        # and molar_concentration / hso4_dissociation constraints
        for blk in model.fs.leach[s].mscontactor.liquid.values():
            csb.set_variable_scaling_factor(blk.flow_vol, 1 / 224.3)
            csb.set_variable_scaling_factor(
                blk.conc_mass_comp["H"], 1 / (2 * acid_conc[s] * 1e3)
            )
            csb.set_variable_scaling_factor(
                blk.conc_mass_comp["SO4"], 1 / (acid_conc[s] * 96e3)
            )
            for comp, sf in _liq_outlet_sf.items():
                csb.set_variable_scaling_factor(blk.conc_mass_comp[comp], sf)
            liq_scaler.variable_scaling_routine(blk, overwrite=False)
            liq_scaler.constraint_scaling_routine(blk, overwrite=False)


# ---------------------------------------------------------------------------
# Read Data
# ---------------------------------------------------------------------------
data = pd.read_csv("parameter estimation.csv")
data = data.dropna(axis=1, how="all")
data = data.set_index(data.columns[0])


# ---------------------------------------------------------------------------
# Parameter estimation model (m) — combined SCM with fitted A_ox, k_prime,
# K_film, D_e
# ---------------------------------------------------------------------------
m = ConcreteModel()

m.OpCond = Set(
    initialize=[
        "0.025M S_L=1/10",
        "0.05M S_L=1/10",
        "0.075M S_L=1/10",
        "0.05M S_L=1.5/10",
        "0.05M S_L=2/10",
        "0.075M S_L=1.5/10",
    ],
    doc="Operating conditions",
)
m.fs = FlowsheetBlock(dynamic=False)

m.fs.leach_soln = SulfuricAcidLeachingParameters()
m.fs.coal = CoalRefuseParameters()
m.fs.leach_rxns = CoalRefuseLeachingCombinedReactionParameterBlock()


m.fs.leach = LeachingTrain(
    m.OpCond,
    number_of_tanks=1,
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

solid_feed_variation = {
    "0.025M S_L=1/10": 22.68,
    "0.05M S_L=1/10": 22.67,
    "0.075M S_L=1/10": 22.68,
    "0.05M S_L=1.5/10": 34.02,
    "0.05M S_L=2/10": 45.36,
    "0.075M S_L=1.5/10": 34.02,
}

acid_conc_variation = {
    "0.025M S_L=1/10": 0.025,
    "0.05M S_L=1/10": 0.05,
    "0.075M S_L=1/10": 0.075,
    "0.05M S_L=1.5/10": 0.05,
    "0.05M S_L=2/10": 0.05,
    "0.075M S_L=1.5/10": 0.075,
}

m.fs.leach[:].liquid_inlet.flow_vol.fix(224.3 * units.L / units.hour)
m.fs.leach[:].liquid_inlet.conc_mass_comp.fix(1e-10 * units.mg / units.L)
m.fs.leach[:].liquid_inlet.conc_mass_comp[:, "H2O"].fix(1e6 * units.mg / units.L)

m.fs.leach[:].liquid_inlet.conc_mass_comp[0, "HSO4"].fix(1e-8 * units.mg / units.L)

for j in m.OpCond:
    m.fs.leach[j].liquid_inlet.conc_mass_comp[0, "H"].fix(
        2 * acid_conc_variation[j] * 1e3 * units.mg / units.L
    )
    m.fs.leach[j].liquid_inlet.conc_mass_comp[0, "SO4"].fix(
        acid_conc_variation[j] * 96e3 * units.mg / units.L
    )
    m.fs.leach[j].solid_inlet.flow_mass.fix(
        solid_feed_variation[j] * units.kg / units.hour
    )

m.fs.leach[:].solid_inlet.mass_frac_comp[0, "inerts"].fix(0.6952 * units.kg / units.kg)
m.fs.leach[:].solid_inlet.mass_frac_comp[0, "Al2O3"].fix(0.237 * units.kg / units.kg)
m.fs.leach[:].solid_inlet.mass_frac_comp[0, "Fe2O3"].fix(0.0642 * units.kg / units.kg)
m.fs.leach[:].solid_inlet.mass_frac_comp[0, "CaO"].fix(3.31e-3 * units.kg / units.kg)
m.fs.leach[:].solid_inlet.mass_frac_comp[0, "Sc2O3"].fix(
    2.77966e-05 * units.kg / units.kg
)
m.fs.leach[:].solid_inlet.mass_frac_comp[0, "Y2O3"].fix(
    3.28653e-05 * units.kg / units.kg
)
m.fs.leach[:].solid_inlet.mass_frac_comp[0, "La2O3"].fix(
    6.77769e-05 * units.kg / units.kg
)
m.fs.leach[:].solid_inlet.mass_frac_comp[0, "Ce2O3"].fix(
    0.000156161 * units.kg / units.kg
)
m.fs.leach[:].solid_inlet.mass_frac_comp[0, "Pr2O3"].fix(
    1.71438e-05 * units.kg / units.kg
)
m.fs.leach[:].solid_inlet.mass_frac_comp[0, "Nd2O3"].fix(
    6.76618e-05 * units.kg / units.kg
)
m.fs.leach[:].solid_inlet.mass_frac_comp[0, "Sm2O3"].fix(
    1.47926e-05 * units.kg / units.kg
)
m.fs.leach[:].solid_inlet.mass_frac_comp[0, "Gd2O3"].fix(
    1.0405e-05 * units.kg / units.kg
)
m.fs.leach[:].solid_inlet.mass_frac_comp[0, "Dy2O3"].fix(
    7.54827e-06 * units.kg / units.kg
)

m.fs.leach[:].volume.fix(100 * units.gallon)


m.scaling_factor = Suffix(direction=Suffix.EXPORT)
csb = CustomScalerBase()

# Reaction parameter variables (the NLP decision variables).
# Scale each by 1/initial_value so the solver sees them near O(1).
# K_film is uniformly initialised at 1e-2 m/hr; D_e at 1e-7 m²/hr.
_k_prime_sf = {
    "Al2O3": 1 / 1e-4,
    "Fe2O3": 1 / 1e-2,
    "CaO": 1 / 0.082,
    "Sc2O3": 1 / 7.55e-4,
    "Y2O3": 1 / 5.29e-4,
    "La2O3": 1 / 1.03e-3,
    "Ce2O3": 1 / 7.05e-3,
    "Pr2O3": 1 / 5.23e-4,
    "Nd2O3": 1 / 6.29e-3,
    "Sm2O3": 1 / 2.52e-4,
    "Gd2O3": 1 / 1.34e-2,
    "Dy2O3": 1 / 4.57e-5,
}
for comp in m.fs.leach_rxns.reaction_idx:
    csb.set_variable_scaling_factor(m.fs.leach_rxns.A_ox[comp], 1)
    csb.set_variable_scaling_factor(m.fs.leach_rxns.k_prime[comp], _k_prime_sf[comp])
    # csb.set_variable_scaling_factor(m.fs.leach_rxns.K_film[comp], 1e2)
    # csb.set_variable_scaling_factor(m.fs.leach_rxns.D_e[comp], 1e7)

_scale_leach_train_blocks(m, csb, m.OpCond, solid_feed_variation, acid_conc_variation)

m.recovery_sse = Objective(
    expr=sum(
        ((m.fs.leach[j].recovery[0, comp] - data.loc[comp, j])) ** 2
        for j in m.OpCond
        for comp in m.fs.coal.component_list - ["inerts"]
    ),
    sense=minimize,
)

# / data.loc[comp, j]

for t in m.fs.leach:
    for s in m.fs.leach[t].mscontactor.liquid:
        m.fs.leach[t].mscontactor.liquid[s].conc_mol_comp["H"].setlb(1e-8)


scaling = TransformationFactory("core.scale_model")
scaled_model = scaling.create_using(m, rename=False)

# Solve scaled model
solver = SolverFactory("ipopt_v2")
solver.options["max_iter"] = 5000
solver.options["halt_on_ampl_error"] = "yes"
solver.options["bound_relax_factor"] = 0
results = solver.solve(scaled_model, tee=True)


scaling.propagate_solution(scaled_model, m)


# ---------------------------------------------------------------------------
# Simulation model (m2) — Andrew's simple shrinking-core model (Params only)
# ---------------------------------------------------------------------------
m2 = ConcreteModel()
m2.OpCond = Set(
    initialize=[
        "0.025M S_L=1/10",
        "0.05M S_L=1/10",
        "0.075M S_L=1/10",
        "0.05M S_L=1.5/10",
        "0.05M S_L=2/10",
        "0.075M S_L=1.5/10",
    ],
)
m2.fs = FlowsheetBlock(dynamic=False)
m2.fs.leach_soln = SulfuricAcidLeachingParameters()
m2.fs.coal = CoalRefuseParameters()
m2.fs.leach_rxns = CoalRefuseLeachingReactionParameterBlock()

m2.fs.leach = LeachingTrain(
    m2.OpCond,
    number_of_tanks=1,
    liquid_phase={
        "property_package": m2.fs.leach_soln,
        "has_energy_balance": False,
        "has_pressure_balance": False,
    },
    solid_phase={
        "property_package": m2.fs.coal,
        "has_energy_balance": False,
        "has_pressure_balance": False,
    },
    reaction_package=m2.fs.leach_rxns,
)

m2.fs.leach[:].liquid_inlet.flow_vol.fix(224.3 * units.L / units.hour)
m2.fs.leach[:].liquid_inlet.conc_mass_comp.fix(1e-10 * units.mg / units.L)
m2.fs.leach[:].liquid_inlet.conc_mass_comp[:, "H2O"].fix(1e6 * units.mg / units.L)
m2.fs.leach[:].liquid_inlet.conc_mass_comp[0, "HSO4"].fix(1e-8 * units.mg / units.L)

for j in m2.OpCond:
    m2.fs.leach[j].liquid_inlet.conc_mass_comp[0, "H"].fix(
        2 * acid_conc_variation[j] * 1e3 * units.mg / units.L
    )
    m2.fs.leach[j].liquid_inlet.conc_mass_comp[0, "SO4"].fix(
        acid_conc_variation[j] * 96e3 * units.mg / units.L
    )
    m2.fs.leach[j].solid_inlet.flow_mass.fix(
        solid_feed_variation[j] * units.kg / units.hour
    )

m2.fs.leach[:].solid_inlet.mass_frac_comp[0, "inerts"].fix(0.6952 * units.kg / units.kg)
m2.fs.leach[:].solid_inlet.mass_frac_comp[0, "Al2O3"].fix(0.237 * units.kg / units.kg)
m2.fs.leach[:].solid_inlet.mass_frac_comp[0, "Fe2O3"].fix(0.0642 * units.kg / units.kg)
m2.fs.leach[:].solid_inlet.mass_frac_comp[0, "CaO"].fix(3.31e-3 * units.kg / units.kg)
m2.fs.leach[:].solid_inlet.mass_frac_comp[0, "Sc2O3"].fix(
    2.77966e-05 * units.kg / units.kg
)
m2.fs.leach[:].solid_inlet.mass_frac_comp[0, "Y2O3"].fix(
    3.28653e-05 * units.kg / units.kg
)
m2.fs.leach[:].solid_inlet.mass_frac_comp[0, "La2O3"].fix(
    6.77769e-05 * units.kg / units.kg
)
m2.fs.leach[:].solid_inlet.mass_frac_comp[0, "Ce2O3"].fix(
    0.000156161 * units.kg / units.kg
)
m2.fs.leach[:].solid_inlet.mass_frac_comp[0, "Pr2O3"].fix(
    1.71438e-05 * units.kg / units.kg
)
m2.fs.leach[:].solid_inlet.mass_frac_comp[0, "Nd2O3"].fix(
    6.76618e-05 * units.kg / units.kg
)
m2.fs.leach[:].solid_inlet.mass_frac_comp[0, "Sm2O3"].fix(
    1.47926e-05 * units.kg / units.kg
)
m2.fs.leach[:].solid_inlet.mass_frac_comp[0, "Gd2O3"].fix(
    1.0405e-05 * units.kg / units.kg
)
m2.fs.leach[:].solid_inlet.mass_frac_comp[0, "Dy2O3"].fix(
    7.54827e-06 * units.kg / units.kg
)
m2.fs.leach[:].volume.fix(100 * units.gallon)


m2.scaling_factor = Suffix(direction=Suffix.EXPORT)
csb2 = CustomScalerBase()

_scale_leach_train_blocks(
    m2, csb2, m2.OpCond, solid_feed_variation, acid_conc_variation
)

scaling2 = TransformationFactory("core.scale_model")
scaled_model2 = scaling2.create_using(m2, rename=False)

solver2 = SolverFactory("ipopt_v2")
solver2.options["max_iter"] = 5000
solver2.solve(scaled_model2, tee=True)

scaling2.propagate_solution(scaled_model2, m2)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
exp_data = {}
for comp in m.fs.coal.component_list - ["inerts"]:
    exp_data[comp] = [data.loc[comp, j] for j in m.OpCond]


model_data = {}
for comp in m.fs.coal.component_list - ["inerts"]:
    model_data[comp] = [m.fs.leach[j].recovery[0, comp]() for j in m.OpCond]

model_data_simple = {}
for comp in m2.fs.coal.component_list - ["inerts"]:
    model_data_simple[comp] = [m2.fs.leach[j].recovery[0, comp]() for j in m2.OpCond]


for comp in m.fs.coal.component_list - ["inerts"]:
    y_exp = np.array(exp_data[comp])
    y_mod = np.array(model_data[comp])
    y_mod_sim = np.array(model_data_simple[comp])

    r2 = r2_score(y_exp, y_mod)
    r2_sim = r2_score(y_exp, y_mod_sim)

    plt.figure(figsize=(5, 5), dpi=100)
    plt.scatter(
        y_exp,
        y_mod,
        label="Current Model",
        marker="o",
        color="#1f77b4",
    )

    plt.scatter(
        y_exp,
        y_mod_sim,
        label="Andrew's Model",
        marker="s",
        color="#ff7f0e",
    )

    plt.plot(
        y_exp,
        y_exp,
        label="Parity line (y = x)",
        color="k",
    )

    plt.xlabel("Experimental Recovery (%)")
    plt.ylabel("Model Recovery (%)")
    plt.title("Parity Plot: Experimental vs Model Extraction for " + comp)
    plt.legend()
    plt.text(
        0.05,
        0.95,
        f"Current model: $R^2 = {r2:.4f}$",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
    )
    plt.text(
        0.05,
        0.88,
        f"Andrew's model: $R^2 = {r2_sim:.4f}$",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Print fitted parameters
# ---------------------------------------------------------------------------
print("\n=== Fitted Parameters ===")
print(f"{'Oxide':<10} {'A_ox':>12} {'k_prime':>14} {'K_film':>14} {'D_e':>14}")
print("-" * 66)
for comp in m.fs.coal.component_list - ["inerts"]:
    print(
        f"{comp:<10} "
        f"{m.fs.leach_rxns.A_ox[comp].value:>12.6f} "
        f"{m.fs.leach_rxns.k_prime[comp].value:>14.6e} "
        f"{m.fs.leach_rxns.K_film[comp].value:>14.6e} "
        f"{m.fs.leach_rxns.D_e[comp].value:>14.6e}"
    )
