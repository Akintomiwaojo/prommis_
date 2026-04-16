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
from prommis.leaching.leach_reactions_combine import (
    CoalRefuseLeachingCombinedReactionParameterBlock,
)
from prommis.leaching.leach_reactions import CoalRefuseLeachingReactionParameterBlock
from prommis.properties.coal_refuse_properties import CoalRefuseParameters
from prommis.properties.sulfuric_acid_leaching_properties import (
    SulfuricAcidLeachingParameters,
)

from sklearn.metrics import r2_score

data = pd.read_csv("parameter estimation.csv")
data = data.dropna(axis=1, how="all")
data = data.set_index(data.columns[0])
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

# param_ele_list = [
#     "Fe2O3",
#     "CaO",
#     "Sc2O3",
#     "Y2O3",
#     "La2O3",
#     "Ce2O3",
#     "Pr2O3",
#     "Sm2O3",
#     "Dy2O3",
#     "Al2O3",
#     "Gd2O3",
#     "Nd2O3",
# ]


m.scaling_factor = Suffix(direction=Suffix.EXPORT)

# Scale estimated reaction parameters (on the shared parameter block, not the
# flowsheet, so must be done explicitly before create_using)
# Scaling factor = 1 / typical_value so the solver works near O(1)
csb = CustomScalerBase()
for comp in m.fs.leach_rxns.reaction_idx:
    csb.set_variable_scaling_factor(m.fs.leach_rxns.A_ox[comp], 1)
    csb.set_variable_scaling_factor(m.fs.leach_rxns.k_prime[comp], 1e4)
    csb.set_variable_scaling_factor(m.fs.leach_rxns.K_film[comp], 1e2)
    csb.set_variable_scaling_factor(m.fs.leach_rxns.D_e[comp], 1e7)

# Scale each leach train using property-block scalers, mirroring
# CocurrentSlurryLeachingFlowsheet.scale_model() in leach_flowsheet.py.
# The scalers handle both variables and constraints; we only override the
# condition-specific factors (solid feed and acid concentration).
for s in m.OpCond:
    solid_scaler = m.fs.leach[s].mscontactor.solid.default_scaler()
    solid_scaler.default_scaling_factors["flow_mass"] = 1 / solid_feed_variation[s]

    liquid_scaler = m.fs.leach[s].mscontactor.liquid.default_scaler()
    liquid_scaler.default_scaling_factors["flow_vol"] = 1 / 224.3
    liquid_scaler.default_scaling_factors["conc_mass_comp[H]"] = 1 / (
        2 * acid_conc_variation[s] * 1e3
    )
    liquid_scaler.default_scaling_factors["conc_mass_comp[SO4]"] = 1 / (
        acid_conc_variation[s] * 96e3
    )

    submodel_scalers = ComponentMap()
    submodel_scalers[m.fs.leach[s].mscontactor.liquid_inlet_state] = liquid_scaler
    submodel_scalers[m.fs.leach[s].mscontactor.liquid] = liquid_scaler
    submodel_scalers[m.fs.leach[s].mscontactor.solid_inlet_state] = solid_scaler
    submodel_scalers[m.fs.leach[s].mscontactor.solid] = solid_scaler

    scaler_obj = m.fs.leach[s].default_scaler()
    scaler_obj.default_scaling_factors["liquid_phase_fraction"] = 1
    scaler_obj.default_scaling_factors["solid_phase_fraction"] = 1
    scaler_obj.scale_model(m.fs.leach[s], submodel_scalers=submodel_scalers)

m.recovery_sse = Objective(
    expr=sum(
        ((m.fs.leach[j].recovery[0, comp] - data.loc[comp, j]) / data.loc[comp, j]) ** 2
        for j in m.OpCond
        for comp in m.fs.coal.component_list - ["inerts"]
    ),
    sense=minimize,
)

scaling = TransformationFactory("core.scale_model")
scaled_model = scaling.create_using(m, rename=False)

# Solve scaled model
solver = SolverFactory("ipopt_v2")
solver.options["max_iter"] = 5000
solver.solve(scaled_model, tee=True)


scaling.propagate_solution(scaled_model, m)

# Print fitted parameters
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


# ── Simple reaction model (leach_reactions.py) ──────────────────────────────
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

for s in m2.OpCond:
    # Scale solid flow mass (inlet and outlet) per operating condition
    for blk in m2.fs.leach[s].mscontactor.solid.values():
        csb2.set_variable_scaling_factor(blk.flow_mass, 1 / solid_feed_variation[s])
    for blk in m2.fs.leach[s].mscontactor.solid_inlet_state.values():
        csb2.set_variable_scaling_factor(blk.flow_mass, 1 / solid_feed_variation[s])

    # Scale liquid inlet: H and SO4 are the large-magnitude driving species
    for blk in m2.fs.leach[s].mscontactor.liquid_inlet_state.values():
        csb2.set_variable_scaling_factor(blk.flow_vol, 1 / 224.3)
        csb2.set_variable_scaling_factor(
            blk.conc_mass_comp["H"], 1 / (2 * acid_conc_variation[s] * 1e3)
        )
        csb2.set_variable_scaling_factor(
            blk.conc_mass_comp["SO4"], 1 / (acid_conc_variation[s] * 96e3)
        )

    # Scale liquid outlet: use flowsheet-derived values for key leached species
    for blk in m2.fs.leach[s].mscontactor.liquid.values():
        csb2.set_variable_scaling_factor(blk.flow_vol, 1 / 224.3)
        csb2.set_variable_scaling_factor(blk.conc_mass_comp["SO4"], 1e-3)
        csb2.set_variable_scaling_factor(blk.conc_mass_comp["Ce"], 1 / 5)
        csb2.set_variable_scaling_factor(blk.conc_mass_comp["Nd"], 1 / 2)
        csb2.set_variable_scaling_factor(blk.conc_mass_comp["La"], 1)

scaling2 = TransformationFactory("core.scale_model")
scaled_model2 = scaling2.create_using(m2, rename=False)

solver2 = SolverFactory("ipopt_v2")
solver2.options["max_iter"] = 5000
solver2.solve(scaled_model2, tee=True)

scaling2.propagate_solution(scaled_model2, m2)


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
