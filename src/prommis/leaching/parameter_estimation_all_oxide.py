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

import pandas as pd

from idaes.core import FlowsheetBlock
from idaes.core.scaling import CustomScalerBase
from idaes.core.util import to_json

from prommis.leaching.leach_train import LeachingTrain, LeachingTrainInitializer
from prommis.leaching.leach_reactions_combine import (
    CoalRefuseLeachingCombinedReactionParameterBlock,
)
from prommis.properties.coal_refuse_properties import CoalRefuseParameters
from prommis.properties.sulfuric_acid_leaching_properties import (
    SulfuricAcidLeachingParameters,
)
from idaes.core.util.model_statistics import degrees_of_freedom

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

param_ele_list = [
    "Fe2O3",
    "CaO",
    "Sc2O3",
    "Y2O3",
    "La2O3",
    "Ce2O3",
    "Pr2O3",
    "Sm2O3",
    "Dy2O3",
    "Al2O3",
    "Gd2O3",
    "Nd2O3",
]


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


# exp_data = []
# for j in m.OpCond:
#     for comp in m.fs.coal.component_list - ["inerts"]:
#         s = data.loc[comp,j]
#         exp_data.append(s)

# model_data = []
# for j in m.OpCond:
#     for comp in m.fs.coal.component_list - ["inerts"]:
#         model_data.append(m.fs.leach[j].recovery[0,comp]())

# plt.figure(figsize=(10,6))
# plt.scatter(exp_data,model_data,label="Experimental Extraction", marker="o",linestyle="-",color="b")
# plt.plot([min(exp_data), max(exp_data)], [min(exp_data), max(exp_data)], "k--",label="Model Extraction %", marker = "x", linestyle = "--", color = "r")
# plt.xlabel("Experimental Recovery (%)")
# plt.ylabel("Model Recovery (%)")
# plt.title("Parity Plot: Experimental vs Model Extraction")
# plt.legend()
# plt.tight_layout()
# plt.show()

exp_data = {}
for comp in m.fs.coal.component_list - ["inerts"]:
    exp_data[comp] = [data.loc[comp, j] for j in m.OpCond]


model_data = {}
for comp in m.fs.coal.component_list - ["inerts"]:
    model_data[comp] = [m.fs.leach[j].recovery[0, comp]() for j in m.OpCond]


for comp in m.fs.coal.component_list - ["inerts"]:
    plt.figure(figsize=(10, 6))
    plt.scatter(
        exp_data[comp],
        model_data[comp],
        label="Experimental Extraction",
        marker="o",
        color="b",
    )

    plt.plot(
        exp_data[comp],
        exp_data[comp],
        # "k--",
        label="Model Extraction %",
        # marker="x",
        # linestyle="--",
        color="r",
    )
    plt.xlabel("Experimental Recovery (%)")
    plt.ylabel("Model Recovery (%)")
    plt.title("Parity Plot: Experimental vs Model Extraction for " + comp)
    plt.legend()
    plt.tight_layout()
    plt.show()
