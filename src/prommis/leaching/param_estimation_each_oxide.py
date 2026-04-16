from pyomo.environ import (
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
from idaes.core.util.scaling import set_scaling_factor

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

m.ree_oxide = Set(
    initialize=[
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

solver = SolverFactory("ipopt_v2")
solver.options["max_iter"] = 5000

m.scaling_factor = Suffix(direction=Suffix.EXPORT)

for s in m.OpCond:
    for j in m.fs.coal.component_list:
        if j not in ["Al2O3", "Fe2O3", "CaO", "inerts"]:
            set_scaling_factor(
                m.fs.leach[s].mscontactor.solid[0.0, 1].mass_frac_comp[j], 1e5
            )
            set_scaling_factor(
                m.fs.leach[s].mscontactor.solid_inlet_state[0.0].mass_frac_comp[j], 1e5
            )
            set_scaling_factor(
                m.fs.leach[s]
                .mscontactor.heterogeneous_reactions[0.0, 1]
                .reaction_rate[j],
                1e5,
            )
            set_scaling_factor(
                m.fs.leach[s].mscontactor.solid[0.0, 1].conversion_comp[j], 1e3
            )
            set_scaling_factor(
                m.fs.leach[s].mscontactor.solid_inlet_state[0.0].conversion_comp[j],
                1e3,
            )

scaling = TransformationFactory("core.scale_model")
scaled_model = scaling.create_using(m, rename=False)

# Fix all four fitted parameters for every oxide; unfix one oxide at a time
scaled_model.fs.leach_rxns.A_ox.fix()
scaled_model.fs.leach_rxns.k_prime.fix()
scaled_model.fs.leach_rxns.K_film.fix()
scaled_model.fs.leach_rxns.D_e.fix()

fitted_params = {}

for e in scaled_model.ree_oxide:
    # Unfix the four parameters for this oxide only
    scaled_model.fs.leach_rxns.A_ox[e].unfix()
    scaled_model.fs.leach_rxns.k_prime[e].unfix()
    scaled_model.fs.leach_rxns.K_film[e].unfix()
    scaled_model.fs.leach_rxns.D_e[e].unfix()

    # @scaled_model.Expression()
    # def sse(scaled_model,e):
    #     return  sum(
    #         (scaled_model.fs.leach[j].recovery[0, e] - data.loc[e, j]) ** 2
    #         for j in scaled_model.OpCond)

    # SSE objective over all operating conditions for this oxide
    # scaled_model.recovery_sse = Objective(
    #     expr=sum(
    #         (scaled_model.fs.leach[j].recovery[0, e] - data.loc[e, j]) ** 2
    #         for j in scaled_model.OpCond),
    #     sense=minimize,
    # )

    scaled_model.recovery_sse = Objective(
        expr=sum(
            (
                (m.fs.leach[j].recovery[0, comp] - data.loc[comp, j])
                / max(data.loc[comp, j], 1e-3)
            )
            ** 2
            for j in m.OpCond
            for comp in m.fs.coal.component_list - ["inerts"]
        ),
        sense=minimize,
    )

    solver.solve(scaled_model, tee=True)

    scaling.propagate_solution(scaled_model, m)

    # Store fitted values from the unscaled model
    fitted_params[e] = {
        "A_ox": m.fs.leach_rxns.A_ox[e].value,
        "k_prime": m.fs.leach_rxns.k_prime[e].value,
        "K_film": m.fs.leach_rxns.K_film[e].value,
        "D_e": m.fs.leach_rxns.D_e[e].value,
    }

    # Remove objective before next iteration
    scaled_model.del_component(scaled_model.recovery_sse)

    # Refix parameters for this oxide
    scaled_model.fs.leach_rxns.A_ox[e].fix()
    scaled_model.fs.leach_rxns.k_prime[e].fix()
    scaled_model.fs.leach_rxns.K_film[e].fix()
    scaled_model.fs.leach_rxns.D_e[e].fix()

    # Parity plot: experimental vs model recovery
    exp_vals = [data.loc[e, j] for j in m.OpCond]
    model_vals = [m.fs.leach[j].recovery[0, e]() for j in m.OpCond]

    plt.figure(figsize=(6, 6))
    plt.scatter(
        exp_vals, model_vals, marker="o", color="b", label="Operating conditions"
    )
    min_val = min(exp_vals + model_vals)
    max_val = max(exp_vals + model_vals)
    plt.plot([min_val, max_val], [min_val, max_val], color="r", label="Perfect fit")
    plt.xlabel("Experimental Recovery (%)")
    plt.ylabel("Model Recovery (%)")
    plt.title(f"Parity Plot: {e}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Print summary of fitted parameters
print("\n=== Fitted Parameters ===")
print(f"{'Oxide':<10} {'A_ox':>12} {'k_prime':>14} {'K_film':>14} {'D_e':>14}")
print("-" * 66)
for e, p in fitted_params.items():
    print(
        f"{e:<10} {p['A_ox']:>12.6f} {p['k_prime']:>14.6e} "
        f"{p['K_film']:>14.6e} {p['D_e']:>14.6e}"
    )
