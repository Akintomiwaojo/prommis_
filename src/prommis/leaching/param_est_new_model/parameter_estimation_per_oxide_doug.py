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
import numpy as np
import pandas as pd

from idaes.core import FlowsheetBlock
from idaes.core.scaling import CustomScalerBase

from prommis.leaching.leach_train import LeachingTrain
from prommis.leaching.leach_reactions import CoalRefuseLeachingReactionParameterBlock
from prommis.leaching.param_est_new_model.leach_reactions_doug import (
    CoalRefuseLeachingCombinedReactionParameterBlock,
)
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
# Liquid outlet scaling factors [mg/L] at ~20% recovery, S/L = 1/10
# ---------------------------------------------------------------------------
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
    liq_scaler = SulfuricAcidLeachingPropertiesScaler()
    solid_scaler = CoalRefusePropertiesScaler()

    for s in op_cond:
        csb.set_variable_scaling_factor(model.fs.leach[s].volume[0, 1], 1 / 378.5)

        for blk in model.fs.leach[s].mscontactor.solid_inlet_state.values():
            csb.set_variable_scaling_factor(blk.flow_mass, 1 / solid_feed[s])
            solid_scaler.variable_scaling_routine(blk, overwrite=False)

        for blk in model.fs.leach[s].mscontactor.solid.values():
            csb.set_variable_scaling_factor(blk.flow_mass, 1 / solid_feed[s])
            for comp in model.fs.coal.component_list:
                x0 = model.fs.coal.mass_frac_comp_initial[comp].value
                csb.set_variable_scaling_factor(blk.mass_frac_comp[comp], 1 / x0)
            if hasattr(blk, "conversion_comp"):
                _conv_sf = {
                    "Al2O3": 1 / 0.15,
                    "Fe2O3": 1 / 0.20,
                    "CaO": 1 / 0.20,
                    "Sc2O3": 1 / 0.15,
                    "Y2O3": 1 / 0.15,
                    "La2O3": 1 / 0.15,
                    "Ce2O3": 1 / 0.18,
                    "Pr2O3": 1 / 0.15,
                    "Nd2O3": 1 / 0.18,
                    "Sm2O3": 1 / 0.15,
                    "Gd2O3": 1 / 0.18,
                    "Dy2O3": 1 / 0.12,
                    "inerts": 1,
                }

                for comp in model.fs.coal.component_list:
                    sf = _conv_sf.get(comp, 1)
                    csb.set_variable_scaling_factor(blk.conversion_comp[comp], sf)
            solid_scaler.constraint_scaling_routine(blk, overwrite=False)

        for blk in model.fs.leach[s].mscontactor.liquid_inlet_state.values():
            csb.set_variable_scaling_factor(blk.flow_vol, 1 / 224.3)
            csb.set_variable_scaling_factor(
                blk.conc_mass_comp["H"], 1 / (2 * acid_conc[s] * 1e3)
            )
            csb.set_variable_scaling_factor(
                blk.conc_mass_comp["SO4"], 1 / (acid_conc[s] * 96e3)
            )
            liq_scaler.variable_scaling_routine(blk, overwrite=False)

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
# Read experimental data
# ---------------------------------------------------------------------------
data = pd.read_csv("parameter estimation.csv")
data = data.dropna(axis=1, how="all")
data = data.set_index(data.columns[0])

op_cond_list = [
    "0.025M S_L=1/10",
    "0.05M S_L=1/10",
    "0.075M S_L=1/10",
    "0.05M S_L=1.5/10",
    "0.05M S_L=2/10",
    "0.075M S_L=1.5/10",
]

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

# ---------------------------------------------------------------------------
# Build Andrew's model (m2) for comparison
# ---------------------------------------------------------------------------
m2 = ConcreteModel()
m2.OpCond = Set(initialize=op_cond_list)
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
solver2.solve(scaled_model2, tee=False)
scaling2.propagate_solution(scaled_model2, m2)


# ---------------------------------------------------------------------------
# Parity plots
# ---------------------------------------------------------------------------
# for ox in oxides:
#     y_exp = np.array([data.loc[ox, j] for j in m.OpCond])
#     y_mod = np.array([m.fs.leach[j].recovery[0, ox]() for j in m.OpCond])
#     y_sim = np.array([m2.fs.leach[j].recovery[0, ox]() for j in m2.OpCond])

#     r2 = r2_score(y_exp, y_mod)
#     r2_sim = r2_score(y_exp, y_sim)

#     plt.figure(figsize=(5, 5), dpi=100)
#     plt.scatter(y_exp, y_mod, label="Current Model", marker="o", color="#1f77b4")
#     plt.scatter(y_exp, y_sim, label="Andrew's Model", marker="s", color="#ff7f0e")
#     plt.plot(y_exp, y_exp, label="Parity line (y = x)", color="k")
#     plt.xlabel("Experimental Recovery (%)")
#     plt.ylabel("Model Recovery (%)")
#     plt.title(f"Parity Plot: Experimental vs Model Extraction for {ox}")
#     plt.legend()
#     plt.text(
#         0.05,
#         0.95,
#         f"Current model: $R^2 = {r2:.4f}$",
#         transform=plt.gca().transAxes,
#         fontsize=12,
#         verticalalignment="top",
#     )
#     plt.text(
#         0.05,
#         0.88,
#         f"Andrew's model: $R^2 = {r2_sim:.4f}$",
#         transform=plt.gca().transAxes,
#         fontsize=12,
#         verticalalignment="top",
#     )
#     plt.tight_layout()
#     plt.show()


# ---------------------------------------------------------------------------
# Build model (current — per-oxide fitting)
# ---------------------------------------------------------------------------
m = ConcreteModel()
m.OpCond = Set(initialize=op_cond_list)
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

# Fix inlet conditions
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


# m.fs.leach_rxns.A_ox["Al2O3"].setlb(0.90)
# m.fs.leach_rxns.A_ox["Al2O3"].setub(2.0)


# Scaling
m.scaling_factor = Suffix(direction=Suffix.EXPORT)
csb = CustomScalerBase()

_k_prime_sf = {
    "Al2O3": 1 / 5e-8,
    "Fe2O3": 1 / 1.0,
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

_scale_leach_train_blocks(m, csb, m.OpCond, solid_feed_variation, acid_conc_variation)

scaling = TransformationFactory("core.scale_model")
solver = SolverFactory("ipopt_v2")
solver.options["max_iter"] = 5000

# ---------------------------------------------------------------------------
# Per-oxide parameter estimation loop
# Each oxide is fitted independently: all other oxide parameters are fixed
# at their initial values while that oxide's k_prime and A_ox are optimised.
# ---------------------------------------------------------------------------
# oxides = list(m.fs.coal.component_list - ["inerts"])

oxides = [
    "Al2O3",
    "Sc2O3",
    "Y2O3",
    "La2O3",
    "Ce2O3",
    "Pr2O3",
    "Nd2O3",
    "Sm2O3",
    "Gd2O3",
    "Dy2O3",
    "CaO",
    "Fe2O3",
]

# Fix ALL oxide parameters before the loop
for ox in oxides:
    m.fs.leach_rxns.k_prime[ox].fix()
    m.fs.leach_rxns.A_ox[ox].fix()

fitted_k_prime = {}
fitted_A_ox = {}

for ox in oxides:
    print(f"\n--- Fitting {ox} ---")

    # Free only this oxide's parameters
    m.fs.leach_rxns.k_prime[ox].unfix()
    m.fs.leach_rxns.A_ox[ox].unfix()

    # Oxide-specific relative SSE objective
    if hasattr(m, "oxide_sse"):
        m.del_component(m.oxide_sse)
    m.oxide_sse = Objective(
        expr=sum(
            ((m.fs.leach[j].recovery[0, ox] - data.loc[ox, j])) ** 2 for j in m.OpCond
        ),
        sense=minimize,
    )

    scaled_m = scaling.create_using(m, rename=False)
    solver.solve(scaled_m, tee=False)
    scaling.propagate_solution(scaled_m, m)

    fitted_k_prime[ox] = m.fs.leach_rxns.k_prime[ox].value
    fitted_A_ox[ox] = m.fs.leach_rxns.A_ox[ox].value

    print(f"  k_prime = {fitted_k_prime[ox]:.4e}   A_ox = {fitted_A_ox[ox]:.6f}")

    # Lock in the fitted value for this oxide before moving to the next
    m.fs.leach_rxns.k_prime[ox].fix()
    m.fs.leach_rxns.A_ox[ox].fix()

    y_exp = np.array([data.loc[ox, j] for j in m.OpCond])
    y_mod = np.array([m.fs.leach[j].recovery[0, ox]() for j in m.OpCond])
    y_sim = np.array([m2.fs.leach[j].recovery[0, ox]() for j in m.OpCond])

    r2 = r2_score(y_exp, y_mod)
    r2_sim = r2_score(y_exp, y_sim)

    plt.figure(figsize=(5, 5), dpi=100)
    plt.scatter(y_exp, y_mod, label="Current Model", marker="o", color="#1f77b4")
    plt.scatter(y_exp, y_sim, label="Andrew's Model", marker="s", color="#ff7f0e")
    plt.plot(y_exp, y_exp, label="Parity line (y = x)", color="k")
    plt.xlabel("Experimental Recovery (%)")
    plt.ylabel("Model Recovery (%)")
    plt.title(f"Parity Plot: Experimental vs Model Extraction for {ox}")
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
print("\n=== Fitted Parameters (per-oxide) ===")
print(f"{'Oxide':<10} {'A_ox':>12} {'k_prime':>14}")
print("-" * 40)
for ox in oxides:
    print(f"{ox:<10} {fitted_A_ox[ox]:>12.6f} {fitted_k_prime[ox]:>14.6e}")
