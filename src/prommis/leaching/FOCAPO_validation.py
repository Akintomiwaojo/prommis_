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

from prommis.leaching.leach_train import (
    LeachingTrain,
    LeachingTrainInitializer,
    LeachingTrainScaler,
)

# from prommis.leaching.leach_reactions_combine import (
#     CoalRefuseLeachingCombinedReactionParameterBlock,
# )

from prommis.leaching.leach_reactions import CoalRefuseLeachingReactionParameterBlock
from prommis.leaching.leach_reactions_FOCAPO import (
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

from sklearn.metrics import r2_score, root_mean_squared_error

import re

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
    Then delegates to the ``LeachingTrainScaler`` for extent variables, material
    balance constraints, and all remaining constraint scaling.

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
    train_scaler = LeachingTrainScaler()

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

        # ---- Extent variables + all constraint scaling via LeachingTrainScaler ----
        # The train scaler delegates to the MSContactor scaler which computes
        # heterogeneous_reaction_extent SFs from stoichiometry and the (already
        # set) property variable SFs, then scales material balances, het-rxn
        # generation/constraint, and the extent constraint (rate × volume).
        train_scaler.variable_scaling_routine(model.fs.leach[s], overwrite=False)
        train_scaler.constraint_scaling_routine(model.fs.leach[s], overwrite=False)


data = pd.read_csv("parameter estimation.csv")
data = data.dropna(axis=1, how="all")
data = data.set_index(data.columns[0])

m = ConcreteModel()
m.OpCond = Set(
    initialize=[
        "0.05M S_L=1/10",
        "0.05M S_L=1.5/10",
    ],
)
m.fs = FlowsheetBlock(dynamic=False)
m.fs.leach_soln = SulfuricAcidLeachingParameters()
m.fs.coal = CoalRefuseParameters()
m.fs.leach_rxns = CoalRefuseLeachingReactionParameterBlock()

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
    "0.05M S_L=1/10": 22.67,
    "0.05M S_L=1.5/10": 34.02,
}

acid_conc_variation = {
    "0.05M S_L=1/10": 0.05,
    "0.05M S_L=1.5/10": 0.05,
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
csb2 = CustomScalerBase()

_scale_leach_train_blocks(m, csb2, m.OpCond, solid_feed_variation, acid_conc_variation)

scaling2 = TransformationFactory("core.scale_model")
scaled_model2 = scaling2.create_using(m, rename=False)

solver2 = SolverFactory("ipopt_v2")
solver2.options["max_iter"] = 5000
solver2.solve(scaled_model2, tee=True)

scaling2.propagate_solution(scaled_model2, m)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
exp_data = {}
for comp in m.fs.coal.component_list - ["inerts"]:
    exp_data[comp] = [data.loc[comp, j] for j in m.OpCond]

model_data = {}
for comp in m.fs.coal.component_list - ["inerts"]:
    model_data[comp] = [m.fs.leach[j].recovery[0, comp]() for j in m.OpCond]


plt.rcParams.update(
    {
        "font.family": "serif",  # Times-like font expected by most journals
        "font.sans-serif": ["Inter"],
        "font.size": 9,  # Base font size
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,  # 300 dpi minimum for print
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.pad": 4,
        "ytick.major.pad": 4,
        "axes.spines.top": False,  # remove top/right spines
        "axes.spines.right": False,
    }
)


def format_oxide(name):
    """Sc2O3 → $\mathrm{Sc_2O_3}$"""
    formatted = re.sub(r"(\d+)", r"_{\1}", name)  # wrap digits in _{...}
    return rf"$\mathrm{{{formatted}}}$"


labels = []
for j in m.OpCond:
    # extract just the S/L ratio part, e.g. "1/10", "1.5/10"
    match = re.search(r"(\d+\.?\d*/\d+)", str(j))
    labels.append(match.group(1) if match else str(j))

for comp in m.fs.coal.component_list - ["inerts"]:
    y_exp = np.array(exp_data[comp])
    y_mod = np.array(model_data[comp])

    r2 = r2_score(y_exp, y_mod)
    RMSE_est = root_mean_squared_error(y_exp, y_mod)

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    x = np.arange(len(y_exp))
    width = 0.35

    ax.bar(
        x - width / 2,
        y_exp,
        width,
        label="Experimental",
        color="#0072B2",
        edgecolor="black",
        linewidth=0.8,
    )
    ax.bar(
        x + width / 2,
        y_mod,
        width,
        label="Model",
        color="#D55E00",
        edgecolor="black",
        linewidth=0.8,
    )

    # labels = [j.replace("_", " : ").replace("S : L", "S/L") for j in m.OpCond]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Recovery (%)")
    ax.set_ylim(0, max(max(y_exp), max(y_mod)) * 1.15)
    ax.set_title(
        f"Recovery Comparison for {format_oxide(comp)}",
        fontsize=9,
        pad=6,
        linespacing=1.6,
    )
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    # ax.annotate(
    #     f"RMSE = {RMSE_est:.2f}%",
    #     xy=(1.0, 1.02), xycoords="axes fraction",
    #     ha="right", va="bottom", fontsize=8,
    #     color="#555555",
    # )
    fig.tight_layout()
    # fig.savefig(f"recovery_{comp}.png", transparent=False)
    # fig.savefig(f"recovery_{comp}.svg")          # vector format for submission
    plt.show()
