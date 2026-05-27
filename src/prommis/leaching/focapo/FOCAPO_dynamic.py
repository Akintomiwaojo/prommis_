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
    Block,
    Var,
    Param,
)
from pyomo.common.config import Bool, ConfigDict, ConfigValue, In, PositiveInt
from pyomo.dae.flatten import flatten_dae_components
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import idaes.logger as idaeslog
from idaes.core import FlowsheetBlock, declare_process_block_class, FlowsheetBlockData
from idaes.core.initialization import ModularInitializerBase
from idaes.core.scaling import CustomScalerBase
from idaes.core.util import to_json, from_json
from idaes.core.solvers import get_solver

from prommis.leaching.leach_train import (
    LeachingTrain,
    LeachingTrainInitializer,
    LeachingTrainScaler,
)

# from prommis.leaching.leach_reactions_combine import (
#     CoalRefuseLeachingCombinedReactionParameterBlock,
# )

from prommis.leaching.leach_reactions import CoalRefuseLeachingReactionParameterBlock
from prommis.leaching.focapo.leach_reaction_FOCAPO_updt import (
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
from prommis.util import scale_time_discretization_equations

from sklearn.metrics import r2_score, root_mean_squared_error

import re


def build_model(time_duration, perturb_time, number_of_tanks):
    """
    Method to build a single stage leaching system using data for
    West Kentucky No. 13 coal refuse.
    Args:
        time_duration: Duration of the simulation in hours.
        number_of_tanks: Number of tanks in the leaching train.
    Returns:
        m: ConcreteModel object with the leaching system.
    """
    m = ConcreteModel()

    m.fs = FlowsheetBlock(
        dynamic=True, time_set=[0, perturb_time, time_duration], time_units=units.hour
    )

    m.fs.leach_soln = SulfuricAcidLeachingParameters()
    m.fs.coal = CoalRefuseParameters()
    m.fs.leach_rxns = CoalRefuseLeachingCombinedReactionParameterBlock()

    m.fs.leach = LeachingTrain(
        number_of_tanks=number_of_tanks,
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

    return m


def discretization(m):
    """
    Discretization of the time domain
    """

    m.discretizer = TransformationFactory("dae.collocation")
    m.discretizer.apply_to(m, nfe=18, ncp=6, wrt=m.fs.time, scheme="LAGRANGE-RADAU")


def copy_first_steady_state(m):
    """
    Function that propagates initial steady state guess to future time points.
    This function is used to initialize all the time discrete variables to the
    initial steady state value.
    """

    regular_vars, time_vars = flatten_dae_components(m, m.fs.time, Var, active=True)
    # Copy initial conditions forward
    for var in time_vars:
        for t in m.fs.time:
            if t == m.fs.time.first():
                continue
            else:
                var[t].value = var[m.fs.time.first()].value


def set_inputs(m, perturb_time):
    """
    Set inlet conditions to leach reactor based on one case study from
    University of Kentucky pilot plant study. The values of the time discrete
    variables at initial time are fixed to the steady state values.
    Args:
        m: ConcreteModel object with the leaching system.
        perturb_time: Time at which the perturbation is applied.
    Returns:
        None
    """

    m.liquid_solid_residence_time_ratio = Param(
        initialize=1 / 32, units=units.dimensionless
    )

    # Liquid feed state
    m.fs.leach.liquid_inlet.flow_vol.fix(224.3 * units.L / units.hour)
    m.fs.leach.liquid_inlet.conc_mass_comp.fix(1e-10 * units.mg / units.L)
    m.fs.leach.liquid_inlet.conc_mass_comp[:, "H2O"].fix(1e6 * units.mg / units.L)
    # m.fs.leach.liquid_inlet.conc_mass_comp[:, "H"].fix(
    #     2 * 0.05 * 1e3 * units.mg / units.L
    # )
    acid_conc_before = 0.05  # mol/L H2SO4 before perturbation
    acid_conc_after = 0.075  # mol/L H2SO4 after perturbation

    for t in m.fs.time:
        if t <= perturb_time:
            c_acid = acid_conc_before
        else:
            c_acid = acid_conc_after
        m.fs.leach.liquid_inlet.conc_mass_comp[t, "H"].fix(
            2 * c_acid * 1e3 * units.mg / units.L
        )
        m.fs.leach.liquid_inlet.conc_mass_comp[t, "SO4"].fix(
            c_acid * 96e3 * units.mg / units.L
        )
    m.fs.leach.liquid_inlet.conc_mass_comp[:, "HSO4"].fix(1e-8 * units.mg / units.L)
    # m.fs.leach.liquid_inlet.conc_mass_comp[:, "SO4"].fix(
    #     0.05 * 96e3 * units.mg / units.L
    # )

    # Solid feed state
    m.fs.leach.solid_inlet.flow_mass.fix(22.68 * units.kg / units.hour)
    # for t in m.fs.time:
    #     if t <= perturb_time:
    #         m.fs.leach.solid_inlet.flow_mass[t].fix(22.68 * units.kg / units.hour)
    #     else:
    #         m.fs.leach.solid_inlet.flow_mass[t].fix(2 * 22.68 * units.kg / units.hour)
    m.fs.leach.solid_inlet.mass_frac_comp[:, "inerts"].fix(0.6952 * units.kg / units.kg)
    m.fs.leach.solid_inlet.mass_frac_comp[:, "Al2O3"].fix(0.237 * units.kg / units.kg)
    m.fs.leach.solid_inlet.mass_frac_comp[:, "Fe2O3"].fix(0.0642 * units.kg / units.kg)
    m.fs.leach.solid_inlet.mass_frac_comp[:, "CaO"].fix(3.31e-3 * units.kg / units.kg)
    m.fs.leach.solid_inlet.mass_frac_comp[:, "Sc2O3"].fix(
        2.77966e-05 * units.kg / units.kg
    )
    m.fs.leach.solid_inlet.mass_frac_comp[:, "Y2O3"].fix(
        3.28653e-05 * units.kg / units.kg
    )
    m.fs.leach.solid_inlet.mass_frac_comp[:, "La2O3"].fix(
        6.77769e-05 * units.kg / units.kg
    )
    m.fs.leach.solid_inlet.mass_frac_comp[:, "Ce2O3"].fix(
        0.000156161 * units.kg / units.kg
    )
    m.fs.leach.solid_inlet.mass_frac_comp[:, "Pr2O3"].fix(
        1.71438e-05 * units.kg / units.kg
    )
    m.fs.leach.solid_inlet.mass_frac_comp[:, "Nd2O3"].fix(
        6.76618e-05 * units.kg / units.kg
    )
    m.fs.leach.solid_inlet.mass_frac_comp[:, "Sm2O3"].fix(
        1.47926e-05 * units.kg / units.kg
    )
    m.fs.leach.solid_inlet.mass_frac_comp[:, "Gd2O3"].fix(
        1.0405e-05 * units.kg / units.kg
    )
    m.fs.leach.solid_inlet.mass_frac_comp[:, "Dy2O3"].fix(
        7.54827e-06 * units.kg / units.kg
    )

    # Fixing the volume of the leach reactor
    m.fs.leach.volume.fix(100 * units.gallon)
    m.fs.leach.mscontactor.volume.fix(100 * units.gallon)

    @m.Constraint(m.fs.time, m.fs.leach.mscontactor.elements)
    def volume_fraction_rule(m, t, s):

        theta_s = m.fs.leach.mscontactor.volume_frac_stream[t, s, "solid"]
        theta_l = m.fs.leach.mscontactor.volume_frac_stream[t, s, "liquid"]
        v_l = m.fs.leach.mscontactor.liquid[t, s].flow_vol
        solid_dens_mass = m.fs.leach.config.solid_phase["property_package"].dens_mass
        v_s = m.fs.leach.mscontactor.solid[t, s].flow_mass / solid_dens_mass
        return theta_l * v_s == theta_s * v_l * m.liquid_solid_residence_time_ratio

    # Fixing the variable values at t=0
    m.fs.leach.mscontactor.liquid[0, :].flow_vol.fix()
    m.fs.leach.mscontactor.liquid[0, :].conc_mass_comp["H"].fix()
    m.fs.leach.mscontactor.liquid[0, :].conc_mass_comp["HSO4"].fix()
    # m.fs.leach.mscontactor.liquid[0, :].conc_mass_comp["H2C2O4"].fix()
    m.fs.leach.mscontactor.liquid[0, :].conc_mass_comp["Cl"].fix()
    m.fs.leach.mscontactor.liquid[0, :].conc_mass_comp["Sc"].fix()
    m.fs.leach.mscontactor.liquid[0, :].conc_mass_comp["Y"].fix()
    m.fs.leach.mscontactor.liquid[0, :].conc_mass_comp["La"].fix()
    m.fs.leach.mscontactor.liquid[0, :].conc_mass_comp["Ce"].fix()
    m.fs.leach.mscontactor.liquid[0, :].conc_mass_comp["Pr"].fix()
    m.fs.leach.mscontactor.liquid[0, :].conc_mass_comp["Nd"].fix()
    m.fs.leach.mscontactor.liquid[0, :].conc_mass_comp["Sm"].fix()
    m.fs.leach.mscontactor.liquid[0, :].conc_mass_comp["Gd"].fix()
    m.fs.leach.mscontactor.liquid[0, :].conc_mass_comp["Dy"].fix()
    m.fs.leach.mscontactor.liquid[0, :].conc_mass_comp["Al"].fix()
    m.fs.leach.mscontactor.liquid[0, :].conc_mass_comp["Ca"].fix()
    m.fs.leach.mscontactor.liquid[0, :].conc_mass_comp["Fe"].fix()

    m.fs.leach.mscontactor.solid[0, :].mass_frac_comp["inerts"].fix()
    m.fs.leach.mscontactor.solid[0, :].mass_frac_comp["Y2O3"].fix()
    m.fs.leach.mscontactor.solid[0, :].mass_frac_comp["La2O3"].fix()
    m.fs.leach.mscontactor.solid[0, :].mass_frac_comp["Ce2O3"].fix()
    m.fs.leach.mscontactor.solid[0, :].mass_frac_comp["Pr2O3"].fix()
    m.fs.leach.mscontactor.solid[0, :].mass_frac_comp["Nd2O3"].fix()
    m.fs.leach.mscontactor.solid[0, :].mass_frac_comp["Sm2O3"].fix()
    m.fs.leach.mscontactor.solid[0, :].mass_frac_comp["Gd2O3"].fix()
    m.fs.leach.mscontactor.solid[0, :].mass_frac_comp["Dy2O3"].fix()
    m.fs.leach.mscontactor.solid[0, :].mass_frac_comp["Al2O3"].fix()
    m.fs.leach.mscontactor.solid[0, :].mass_frac_comp["CaO"].fix()
    m.fs.leach.mscontactor.solid[0, :].mass_frac_comp["Fe2O3"].fix()
    m.fs.leach.mscontactor.solid[0, :].flow_mass.fix()

    m.fs.leach.mscontactor.liquid_inherent_reaction_extent[0.0, :, "Ka2"].fix()


# Expected outlet liquid concentrations [mg/L] at ~20% recovery, S/L = 1/10.
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

# Per-component conversion scaling factors for the solid phase.
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


def _scale_leach_train_blocks(fs, csb):
    """Apply physical-state scaling to all leach train sub-blocks at every
    time point. Adapted from ``_scale_leach_train_blocks`` in
    ``FOCAPO_steady.py`` to iterate over the dynamic time index.
    """
    liq_scaler = SulfuricAcidLeachingPropertiesScaler()
    solid_scaler = CoalRefusePropertiesScaler()
    train_scaler = LeachingTrainScaler()

    # Volume (100 gal ~ 378.5 L)
    for vardata in fs.leach.volume.values():
        csb.set_variable_scaling_factor(vardata, 1 / 378.5)

    # Solid inlet at every time point
    for blk in fs.leach.mscontactor.solid_inlet_state.values():
        csb.set_variable_scaling_factor(blk.flow_mass, 1 / 22.67)
        solid_scaler.variable_scaling_routine(blk, overwrite=False)

    # Solid holdup blocks at every time point
    for blk in fs.leach.mscontactor.solid.values():
        csb.set_variable_scaling_factor(blk.flow_mass, 1 / 22.67)
        for comp in fs.coal.component_list:
            x0 = fs.coal.mass_frac_comp_initial[comp].value
            csb.set_variable_scaling_factor(blk.mass_frac_comp[comp], 1 / x0)

        if hasattr(blk, "conversion_comp"):
            for comp in fs.coal.component_list:
                sf = _conv_sf.get(comp, 1)
                csb.set_variable_scaling_factor(blk.conversion_comp[comp], sf)
        solid_scaler.constraint_scaling_routine(blk, overwrite=False)

    # Liquid inlet at every time point
    for blk in fs.leach.mscontactor.liquid_inlet_state.values():
        csb.set_variable_scaling_factor(blk.flow_vol, 1 / 224.3)
        csb.set_variable_scaling_factor(blk.conc_mass_comp["H"], 1 / (2 * 0.05 * 1e3))
        csb.set_variable_scaling_factor(blk.conc_mass_comp["SO4"], 1 / (0.05 * 96e3))
        liq_scaler.variable_scaling_routine(blk, overwrite=False)

    # Liquid holdup blocks at every time point
    for blk in fs.leach.mscontactor.liquid.values():
        csb.set_variable_scaling_factor(blk.flow_vol, 1 / 224.3)
        csb.set_variable_scaling_factor(blk.conc_mass_comp["H"], 1 / (2 * 0.05 * 1e3))
        csb.set_variable_scaling_factor(blk.conc_mass_comp["SO4"], 1 / (0.05 * 96e3))
        for comp, sf in _liq_outlet_sf.items():
            csb.set_variable_scaling_factor(blk.conc_mass_comp[comp], sf)
        liq_scaler.variable_scaling_routine(blk, overwrite=False)
        liq_scaler.constraint_scaling_routine(blk, overwrite=False)

    train_scaler.variable_scaling_routine(fs.leach, overwrite=False)
    train_scaler.constraint_scaling_routine(fs.leach, overwrite=False)


def scale_dynamics(fs, sf_t):
    """Apply physical-state scaling at every time point and scale the time
    discretization equations. Mirrors the combined ``scale_model`` +
    ``scale_dynamics`` pattern from ``leach_flowsheet.py``, adapted for the
    FOCAPO dynamic flowsheet (no DAE index reduction).
    """
    parent = fs.parent_block()
    if not hasattr(parent, "scaling_factor"):
        parent.scaling_factor = Suffix(direction=Suffix.EXPORT)

    csb = CustomScalerBase()
    _scale_leach_train_blocks(fs, csb)
    scale_time_discretization_equations(fs, fs.time, sf_t)


if __name__ == "__main__":

    time_duration = 36
    perturb_time = 12
    number_of_tanks = 1

    # Call the build_model function to create the model
    m = build_model(time_duration, perturb_time, number_of_tanks)

    # Discretize the model
    discretization(m)

    # Import steady state values from JSON file
    from_json(m, fname="FOCAPO_leaching.json")

    # Initialize the model at steady state values
    copy_first_steady_state(m)

    # Set the inputs for the model
    set_inputs(m, perturb_time)

    # set_scaling(m)
    scale_dynamics(m.fs, 1)

    scaling = TransformationFactory("core.scale_model")
    scaled_model = scaling.create_using(m, rename=False)

    # Solve the model
    solver = get_solver("ipopt_v2")
    solver.options["max_iter"] = 5000
    solver.solve(scaled_model, tee=True)

    scaling.propagate_solution(scaled_model, m)

    # Solid stream outlet values at final time
    m.fs.leach.mscontactor.solid[time_duration, number_of_tanks].flow_mass.pprint()
    m.fs.leach.mscontactor.solid[time_duration, number_of_tanks].mass_frac_comp.pprint()

    # Liquid stream outlet values at final time
    m.fs.leach.mscontactor.liquid[time_duration, number_of_tanks].flow_vol.pprint()
    m.fs.leach.mscontactor.liquid[
        time_duration, number_of_tanks
    ].conc_mass_comp.pprint()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- Font handling: only use Inter if it's actually installed ---
available_fonts = {f.name for f in fm.fontManager.ttflist}
preferred_font = "Inter" if "Inter" in available_fonts else "DejaVu Sans"

rc_update = {
    "font.family": "sans-serif",
    "font.sans-serif": [preferred_font, "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "hatch.linewidth": 0.6,
}
if preferred_font == "Inter":
    rc_update.update(
        {
            "mathtext.fontset": "custom",
            "mathtext.rm": "Inter",
            "mathtext.it": "Inter:italic",
            "mathtext.bf": "Inter:bold",
        }
    )
plt.rcParams.update(rc_update)

# --- Compute recovery -----------------------------------------------------
REE_list = [
    e
    for e in m.fs.leach_soln.component_list
    if e not in ["H2O", "H", "HSO4", "SO4", "Cl", "H2C2O4"]
]

recovery = {}
for k in REE_list:
    oxide = f"{k}O" if k == "Ca" else f"{k}2O3"
    recovery[k] = [
        (
            (
                m.fs.leach.liquid_outlet.flow_vol[t]()
                * m.fs.leach.liquid_outlet.conc_mass_comp[t, k]()
                - m.fs.leach.liquid_inlet.flow_vol[t]()
                * m.fs.leach.liquid_inlet.conc_mass_comp[t, k]()
            )
            / (
                m.fs.leach.solid_inlet.flow_mass[t]()
                * m.fs.leach.solid_inlet.mass_frac_comp[t, oxide]()
                * 1e6
            )
        )
        * 100
        for t in m.fs.time
    ]


# --- Plot configuration ---------------------------------------------------
def cation_label(el: str) -> str:
    return rf"{el}$^{{3+}}$"


def open_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(top=False, right=False)


# Six distinct Okabe-Ito (Wong) colors, paired with three line styles per panel
series_styles = {
    "Ce": {"color": "#CC79A7", "linestyle": "-"},  # rose,         solid
    "Pr": {"color": "#E69F00", "linestyle": "--"},  # orange,       dashed
    "Nd": {"color": "#56B4E9", "linestyle": "-."},  # sky blue,     dash-dot
    "Sm": {"color": "#009E73", "linestyle": "-"},  # bluish green, solid
    "Y": {"color": "#0072B2", "linestyle": "--"},  # blue,         dashed
    "Dy": {"color": "#D55E00", "linestyle": "-."},  # vermillion,   dash-dot
}

light_rees = ["Ce", "Pr", "Nd"]
heavy_rees = ["Sm", "Y", "Dy"]

t_arr = list(m.fs.time)


def plot_group(group):
    fig, ax = plt.subplots(figsize=(3.5, 2.8), constrained_layout=True)

    for el in group:
        ax.plot(
            t_arr,
            recovery[el],
            label=cation_label(el),
            linewidth=1.8,
            **series_styles[el],
        )

    ax.axvline(x=perturb_time, color="black", linestyle=":", linewidth=1.0)
    # ax.text(
    #     perturb_time + 0.3, 0.97, "perturbation",
    #     transform=ax.get_xaxis_transform(),
    #     fontsize=8, va="top", ha="left",
    # )

    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Recovery (%)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    open_axes(ax)
    plt.show()


#################################################################
#################################################################
# Plots

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- Font handling: only use Inter if it's actually installed ---
available_fonts = {f.name for f in fm.fontManager.ttflist}
preferred_font = "Inter" if "Inter" in available_fonts else "DejaVu Sans"

rc_update = {
    "font.family": "sans-serif",
    "font.sans-serif": [preferred_font, "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "hatch.linewidth": 0.6,
}
if preferred_font == "Inter":
    rc_update.update(
        {
            "mathtext.fontset": "custom",
            "mathtext.rm": "Inter",
            "mathtext.it": "Inter:italic",
            "mathtext.bf": "Inter:bold",
        }
    )
plt.rcParams.update(rc_update)

# --- Compute recovery -----------------------------------------------------
REE_list = [
    e
    for e in m.fs.leach_soln.component_list
    if e not in ["H2O", "H", "HSO4", "SO4", "Cl", "H2C2O4"]
]

recovery = {}
for k in REE_list:
    oxide = f"{k}O" if k == "Ca" else f"{k}2O3"
    recovery[k] = [
        (
            (
                m.fs.leach.liquid_outlet.flow_vol[t]()
                * m.fs.leach.liquid_outlet.conc_mass_comp[t, k]()
                - m.fs.leach.liquid_inlet.flow_vol[t]()
                * m.fs.leach.liquid_inlet.conc_mass_comp[t, k]()
            )
            / (
                m.fs.leach.solid_inlet.flow_mass[t]()
                * m.fs.leach.solid_inlet.mass_frac_comp[t, oxide]()
                * 1e6
            )
        )
        * 100
        for t in m.fs.time
    ]


# --- Plot configuration ---------------------------------------------------
def cation_label(el: str) -> str:
    return rf"{el}$^{{3+}}$"


def open_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(top=False, right=False)


# Six distinct Okabe-Ito (Wong) colors, paired with three line styles per panel
series_styles = {
    "Ce": {"color": "#CC79A7", "linestyle": "-"},  # rose,         solid
    "Pr": {"color": "#E69F00", "linestyle": "--"},  # orange,       dashed
    "Nd": {"color": "#56B4E9", "linestyle": "-."},  # sky blue,     dash-dot
    "Sm": {"color": "#009E73", "linestyle": "-"},  # bluish green, solid
    "Y": {"color": "#0072B2", "linestyle": "--"},  # blue,         dashed
    "Dy": {"color": "#D55E00", "linestyle": "-."},  # vermillion,   dash-dot
}

light_rees = ["Ce", "Pr", "Nd"]
heavy_rees = ["Sm", "Y", "Dy"]

t_arr = list(m.fs.time)


def plot_group(group):
    fig, ax = plt.subplots(figsize=(3.5, 2.8), constrained_layout=True)

    for el in group:
        ax.plot(
            t_arr,
            recovery[el],
            label=cation_label(el),
            linewidth=1.8,
            **series_styles[el],
        )

    ax.axvline(x=perturb_time, color="black", linestyle=":", linewidth=1.0)
    # ax.text(
    #     perturb_time + 0.3, 0.97, "perturbation",
    #     transform=ax.get_xaxis_transform(),
    #     fontsize=8, va="top", ha="left",
    # )

    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Recovery (%)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    open_axes(ax)
    plt.show()


plot_group(light_rees)
plot_group(heavy_rees)


# Acid concentration change
# plt.rcParams.update({
#     "font.family": "serif",
#     "font.sans-serif": ["Inter"],
#     "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
#     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
#     "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
#     "axes.linewidth": 0.8,
#     "xtick.major.width": 0.8, "ytick.major.width": 0.8,
#     "xtick.direction": "in", "ytick.direction": "in",
#     "xtick.major.pad": 4, "ytick.major.pad": 4,
#     "axes.spines.top": False, "axes.spines.right": False,
# })

# t_points = sorted(m.fs.time)
# # H+ mass conc -> H2SO4 molar conc:  c_H2SO4 = c_H[mg/L] / (2 * 1000)
# acid_M = [m.fs.leach.liquid_inlet.conc_mass_comp[t, "H"]() / (2 * 1e3)
#           for t in t_points]

# fig, ax = plt.subplots(figsize=(3.5, 3.0))
# ax.plot(t_points, acid_M,
#         drawstyle="steps-post", linewidth=1.2, color="#D62728")

# ax.axvline(perturb_time, color="black", linestyle=":", linewidth=0.8)
# y_top = max(acid_M) * 1.15
# ax.set_ylim(0, y_top)
# # ax.text(perturb_time, y_top * 0.98, " perturbation",
# #         fontsize=7, va="top", ha="left")

# ax.set_xlabel("Time (h)")
# ax.set_ylabel(r"H$_2$SO$_4$ feed concentration (mol/L)")
# ax.set_xlim(0, time_duration)
# fig.tight_layout()
# plt.show()


# Solid flow rate change
# ---  solid feed flow rate plot -------------------------------------------
# y_data = [m.fs.leach.solid_inlet.flow_mass[t]() for t in t_arr]

# fig, ax = plt.subplots(figsize=(3.5, 2.8), constrained_layout=True)
# ax.plot(
#     t_arr, y_data,
#     color="#1F77B4", linewidth=1.8, linestyle="-",
# )
# ax.axvline(x=perturb_time, color="black", linestyle=":", linewidth=1.0)
# ax.set_xlabel("Time (h)")
# ax.set_ylabel("Solid feed flow rate (kg/hr)")
# open_axes(ax)
# # anchor_x_at_zero(ax)
# ax.set_ylim(0, max(y_data) * 1.5)
# plt.show()
