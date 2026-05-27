from pyomo.environ import (
    ConcreteModel,
    SolverFactory,
    Suffix,
    TransformationFactory,
    units,
    value,
)

import matplotlib.pyplot as plt
import numpy as np

from idaes.core import FlowsheetBlock
from idaes.core.scaling import CustomScalerBase

from prommis.leaching.leach_train import (
    LeachingTrain,
    LeachingTrainScaler,
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

# Expected outlet metal concentrations [mg/L], derived from the solid feed
# composition (ree_mass_frac) × S/L ratio (≈ 0.20 kg/L) × metal-in-oxide
# mass fraction × experimental recovery (Exp_data). Absent species
# (Ce, Pr, Gd) omitted to stay consistent with ree_mass_frac.
_LIQ_OUTLET_SF = {
    "Al": 1 / 15600,
    "Fe": 1 / 6400,
    "Ca": 1 / 2700,
    "Sc": 1 / 2.6,
    "Y": 1 / 3.9,
    "La": 1 / 8.4,
    "Nd": 1 / 4.5,
    "Sm": 1 / 1.56,
    "Dy": 1 / 0.011,
}

# Expected per-species solid conversion, taken as the experimental recovery
# (Exp_data, fraction). Absent oxides omitted — they default to SF=1 via
# the .get(comp, 1) lookup and their conversion is fixed at ~1 elsewhere.
_CONV_SF = {
    "Al2O3": 1 / 0.63,
    "Fe2O3": 1 / 0.64,
    "CaO": 1 / 0.64,
    "Sc2O3": 1 / 0.63,
    "Y2O3": 1 / 0.64,
    "La2O3": 1 / 0.72,
    "Nd2O3": 1 / 0.80,
    "Sm2O3": 1 / 0.74,
    "Dy2O3": 1 / 0.07,
    "inerts": 1,
}


def _scale_leach_train_blocks(model, csb):
    """Apply physical-state scaling to all leach train sub-blocks.

    All inlet/volume/composition scaling factors are derived from the
    currently *fixed* model values via ``pyomo.value`` — so the scaling
    self-consistently tracks the operating point and absent species
    (fixed at trace values) get the correct order of magnitude rather
    than the parameter package's nominal defaults.

    Outlet REE concentrations and per-species conversions use the
    hard-coded estimates derived from ``ree_mass_frac`` × experimental
    recovery. Absent species' outlet concentrations are pinned with
    explicit small SFs (their value stays at the trace inlet level).
    Finally delegates to the ``LeachingTrainScaler`` for extents and
    remaining constraint scaling.
    """
    liq_scaler = SulfuricAcidLeachingPropertiesScaler()
    solid_scaler = CoalRefusePropertiesScaler()
    train_scaler = LeachingTrainScaler()

    # SFs derived from the fixed operating point (internal units).
    vol_sf = 1 / value(model.fs.leach.volume[0, 1])
    solid_flow_sf = 1 / value(model.fs.leach.solid_inlet.flow_mass[0])
    liq_flow_sf = 1 / value(model.fs.leach.liquid_inlet.flow_vol[0])
    h_sf = 1 / value(model.fs.leach.liquid_inlet.conc_mass_comp[0, "H"])
    so4_sf = 1 / value(model.fs.leach.liquid_inlet.conc_mass_comp[0, "SO4"])

    # Absent species in the solid (fixed at 1e-12) → matching absent
    # ions in the liquid (fixed at 1e-10 mg/L). Their outlet variables
    # stay at trace levels; give them a moderate SF so the scaled value
    # is small but the resulting Jacobian entries stay well within
    # double precision (avoid SF = 1/trace, which would be ~ 1e10–1e12
    # and produce SF·MW ≈ 1e15 in the molar-concentration coupling).
    absent_liq = {"Ce", "Pr", "Gd"}
    absent_liq_sf = 1e3
    absent_solid_sf = 1e6

    # Solid mass fractions: SF = 1/(fixed inlet value), floored at 1e-6
    # so present species track their true magnitude, absent species get
    # the moderate absent_solid_sf instead of an extreme 1/1e-12.
    solid_xfrac_sf = {}
    for comp in model.fs.coal.component_list:
        x0 = value(model.fs.leach.solid_inlet.mass_frac_comp[0, comp])
        if x0 <= 1e-9:  # absent species
            solid_xfrac_sf[comp] = absent_solid_sf
        else:
            solid_xfrac_sf[comp] = 1 / x0

    csb.set_variable_scaling_factor(model.fs.leach.volume[0, 1], vol_sf)

    # Solid inlet: flow mass + mass fractions matching the fixed feed
    for blk in model.fs.leach.mscontactor.solid_inlet_state.values():
        csb.set_variable_scaling_factor(blk.flow_mass, solid_flow_sf)
        for comp, sf in solid_xfrac_sf.items():
            csb.set_variable_scaling_factor(blk.mass_frac_comp[comp], sf)
        solid_scaler.variable_scaling_routine(blk, overwrite=False)

    # Solid outlet: flow mass, mass fractions (by feed composition), conversions
    for blk in model.fs.leach.mscontactor.solid.values():
        csb.set_variable_scaling_factor(blk.flow_mass, solid_flow_sf)
        for comp, sf in solid_xfrac_sf.items():
            csb.set_variable_scaling_factor(blk.mass_frac_comp[comp], sf)

        if hasattr(blk, "conversion_comp"):
            for comp in model.fs.coal.component_list:
                sf = _CONV_SF.get(comp, 1)
                csb.set_variable_scaling_factor(blk.conversion_comp[comp], sf)
        solid_scaler.constraint_scaling_routine(blk, overwrite=False)

    # Liquid inlet: condition-specific H+ and SO4 + trace absent ions;
    # scaler then fills conc_mol_comp / pH consistent with these.
    for blk in model.fs.leach.mscontactor.liquid_inlet_state.values():
        csb.set_variable_scaling_factor(blk.flow_vol, liq_flow_sf)
        csb.set_variable_scaling_factor(blk.conc_mass_comp["H"], h_sf)
        csb.set_variable_scaling_factor(blk.conc_mass_comp["SO4"], so4_sf)
        for ab in absent_liq:
            csb.set_variable_scaling_factor(blk.conc_mass_comp[ab], absent_liq_sf)
        liq_scaler.variable_scaling_routine(blk, overwrite=False)

    for blk in model.fs.leach.mscontactor.liquid.values():
        csb.set_variable_scaling_factor(blk.flow_vol, liq_flow_sf)
        csb.set_variable_scaling_factor(blk.conc_mass_comp["H"], h_sf)
        csb.set_variable_scaling_factor(blk.conc_mass_comp["SO4"], so4_sf)
        for comp, sf in _LIQ_OUTLET_SF.items():
            csb.set_variable_scaling_factor(blk.conc_mass_comp[comp], sf)
        for ab in absent_liq:
            csb.set_variable_scaling_factor(blk.conc_mass_comp[ab], absent_liq_sf)
        liq_scaler.variable_scaling_routine(blk, overwrite=False)
        liq_scaler.constraint_scaling_routine(blk, overwrite=False)

    # ---- Extent variables + all constraint scaling via LeachingTrainScaler ----

    train_scaler.variable_scaling_routine(model.fs.leach, overwrite=False)
    train_scaler.constraint_scaling_routine(model.fs.leach, overwrite=False)


m = ConcreteModel()
m.fs = FlowsheetBlock(dynamic=False)
m.fs.leach_soln = SulfuricAcidLeachingParameters()
m.fs.coal = CoalRefuseParameters()
m.fs.leach_rxns = CoalRefuseLeachingReactionParameterBlock()

m.fs.leach = LeachingTrain(
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

m.fs.leach[:].liquid_inlet.flow_vol.fix(454.249 * units.L / units.hour)
m.fs.leach[:].liquid_inlet.conc_mass_comp.fix(1e-10 * units.mg / units.L)
m.fs.leach[:].liquid_inlet.conc_mass_comp[:, "H2O"].fix(1e6 * units.mg / units.L)
m.fs.leach[:].liquid_inlet.conc_mass_comp[0, "HSO4"].fix(1e-8 * units.mg / units.L)
m.fs.leach.liquid_inlet.conc_mass_comp[0, "H"].fix(2 * 0.3 * 1e3 * units.mg / units.L)
m.fs.leach.liquid_inlet.conc_mass_comp[0, "SO4"].fix(0.3 * 96e3 * units.mg / units.L)
m.fs.leach.solid_inlet.flow_mass.fix(90.7185 * units.kg / units.hour)

ree_mass_frac = {
    "Sc2O3": 4.63e-06,
    "Y2O3": 3.80e-05,
    "La2O3": 3.80e-05,
    "Ce2O3": 3.77e-05,
    "Nd2O3": 2.68e-05,
    "Sm2O3": 4.68e-06,
    "Dy2O3": 5.20e-06,
    "Al2O3": 2.38e-02,
    "CaO": 1.12e-02,
    "Fe2O3": 9.25e-03,
}

absent_solid = ["Pr2O3", "Gd2O3"]

for comp, x in ree_mass_frac.items():
    m.fs.leach.solid_inlet.mass_frac_comp[0, comp].fix(x)
for comp in absent_solid:
    m.fs.leach.solid_inlet.mass_frac_comp[0, comp].fix(1e-12)
m.fs.leach.solid_inlet.mass_frac_comp[0, "inerts"].fix(
    1.0 - sum(ree_mass_frac.values())
)

# The leach reaction defines each oxide's conversion (and thus the reported
# recovery) RELATIVE to coal.mass_frac_comp_initial, which is treated as the
# fresh, unreacted feed. It must therefore equal the actual solid feed above.
# Left at the default UKy assay, the SHXP feed makes e.g. CaO conversion
# evaluate to -1.46 at the inlet -- outside its (0, 1) bounds -- so the model
# is infeasible no matter how it is scaled. Sync it to the SHXP feed.
for comp in m.fs.coal.component_list:
    m.fs.coal.mass_frac_comp_initial[comp] = value(
        m.fs.leach.solid_inlet.mass_frac_comp[0, comp]
    )

msc = m.fs.leach.mscontactor
for s in msc.elements:
    for r in absent_solid:
        msc.heterogeneous_reaction_extent[0, s, r].fix(0.0)
        msc.heterogeneous_reaction_extent_constraint[0, s, r].deactivate()

for s in msc.elements:
    for r in absent_solid:
        msc.solid[0, s].conversion_comp[r].fix(0.999)
        msc.solid[0, s].conversion_comp_eqn[r].deactivate()

m.fs.leach.volume.fix(100 * units.gallon)


# for blk in m.fs.leach.mscontactor.liquid.values():
#     blk.conc_mol_comp["H"].setlb(1e-8)

m.scaling_factor = Suffix(direction=Suffix.EXPORT)
csb = CustomScalerBase()

_scale_leach_train_blocks(m, csb)

# Initialize before the final solve. Solving from a cold start lands ipopt at
# a locally-infeasible point; the scaling factors set just above are what let
# the initializer's internal solves converge.
m.fs.leach.default_initializer().initialize(m.fs.leach)

scaling = TransformationFactory("core.scale_model")
scaled_model = scaling.create_using(m, rename=False)

solver = SolverFactory("ipopt_v2")
solver.options["max_iter"] = 5000
solver.options["halt_on_ampl_error"] = "yes"
solver.solve(scaled_model, tee=True)

scaling.propagate_solution(scaled_model, m)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
Exp_data = {
    "Sc2O3": 22.37907945,
    "Y2O3": 74.92418419,
    "La2O3": 59.28795056,
    "Ce2O3": 86.68004663,
    "Nd2O3": 49.28670074,
    "Sm2O3": 57.79054474,
    "Dy2O3": 27.32244004,
    "Al2O3": 4.071739985,
    "CaO": 34.29232524,
    "Fe2O3": 8.74419426,
}

recovery = {}
for e in m.fs.coal.component_list:
    if e not in ["Pr2O3", "Gd2O3", "inerts"]:
        recovery[e] = m.fs.leach.recovery[0, e]()

# Align keys (same order as Exp_data)
components = list(Exp_data.keys())
exp_vals = [Exp_data[c] for c in components]
mod_vals = [recovery[c] for c in components]


# Pretty labels with subscripts (e.g., Sc2O3 -> Sc$_2$O$_3$)
def pretty(label):
    return label.replace("2O3", r"$_2$O$_3$")


xlabels = [pretty(c) for c in components]

# --- Journal-ready styling ---
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "axes.linewidth": 1.0,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    }
)

x = np.arange(len(components))
width = 0.38

fig, ax = plt.subplots(figsize=(7.0, 4.0))

bars_exp = ax.bar(
    x - width / 2,
    exp_vals,
    width,
    label="Experimental",
    color="#4C72B0",
    edgecolor="black",
    linewidth=0.8,
)
bars_mod = ax.bar(
    x + width / 2,
    mod_vals,
    width,
    label="Model",
    color="#DD8452",
    edgecolor="black",
    linewidth=0.8,
)

ax.set_xlabel("Rare earth oxide")
ax.set_ylabel("Recovery (%)")
ax.set_xticks(x)
ax.set_xticklabels(xlabels, rotation=0)
ax.set_ylim(0, max(max(exp_vals), max(mod_vals)) * 1.15)
ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
ax.tick_params(which="both", top=True, right=True)
ax.legend(loc="upper right", ncol=2)

# Optional value labels on top of each bar
for bars in (bars_exp, bars_mod):
    for b in bars:
        h = b.get_height()
        ax.annotate(
            f"{h:.1f}",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

plt.tight_layout()
# plt.savefig('recovery_comparison.png', dpi=600)
# plt.savefig('recovery_comparison.pdf')  # vector for submission
plt.show()
