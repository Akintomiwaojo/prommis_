#####################################################################################################
# "PrOMMiS" was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# ("PrOMMiS") initiative, and is copyright (c) 2023-2025 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
r"""
Batch Leach Train
=================

Author: Andrew Lee

The Batch Leach Train unit model represents a closed-system (batch) leaching vessel in which a
fixed charge of solid and liquid are contacted and undergo heterogeneous chemical reactions.
Unlike the continuous :class:`LeachingTrain`, there is no continuous inflow or outflow.

The model uses a :class:`ControlVolume0DBlock` to hold the liquid phase property blocks, with
separate state blocks for the initial and final solid phase. Custom batch material balance
constraints relate the initial conditions to the final state via the reaction extents.

Configuration Arguments
-----------------------

A Batch Leach Train model requires a "heterogeneous reaction package" to define reactions and
reaction rates. A ``number_of_tanks`` option is not present — the batch vessel is a single
well-mixed unit.

Degrees of Freedom
------------------

The user must fix the following variables before solving:

* ``liquid_cv.volume[t]``  — volume of liquid in the vessel [m³ or L]
* ``batch_time[t]``  — duration of the batch [hr]
* ``solid_mass_total[t]``  — total mass of solid charged to the vessel [kg]
* ``solid_cv.properties_in[t].mass_frac_comp[j]``  — initial mass fraction of each solid component
* ``liquid_cv.properties_in[t].conc_mass_comp[j]``  — initial liquid concentrations [mg/L]
  (excluding H2O and HSO4 when the liquid property package fixes those internally)

Model Structure
---------------

* ``liquid_cv``  — :class:`ControlVolume0DBlock` for the liquid phase
  * ``liquid_cv.properties_in[t]``  — initial liquid state (user-specified)
  * ``liquid_cv.properties_out[t]``  — final liquid state (solved)
  * ``liquid_cv.volume[t]``  — liquid volume
* ``solid_cv``  — :class:`ControlVolume0DBlock` for the solid phase
  * ``solid_cv.properties_in[t]``  — initial solid state (user-specified)
  * ``solid_cv.properties_out[t]``  — final solid state (solved)
  * ``solid_cv.volume[t]``  — solid volume (available for geometric use)
* ``liquid[t]``  — Reference to ``liquid_cv.properties_out[t]`` (used by reaction package)
* ``solid[t]``  — Reference to ``solid_cv.properties_out[t]`` (used by reaction package)
* ``heterogeneous_reactions[t]``  — heterogeneous reaction block

Additional Constraints
----------------------

* ``reaction_extent_eqn`` — extent of each reaction per unit time [mol/hr]:

  .. math:: X_{t,r} = r_{t,r} \cdot V_{t}

  where :math:`r_{t,r}` is the volumetric reaction rate [mol/L/hr] and :math:`V_t` is the
  liquid volume [L].

* ``liquid_flow_vol_in_eqn``, ``liquid_flow_vol_out_eqn`` — link ``flow_vol`` to
  ``liquid_cv.volume / batch_time`` so that the reaction package computes the correct pulp
  density.

* ``solid_flow_mass_in_eqn`` — links ``solid_in.flow_mass`` to
  ``solid_mass_total / batch_time``.

* ``liquid_material_balance`` — batch liquid component balance:

  .. math:: \dot{V}_L \left( c_{j,\text{out}} - c_{j,\text{in}} \right) = \sum_r \nu_{r,j}^{L} X_{t,r}

* ``solid_material_balance`` — batch solid component balance:

  .. math::
     \frac{\dot{m}_{\text{out}} \cdot x_{j,\text{out}} - \dot{m}_{\text{in}} \cdot x_{j,\text{in}}}{M_j}
     = \sum_r \nu_{r,j}^{S} X_{t,r}

Recovery
--------

Recovery of solid component :math:`j` is:

.. math::

    \text{recovery}_{t,j} =
    \left(1 - \frac{\dot{m}_{\text{out}} \cdot x_{j,\text{out}}}
                   {\dot{m}_{\text{in}} \cdot x_{j,\text{in}}}\right) \times 100

"""
from pyomo.common.config import Bool, ConfigDict, ConfigValue
from pyomo.environ import Block, Constraint, Reference, Var, units

from idaes.core import (
    ControlVolume0DBlock,
    UnitModelBlockData,
    declare_process_block_class,
    useDefault,
)
from idaes.core.initialization import ModularInitializerBase
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.scaling import CustomScalerBase


class BatchLeachingTrainInitializer(ModularInitializerBase):
    """
    General purpose Initializer for the Batch Leaching Train unit model.
    """

    CONFIG = ModularInitializerBase.CONFIG()

    CONFIG.declare(
        "ssc_solver_options",
        ConfigDict(
            implicit=True,
            description="Dict of arguments for solver calls by ssc_solver",
        ),
    )
    CONFIG.declare(
        "calculate_variable_options",
        ConfigDict(
            implicit=True,
            description="Dict of options to pass to 1x1 block solver",
            doc="Dict of options to pass to calc_var_kwds argument in "
            "scc_solver method.",
        ),
    )

    def initialization_routine(self, model: Block):
        """
        Initialization routine for BatchLeachingTrain.

        Args:
            model: model to be initialized

        Returns:
            solver results
        """
        solver = self._get_solver()
        results = solver.solve(model)
        return results


StreamCONFIG = ConfigDict()
StreamCONFIG.declare(
    "property_package",
    ConfigValue(
        default=useDefault,
        domain=is_physical_parameter_block,
        description="Property package to use for given stream",
    ),
)
StreamCONFIG.declare(
    "property_package_args",
    ConfigDict(
        implicit=True,
        description="Dict of arguments to use for constructing property package",
    ),
)
StreamCONFIG.declare(
    "has_energy_balance",
    ConfigValue(
        default=False,
        domain=Bool,
        doc="Bool indicating whether to include energy balance for stream.",
    ),
)
StreamCONFIG.declare(
    "has_pressure_balance",
    ConfigValue(
        default=False,
        domain=Bool,
        doc="Bool indicating whether to include pressure balance for stream.",
    ),
)


@declare_process_block_class("BatchLeachingTrain")
class BatchLeachingTrainData(UnitModelBlockData):
    """
    Batch Leaching Train Unit Model Class.

    Models a closed-system batch leaching operation using a ControlVolume0DBlock
    for the liquid phase. No continuous inflow or outflow. Initial and final states
    of both liquid and solid are represented explicitly, connected by heterogeneous
    reaction extents and batch material balance constraints.
    """

    default_initializer = BatchLeachingTrainInitializer

    CONFIG = UnitModelBlockData.CONFIG()

    CONFIG.declare(
        "liquid_phase",
        StreamCONFIG(description="Liquid phase properties"),
    )
    CONFIG.declare(
        "solid_phase",
        StreamCONFIG(description="Solid phase properties"),
    )
    CONFIG.declare(
        "reaction_package",
        ConfigValue(
            description="Heterogeneous reaction package for leaching.",
        ),
    )
    CONFIG.declare(
        "reaction_package_args",
        ConfigValue(
            default=None,
            domain=dict,
            description="Arguments for heterogeneous reaction package.",
        ),
    )

    def build(self):
        """
        Build method for BatchLeachingTrain unit model.
        """
        super().build()

        time = self.flowsheet().time
        liquid_pkg = self.config.liquid_phase.property_package
        liquid_pkg_args = self.config.liquid_phase.property_package_args
        solid_pkg = self.config.solid_phase.property_package
        solid_pkg_args = self.config.solid_phase.property_package_args
        rxn_pkg = self.config.reaction_package
        rxn_pkg_args = self.config.reaction_package_args or {}

        # ------------------------------------------------------------------ #
        # 1. Liquid control volume
        #    - add_geometry()     → liquid_cv.volume[t]  [m³]
        #    - add_state_blocks() → liquid_cv.properties_in[t]  (defined_state=True)
        #                           liquid_cv.properties_out[t] (defined_state=False)
        # ------------------------------------------------------------------ #
        self.liquid_cv = ControlVolume0DBlock(
            property_package=liquid_pkg,
            property_package_args=liquid_pkg_args,
            dynamic=self.config.dynamic,
            has_holdup=self.config.has_holdup,
        )
        self.liquid_cv.add_geometry()
        self.liquid_cv.add_state_blocks(has_phase_equilibrium=False)

        # ------------------------------------------------------------------ #
        # 2. Solid control volume
        #    - add_geometry()     → solid_cv.volume[t]  [m³]  (not used in balances
        #                           but available for geometric calculations)
        #    - add_state_blocks() → solid_cv.properties_in[t]  (defined_state=True)
        #                           solid_cv.properties_out[t] (defined_state=False)
        # ------------------------------------------------------------------ #
        self.solid_cv = ControlVolume0DBlock(
            property_package=solid_pkg,
            property_package_args=solid_pkg_args,
            dynamic=self.config.dynamic,
            has_holdup=self.config.has_holdup,
        )
        self.solid_cv.add_geometry()
        self.solid_cv.add_state_blocks(has_phase_equilibrium=False)

        # ------------------------------------------------------------------ #
        # 3. Batch-specific variables
        # ------------------------------------------------------------------ #
        uom = liquid_pkg.get_metadata().default_units

        self.batch_time = Var(
            time,
            initialize=1,
            units=uom.TIME,
            doc="Duration of the batch [hr].",
        )

        self.solid_mass_total = Var(
            time,
            initialize=1,
            units=uom.MASS,
            doc="Total mass of solid charged to the batch vessel [kg].",
        )

        # ------------------------------------------------------------------ #
        # 4. References so the reaction package can find liquid and solid
        #    via b.parent_block().liquid[t] and b.parent_block().solid[t]
        # ------------------------------------------------------------------ #
        self.liquid = Reference(self.liquid_cv.properties_out)
        self.solid = Reference(self.solid_cv.properties_out)

        # ------------------------------------------------------------------ #
        # 5. Heterogeneous reaction block
        # ------------------------------------------------------------------ #
        self.heterogeneous_reactions = rxn_pkg.build_reaction_block(
            time,
            **rxn_pkg_args,
        )

        # ------------------------------------------------------------------ #
        # 6. Reaction extent per unit time  X[t, r]  [mol/hr]
        #    X = rate [mol/L/hr] × V_liquid [L]
        # ------------------------------------------------------------------ #
        self.heterogeneous_reaction_extent = Var(
            time,
            rxn_pkg.reaction_idx,
            initialize=0,
            units=uom.AMOUNT / uom.TIME,
            doc="Reaction extent per unit time [mol/hr].",
        )

        @self.Constraint(time, rxn_pkg.reaction_idx)
        def reaction_extent_eqn(b, t, r):
            return b.heterogeneous_reaction_extent[t, r] == units.convert(
                b.heterogeneous_reactions[t].reaction_rate[r]
                * b.liquid_cv.volume[t],
                to_units=uom.AMOUNT / uom.TIME,
            )

        # ------------------------------------------------------------------ #
        # 7. Link flow_vol = V_liquid / batch_time
        #    This ensures the pulp density in the reaction package is correct:
        #      eps = m_dot_solid / (m_dot_solid/dens + flow_vol)
        #          = (M_solid/t) / (M_solid/(t*dens) + V_liquid/t)
        #          = M_solid / (M_solid/dens + V_liquid)   [kg/L]
        # ------------------------------------------------------------------ #
        @self.Constraint(time)
        def liquid_flow_vol_in_eqn(b, t):
            return b.liquid_cv.properties_in[t].flow_vol == units.convert(
                b.liquid_cv.volume[t] / b.batch_time[t],
                to_units=liquid_pkg.get_metadata().default_units.FLOW_VOL,
            )

        @self.Constraint(time)
        def liquid_flow_vol_out_eqn(b, t):
            return b.liquid_cv.properties_out[t].flow_vol == units.convert(
                b.liquid_cv.volume[t] / b.batch_time[t],
                to_units=liquid_pkg.get_metadata().default_units.FLOW_VOL,
            )

        # ------------------------------------------------------------------ #
        # 8. Link solid_cv.properties_in.flow_mass = solid_mass_total / batch_time
        #    Keeps solid flow rate consistent with batch quantities.
        # ------------------------------------------------------------------ #
        @self.Constraint(time)
        def solid_flow_mass_in_eqn(b, t):
            return b.solid_cv.properties_in[t].flow_mass == units.convert(
                b.solid_mass_total[t] / b.batch_time[t],
                to_units=uom.MASS / uom.TIME,
            )

        # ------------------------------------------------------------------ #
        # 9. Liquid material balance
        #    flow_vol * (conc_mol_out[j] - conc_mol_in[j]) = Σ_r ν[r,j] * X[t,r]
        #
        #    H2O  → fixed at constant density by h2o_concentration constraint
        #           in properties_out; skip here.
        #    HSO4 → determined by Ka2 equilibrium in properties_out; skip here.
        # ------------------------------------------------------------------ #
        rxn_stoich = rxn_pkg.reaction_stoichiometry
        liquid_comp_list = liquid_pkg.component_list

        skip_liquid = {"H2O", "HSO4"}
        liquid_balance_comps = [j for j in liquid_comp_list if j not in skip_liquid]

        @self.Constraint(time, liquid_balance_comps)
        def liquid_material_balance(b, t, j):
            flow_vol = b.liquid_cv.properties_out[t].flow_vol
            conc_out = b.liquid_cv.properties_out[t].conc_mol_comp[j]
            conc_in = b.liquid_cv.properties_in[t].conc_mol_comp[j]

            transfer = sum(
                rxn_stoich[r, "liquid", j] * b.heterogeneous_reaction_extent[t, r]
                for r in rxn_pkg.reaction_idx
                if (r, "liquid", j) in rxn_stoich
            )
            return units.convert(
                flow_vol * (conc_out - conc_in),
                to_units=uom.AMOUNT / uom.TIME,
            ) == transfer

        # ------------------------------------------------------------------ #
        # 10. Solid material balance (all components, including inerts)
        #     (m_dot_out * x_out[j] - m_dot_in * x_in[j]) / mw[j]
        #         = Σ_r ν[r,j]^solid * X[t,r]
        #
        #     For inerts: ν = 0, so m_dot_out * x_out = m_dot_in * x_in.
        #     The solid_out.sum_mass_frac constraint (Σ x_out[j] = 1) together
        #     with these N_j component balances fully determines solid_out.flow_mass
        #     and solid_out.mass_frac_comp[j].
        # ------------------------------------------------------------------ #
        solid_comp_list = solid_pkg.component_list

        @self.Constraint(time, solid_comp_list)
        def solid_material_balance(b, t, j):
            m_dot_out = units.convert(
                b.solid_cv.properties_out[t].flow_mass
                * b.solid_cv.properties_out[t].mass_frac_comp[j]
                / solid_pkg.mw[j],
                to_units=uom.AMOUNT / uom.TIME,
            )
            m_dot_in = units.convert(
                b.solid_cv.properties_in[t].flow_mass
                * b.solid_cv.properties_in[t].mass_frac_comp[j]
                / solid_pkg.mw[j],
                to_units=uom.AMOUNT / uom.TIME,
            )
            transfer = sum(
                rxn_stoich[r, "solid", j] * b.heterogeneous_reaction_extent[t, r]
                for r in rxn_pkg.reaction_idx
                if (r, "solid", j) in rxn_stoich
            )
            return m_dot_out - m_dot_in == transfer

        # ------------------------------------------------------------------ #
        # 11. Recovery expression
        #     (1 - final_solid_mass_j / initial_solid_mass_j) × 100
        # ------------------------------------------------------------------ #
        @self.Expression(
            time,
            solid_comp_list,
            doc="Percent recovery of each solid component from the batch",
        )
        def recovery(b, t, j):
            m_in = (
                b.solid_cv.properties_in[t].flow_mass
                * b.solid_cv.properties_in[t].mass_frac_comp[j]
            )
            m_out = (
                b.solid_cv.properties_out[t].flow_mass
                * b.solid_cv.properties_out[t].mass_frac_comp[j]
            )
            return (1 - m_out / m_in) * 100

    def _get_performance_contents(self, time_point=0):
        exprs = {}
        for j in self.config.solid_phase.property_package.component_list:
            exprs[f"Recovery % {j}"] = self.recovery[time_point, j]
        return {"exprs": exprs}
