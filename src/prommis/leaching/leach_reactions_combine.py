#####################################################################################################
# "PrOMMiS" was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# ("PrOMMiS") initiative, and is copyright (c) 2023-2025 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
r"""
Combined shrinking core leaching model.



The reaction rate is based on a combined shrinking core model (SCM) accounting for three
resistances in series:

1. Film diffusion at the particle surface
2. Diffusion through the ash/product layer
3. Surface chemical reaction


Includes reactions for the following components with H2SO4:

* Rare Earth Oxides: Sc2O3, Y2O3, La2O3, Ce2O3, Pr2O3, Nd2O3, Sm2O3, Gd2O3, Dy2O3
* Impurities: Al2O3, CaO, Fe2O3

"""

from pyomo.common.config import ConfigValue
from pyomo.environ import Expression, Param, Set, Var, units

from idaes.core import ProcessBlock, ProcessBlockData, declare_process_block_class
from idaes.core.base import property_meta
from idaes.core.scaling import CustomScalerBase
from idaes.core.util.misc import add_object_reference


class LeachCombinedReactionScaler(CustomScalerBase):
    """
    Scaler for the combined leach reaction package.
    No variables or constraints, so no need for scaling.
    """

    DEFAULT_SCALING_FACTORS = {}

    def variable_scaling_routine(
        self, model, overwrite: bool = False, submodel_scalers: dict = None
    ):
        pass

    def constraint_scaling_routine(
        self, model, overwrite: bool = False, submodel_scalers: dict = None
    ):
        pass


# -----------------------------------------------------------------------------
# Leach solution property package
@declare_process_block_class("CoalRefuseLeachingCombinedReactionParameterBlock")
class CoalRefuseLeachingCombinedReactionParameterData(
    ProcessBlockData, property_meta.HasPropertyClassMetadata
):
    """
    Reaction package for heterogeneous reactions involved in leaching REEs from
    solid coal refuse using H2SO4.

    This reaction package is designed to be used with the LeachTrain or MSContactor
    unit models and assumed two streams named 'liquid' and 'solid'.

    Implements a combined shrinking core model (SCM) with film diffusion,
    ash/product layer diffusion, and surface reaction resistances in series.

    Includes reactions for the following components with H2SO4:

    * Rare Earth Oxides: Sc2O3, Y2O3, La2O3, Ce2O3, Pr2O3, Nd2O3, Sm2O3, Gd2O3, Dy2O3
    * Impurities: Al2O3, CaO, Fe2O3


    Note: K_film [m/hour], D_e [m^2/hour], k' [mol/m^2/hour], and A_ox [dimensionless]
    parameters must be fitted to experimental data.
    """

    def build(self):
        super().build()

        self._reaction_block_class = CoalRefuseLeachingCombinedReactionBlock

        self.reaction_idx = Set(
            initialize=[
                "Al2O3",
                "Fe2O3",
                "CaO",
                "Sc2O3",
                "Y2O3",
                "La2O3",
                "Ce2O3",
                "Pr2O3",
                "Nd2O3",
                "Sm2O3",
                "Gd2O3",
                "Dy2O3",
            ]
        )

        self.reaction_stoichiometry = {
            ("Al2O3", "liquid", "Al"): 2,
            ("Al2O3", "liquid", "H2O"): 3,
            ("Al2O3", "solid", "Al2O3"): -1,
            ("Al2O3", "liquid", "H"): -6,
            ("Fe2O3", "liquid", "Fe"): 2,
            ("Fe2O3", "liquid", "H2O"): 3,
            ("Fe2O3", "solid", "Fe2O3"): -1,
            ("Fe2O3", "liquid", "H"): -6,
            ("CaO", "liquid", "Ca"): 1,
            ("CaO", "liquid", "H2O"): 1,
            ("CaO", "solid", "CaO"): -1,
            ("CaO", "liquid", "H"): -2,
            ("Sc2O3", "liquid", "Sc"): 2,
            ("Sc2O3", "liquid", "H2O"): 3,
            ("Sc2O3", "solid", "Sc2O3"): -1,
            ("Sc2O3", "liquid", "H"): -6,
            ("Y2O3", "liquid", "Y"): 2,
            ("Y2O3", "liquid", "H2O"): 3,
            ("Y2O3", "solid", "Y2O3"): -1,
            ("Y2O3", "liquid", "H"): -6,
            ("La2O3", "liquid", "La"): 2,
            ("La2O3", "liquid", "H2O"): 3,
            ("La2O3", "solid", "La2O3"): -1,
            ("La2O3", "liquid", "H"): -6,
            ("Ce2O3", "liquid", "Ce"): 2,
            ("Ce2O3", "liquid", "H2O"): 3,
            ("Ce2O3", "solid", "Ce2O3"): -1,
            ("Ce2O3", "liquid", "H"): -6,
            ("Pr2O3", "liquid", "Pr"): 2,
            ("Pr2O3", "liquid", "H2O"): 3,
            ("Pr2O3", "solid", "Pr2O3"): -1,
            ("Pr2O3", "liquid", "H"): -6,
            ("Nd2O3", "liquid", "Nd"): 2,
            ("Nd2O3", "liquid", "H2O"): 3,
            ("Nd2O3", "solid", "Nd2O3"): -1,
            ("Nd2O3", "liquid", "H"): -6,
            ("Sm2O3", "liquid", "Sm"): 2,
            ("Sm2O3", "liquid", "H2O"): 3,
            ("Sm2O3", "solid", "Sm2O3"): -1,
            ("Sm2O3", "liquid", "H"): -6,
            ("Gd2O3", "liquid", "Gd"): 2,
            ("Gd2O3", "liquid", "H2O"): 3,
            ("Gd2O3", "solid", "Gd2O3"): -1,
            ("Gd2O3", "liquid", "H"): -6,
            ("Dy2O3", "liquid", "Dy"): 2,
            ("Dy2O3", "liquid", "H2O"): 3,
            ("Dy2O3", "solid", "Dy2O3"): -1,
            ("Dy2O3", "liquid", "H"): -6,
        }

        self.R_p = Param(
            initialize=4.35e-6,
            units=units.m,
            mutable=True,
            doc="Mean particle radius",
        )

        self.rho_liquid = Param(
            initialize=1.0,
            units=units.kg / units.litre,
            mutable=True,
            doc="Liquid phase density",
        )

        # Volume fraction of solid in the solid inlet stream [-]
        # The solid feed is a slurry (solid + carrier liquid), so phi_s_inlet
        # is the solid volume fraction in that particular inlet stream, not the
        # overall reactor mixture. Must be set based on feed preparation conditions.
        self.phi_s_inlet = Param(
            initialize=0.95,
            units=units.dimensionless,
            mutable=True,
            doc="Volume fraction of solid in the solid inlet slurry stream",
        )

        # Reaction order for H+ (dimensionless) - fitted per oxide
        self.A_ox = Var(
            self.reaction_idx,
            initialize={
                "Al2O3": 1.496716606,
                "Fe2O3": 0.902948175,
                "CaO": 0.159744406,
                "Sc2O3": 0.763763942,
                "Y2O3": 0.580700988,
                "La2O3": 0.443101432,
                "Ce2O3": 0.601391182,
                "Pr2O3": 0.501916124,
                "Nd2O3": 0.702951111,
                "Sm2O3": 0.578717372,
                "Gd2O3": 1.063666638,
                "Dy2O3": 0.428087853,
            },
            bounds=(1e-4, 2),
            units=units.dimensionless,
            doc="Reaction order with respect to H+ concentration",
        )

        # Surface reaction rate constant [mol/m²/hour] - fitted per oxide
        self.k_prime = Var(
            self.reaction_idx,
            initialize={
                # "Al2O3": 405.5050676,
                # "Fe2O3": 11.34710708,
                "Al2O3": 4e-3,
                "Fe2O3": 1e-3,
                "CaO": 0.081844698,
                "Sc2O3": 0.000755073,
                "Y2O3": 0.000528612,
                "La2O3": 0.001028295,
                "Ce2O3": 0.007050046,
                "Pr2O3": 0.000522858,
                "Nd2O3": 0.006289942,
                "Sm2O3": 0.000252479,
                "Gd2O3": 0.013408467,
                "Dy2O3": 4.56708e-05,
            },
            # bounds=(1e-10, 1e6),
            bounds=(1e-6, 1e2),
            units=units.mol * units.m**-2 * units.hour**-1,
            doc="Surface reaction rate constant [mol/m²/hour]",
        )

        # Film mass transfer coefficient [m/hour] - fitted per oxide
        self.K_film = Var(
            self.reaction_idx,
            # initialize=0.5,
            # bounds=(1e-3, 10),
            initialize=1e-2,
            bounds=(1e-3, 1e2),
            units=units.m * units.hour**-1,
            doc="Film mass transfer coefficient",
        )

        # Effective diffusivity in ash/product layer [m^2/hour] - fitted per oxide
        self.D_e = Var(
            self.reaction_idx,
            initialize=1e-7,
            # bounds=(1e-7, 1),
            bounds=(1e-10, 1e-5),
            units=units.m**2 * units.hour**-1,
            doc="Effective diffusivity through ash/product layer",
        )

    @classmethod
    def define_metadata(cls, obj):
        obj.add_default_units(
            {
                "time": units.hour,
                "length": units.m,
                "mass": units.kg,
                "amount": units.mol,
                "temperature": units.K,
            }
        )

    @property
    def reaction_block_class(self):
        if self._reaction_block_class is not None:
            return self._reaction_block_class
        else:
            raise AttributeError(
                "{} has not assigned a ReactionBlock class to be associated "
                "with this reaction package. Please contact the developer of "
                "the reaction package.".format(self.name)
            )

    def build_reaction_block(self, *args, **kwargs):
        """
        Methods to construct a ReactionBlock associated with this
        ReactionParameterBlock. This will automatically set the parameters
        construction argument for the ReactionBlock.

        Returns:
            ReactionBlock

        """
        default = kwargs.pop("default", {})
        initialize = kwargs.pop("initialize", {})

        if initialize == {}:
            default["parameters"] = self
        else:
            for i in initialize.keys():
                initialize[i]["parameters"] = self

        return self.reaction_block_class(  # pylint: disable=not-callable
            *args, **kwargs, **default, initialize=initialize
        )


class _CoalRefuseLeachingCombinedReactionBlock(ProcessBlock):
    pass


@declare_process_block_class(
    "CoalRefuseLeachingCombinedReactionBlock",
    block_class=_CoalRefuseLeachingCombinedReactionBlock,
)
class CoalRefuseLeachingCombinedReactionData(ProcessBlockData):
    default_scaler = LeachCombinedReactionScaler
    # Create Class ConfigBlock
    CONFIG = ProcessBlockData.CONFIG()
    CONFIG.declare(
        "parameters",
        ConfigValue(
            # TODO
            # domain=is_reaction_parameter_block,
            description="""A reference to an instance of the Heterogeneous Reaction Parameter
    Block associated with this property package.""",
        ),
    )

    def build(self):
        """
        Reaction block for combined shrinking core leaching of coal refuse in H2SO4.

        Implements the combined resistance rate expression:


        """
        super().build()

        add_object_reference(self, "_params", self.config.parameters)

        def rule_reaction_rate(b, r):
            l_block = b.parent_block().liquid[b.index()]
            s_block = b.parent_block().solid[b.index()]

            # Solid phase conversion for species r
            X = s_block.conversion_comp[r]

            # H+ concentration in mol/m³
            h_conc_m3 = units.convert(
                l_block.conc_mol_comp["H"], to_units=units.mol / units.m**3
            )

            # Dimensionless H+ concentration for R_rxn: strip mol/m³
            c_h = h_conc_m3 / (units.mol / units.m**3)

            # Volume fraction of solid in the solid inlet stream [-]
            # Read from parameter block (set by user based on feed preparation)
            phi_s_inlet = b.params.phi_s_inlet

            # Solid particle density [kg/L]
            rho_solid = units.convert(
                s_block.params.dens_mass, to_units=units.kg / units.litre
            )

            # Liquid density [kg/L]
            rho_liquid = b.params.rho_liquid

            # Solid volume fraction in the reactor [-] (distinct from phi_s_inlet)
            # Computed from actual stream flow rates in the contactor
            V_solid = s_block.flow_mass / s_block.params.dens_mass  # [L/hour]
            V_total = V_solid + l_block.flow_vol  # [L/hour]
            phi_s_reactor = V_solid / V_total

            rho_pulp = 1 / (
                (phi_s_reactor / rho_solid) + ((1 - phi_s_reactor) / rho_liquid)
            )

            # Liquid-to-solid volumetric ratio [L/kg]
            v_L_per_m_s = units.convert(
                l_block.flow_vol / s_block.flow_mass,
                to_units=units.litre / units.kg,
            )

            # Outer factor: R_p [m] * (1/rho_pulp [L/kg] + v_L/m_s [L/kg]) * rho_solid [kg/L]
            # Units: m * (L/kg) * (kg/L) = m
            outer_factor = b.params.R_p * (1 / rho_pulp + v_L_per_m_s) * rho_solid

            # --- Three series resistances, all in m²·hour/mol ---

            # 1) Film diffusion resistance [m²·hour/mol]:
            R_film = 1 / (b.params.K_film[r] * h_conc_m3)

            # 2) Ash/product layer diffusion resistance [m²·hour/mol]:
            R_ash = (
                (1 - (1 - X) ** (1 / 3))
                * b.params.R_p
                / (b.params.D_e[r] * h_conc_m3 * (1 - X) ** (1 / 3))
            )

            # 3) Surface chemical reaction resistance [m²·hour/mol]:
            R_rxn = 1 / (
                b.params.k_prime[r] * c_h ** b.params.A_ox[r] * (1 - X) ** (2 / 3)
            )

            # Combined rate: 3·phi_s / (outer[m] · resistance_sum[m²·hr/mol])
            # = mol/(m³·hr) → convert to mol/L/hr
            return units.convert(
                3 * phi_s_inlet / (outer_factor * (R_film + R_ash + R_rxn)),
                to_units=units.mol / units.litre / units.hour,
            )

        self.reaction_rate = Expression(
            self.params.reaction_idx, rule=rule_reaction_rate
        )

    @property
    def params(self):
        return self._params
