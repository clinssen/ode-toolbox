#
# test_cse_analytic_solver_integration.py
#
# This file is part of the NEST ODE toolbox.
#
# Copyright (C) 2017 The NEST Initiative
#
# The NEST ODE toolbox is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 2 of
# the License, or (at your option) any later version.
#
# The NEST ODE toolbox is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

# test_cse_integrators.py
#
# This file is part of the NEST ODE toolbox.
#

import copy
import logging
import numpy as np
import pytest
import sympy
import odetoolbox
from odetoolbox.analytic_integrator import (AnalyticIntegrator)
from odetoolbox.mixed_integrator import (MixedIntegrator)
from tests.test_utils import load_test_json

try:
    import pygsl.odeiv as odeiv
    PYGSL_AVAILABLE = True

except ImportError:
    PYGSL_AVAILABLE = False


class TestCSENumericalSolver:
    """
    Isolated ODE-toolbox validation of CSE applied to a numerical solver.
    """

    def test_cse_analytic_integrator_matches_baseline():
        """
        Verify that analytical CSE does not change the trajectory produced
        by ODE-toolbox's AnalyticIntegrator.
        """

        indict = load_test_json(
            "cse_analytical.json"
        )

        #
        # Analyse exactly the same model twice:
        #     1. normal ODE-toolbox
        #     2. ODE-toolbox with CSE
        #
        baseline_solvers, _, _ = odetoolbox._analysis(
            copy.deepcopy(indict),
            disable_stiffness_check=True,
            disable_singularity_detection=True,
            enable_cse=False,
            log_level=logging.DEBUG,
        )

        cse_solvers, _, _ = odetoolbox._analysis(
            copy.deepcopy(indict),
            disable_stiffness_check=True,
            disable_singularity_detection=True,
            enable_cse=True,
            log_level=logging.DEBUG,
        )

        assert len(baseline_solvers) == 1
        assert len(cse_solvers) == 1

        baseline_solver = baseline_solvers[0]
        cse_solver = cse_solvers[0]

        assert baseline_solver["solver"] == "analytical"
        assert cse_solver["solver"] == "analytical"

        #
        # Prove that this genuinely exercised CSE.
        #
        assert "cse" not in baseline_solver
        assert "cse" in cse_solver
        assert "propagators" in cse_solver["cse"]

        #
        # Make parameter values explicitly available to AnalyticIntegrator.
        #
        baseline_solver.setdefault(
            "parameters",
            {}
        )
        cse_solver.setdefault(
            "parameters",
            {}
        )

        baseline_solver["parameters"].update(
            indict.get("parameters", {})
        )
        cse_solver["parameters"].update(
            indict.get("parameters", {})
        )

        #
        # Run through the REAL existing analytical integrator.
        #
        baseline_integrator = AnalyticIntegrator(
            baseline_solver,
            enable_cse=False,
        )

        cse_integrator = AnalyticIntegrator(
            cse_solver,
            enable_cse=True,
        )

        #
        # Compare the trajectory at many time points.
        #
        time_points = np.linspace(
            0.0,
            20.0,
            101,
        )

        for t in time_points:

            baseline_state = (
                baseline_integrator.get_value(t)
            )

            cse_state = (
                cse_integrator.get_value(t)
            )

            assert (
                baseline_state.keys()
                ==
                cse_state.keys()
            )

            for symbol in baseline_state:

                np.testing.assert_allclose(
                    cse_state[symbol],
                    baseline_state[symbol],
                    rtol=1E-10,
                    atol=1E-12,
                )

        print(
            "\nAnalytical baseline final:",
            baseline_state,
        )

        print(
            "Analytical CSE final:",
            cse_state,
        )

