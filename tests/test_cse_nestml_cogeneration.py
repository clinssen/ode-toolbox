#
# test_cse_nestml_cogeneration.py
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
#


import pytest
import sympy 
import odetoolbox.expression_optimisation as eo 
from .context import odetoolbox


class TestCSECodeGeneration:

    r"""This script provides a list of functions which test the NESTML cogeneration pipeline when common subexpression elimination is applied. """

    # do we need to have any ensurances? 
    # @pytest.mark.parametrize("use_alternative_expM", [True, False])
    # @pytest.mark.parametrize("tau_1, tau_2", [(10., 2.), (10., 10.)])
         
    def test_cse_no_common_expression(self):

        """"
        This script provides a test for an analytical solver cse, utilising the generate_propagator_solver()
        """

        x, y, z = sympy.symbols("x y z", real=True)

        expressions = { 
            "a": sympy.exp(x * y + z), 
            "b": x + y,
        }

        # Explicitly testing the False behavior
        replacement, reduced = eo._run_profitable_cse(expressions, "__test_", solver_name="test_solver", region_name="no_reuse") 
    
        assert replacement == []
        assert reduced == expressions


    def test_cse_preserves_expression():

        """"
        This script provides tests that cse reduces the number mathematical operations from the original eq. 
        """

        x, y, a, b = sympy.symbols( # define sympy symbols 
            "x y a b",
            real=True
        )

        expressions = {
            "eq1": (x + y) * a,  # unoptimised eq
            "eq2": (x + y) * b,
        }

        before = count_operations(expressions.values()) # count number of mathematical operations for original eq

        replacements, reduced = (common_subexpression_elimination(expressions)) # perform cse 

        after = count_cse_operations(replacements, reduced) # count number of mathematical operations for cse eq 

        # Print blocks for manual validation (visible pytest -s)
        print(f"\n Operations Before CSE: {before}")
        print(f" Operations After CSE: {after}")

        # enforce logic that optimisation must decrease operations 
        assert after < before, f"Optimisation failed, cost did not decrease. Before: {before}. After: {after}"


    def test_cse_preserves_condition_solver():

        """
        In this test we will see how the cse solver reacts to a condition in the solver
        """


        indict = { # defining indict non-linear that will need a singularity condition 
            "dynamics": [
                {
                    "expression": "V_m' = -V_m / tau_m + I/C_m",
                    "initial_value": "0"
                }
            ],
            "parameters": {
                "tau_m": "10",
                "C_m": "250"
            },
            "options": {
                "output_timestep_symbol": "__h"
            }
        }


        solver = odetoolbox.analysis(indict)  # baseline, no CSE
        
        # TO DO FIX THE SINGULARITY DETECTION, this has been defaulted to off for the timebeing 
        result = odetoolbox.analysis(indict, enable_cse=True) # apply cse 

        
        print(json.dumps(result, indent=2)) # print result output
        
        assert result[0]["conditions"] == solver[0]["conditions"] # ensure singularity conditions are present in both 

        propagators = result[0]["conditions"]["default"]["propagators"]
        update_exprs = result[0]["conditions"]["default"]["update_expressions"]
        all_text = " ".join(list(propagators.values()) + list(update_exprs.values()))
        assert "cse" not in all_text  # or whatever the actual generated symbol prefix is
        
        

    def test_cse_preserves_expression():

        """"
        This script provides a test of whether the original equation and CSE-optimised equation is preserved (mathematically==0). 
        """


        x, y, a, b = sympy.symbols( # define sympy symbols 
            "x y a b",
            real=True
        )

        expressions = {
            "eq1": (x + y) * a,  # unoptimised eq
            "eq2": (x + y) * b,
        }

        replacements, reduced = (common_subexpression_elimination(expressions)) # perform cse 

        for name in expressions: # looping over per eq

            restored = restore_cse_expression(
                reduced[name],
                replacements
            )

            # simplify() is a SymPy function that rewrites mathematical expressions into a shorter or simpler form.
            difference = sympy.simplify(restored - expressions[name])

            # The text after the comma only shows up if difference != 0
            assert difference == 0, f"CSE failed. The restored expression differed from the original by: {difference}"




    def test_condition_branches_cse_independently():


        """
        This test is a test of singularity, and branches of independent singularities having 
        independent cse expressions and there is not leakage. 
        """

        x, y, a, b = sympy.symbols(
            "x y a b",
            real=True,
        )

        # Construct a mock solver profile containing a default region and a singularity region.
        # Note that both regions reuse the exact same subexpressions (x + y) and (x - y).
        solver = {
            "solver": "analytical",

            "conditions": {

                "default": {
                    "update_expressions": {"x": a * (x + y), "y": b * (x + y)}},

                "(tau_1 == tau_2)": {
                    "update_expressions": {"x": a * (x - y), "y": b * (x - y)}},
            }
        }

        # execute region cse optimisation pass 
        result = eo.apply_cse_to_solver(solver, optimise_condition_branches=True)

        # Isolate the processed output regions
        default_branch = (result["conditions"]["default"]) # default solver with no known singularities 
        singularity_branch = (result["conditions"]["(tau_1 == tau_2)"]) # singularity branch from l^hopital 

        print(f"Default Branch::: {default_branch}")
        print(f"Singularity Branch::: {singularity_branch}")

        # test cse actually triggered and extracted terms in these branches
        assert "cse" in default_branch
        assert "cse" in singularity_branch

        # extract [(temporary_var, original_expression)] from default solver; update_expression  
        default_replacements = (
            default_branch["cse"]
            ["update_expressions"]
        )

        # extract [(temporary_var, original_expression)] from singularity branch; update_expression  
        singular_replacements = (
            singularity_branch["cse"]
            ["update_expressions"]
        )

        assert default_replacements # do they exist? 
        assert singular_replacements

        # Collect only the left-hand names of the replacements (the new variable names)
        default_symbols = {temporary for temporary, _ in default_replacements}
        singular_symbols = {temporary for temporary, _ in singular_replacements}

        # No variable name generated in the default branch exists in the singularity branch
        assert default_symbols.isdisjoint(singular_symbols) # if this fails c++ will overwrite these variables and cause conflicts 

        for condition, original_branch in (
            
            solver["conditions"].items()):

            optimized_branch = (result["conditions"][condition])

            replacements = (optimized_branch.get("cse", {}).get("update_expressions",[]))

            for variable, original in (
                
                original_branch["update_expressions"].items()):

                reduced = (optimized_branch["update_expressions"][variable])
                restored = restore_cse_expression(reduced,replacements) # restore the cse equations back to the original form 

                assert sympy.simplify(restored - original) == 0 # ensure the original equations equal 



    def test_analytical_solver_cse():

        """"
        This script provides a test for an analytical solver cse, utilising the generate_propagator_solver()
        """

        # This defines the standard physical time step parameter (__h) and an exponential decay time constant (tau).
        h, tau = sympy.symbols("__h tau", real=True) 
        common_propagator = sympy.exp(-h / tau) # common propagator found across neuronal models 

        P_x = sympy.Symbol("__P__x__x", real=True) # defining propagators 
        P_y = sympy.Symbol("__P__y__y", real=True)

        solver_dict = { # mocking an analytical matrix layout 
            "solver": "analytical",
            "state_variables": [
                "x",
                "y",
            ],

            "propagators": {
                "__P__x__x": common_propagator, # defining the RHS as the common propagator 
                "__P__y__y": common_propagator,
            },

            "update_expressions": {
                "x": P_x * sympy.Symbol("x", real=True), # x,y real sympy objects
                "y": P_y * sympy.Symbol("y", real=True),
            },
        }

        original_propagators = dict(solver_dict["propagators"]) # saving benchmark 

        result = eo.apply_cse_to_solver(solver_dict) # apply function 

        
        assert result["solver"] == "analytical", "Failed: The solver type mutated or was lost."
        
        assert "propagators" in result["cse"]

        replacements = result["cse"]["propagators"]

        assert len(replacements) > 0 # checking optimisation occured

        # prove equilance to original eq
        for variable, original in original_propagators.items(): # very similar code to numeric test 
            
            reduced = result["propagators"][variable] # Used singular 'variable'

            # Convert the reduced string expression back to a SymPy object for testing
            restored = restore_cse_expression(reduced, replacements) 

            assert sympy.simplify(restored - original) == 0


    def test_numeric_solver_cse():

        """"
        This script provides a test for numeric solver cse, can it optimise the RHS correctly,
        update expressions, preserve its structure and mathematics? 
        """

        # define x,y as real sympy objects 
        x,y = sympy.symbols("x y", real=True)

        common_term = sympy.exp(x + y) # define common term as exp^x+y sympy object 

        # mocking solver data structure, in the expected format for sympy 
        solver_dict = {
            "solver": "numeric",
            "state_variables": ["x", "y"],
            "update_expressions": {
                "x": x + common_term,
                "y": y + 2 * common_term,
            },
        }

        original_expressions = dict(solver_dict["update_expressions"]) # original update expressions 

        result = eo.apply_cse_to_solver(solver_dict) # apply function 
        
        assert result["solver"] == "numeric", "Failed: The solver type mutated or was lost."
        assert "cse" in result, "Failed: 'cse' key dictionary was never initialised."
        assert "update_expressions" in result["cse"], ("Failed: 'update_expressions' was not processed or missing inside the inner cse tracker.")

        replacements = result["cse"]["update_expressions"]

        assert len(replacements) > 0 # checking optimisation occured

        # check cse didnt remove eq
        assert set(result["update_expressions"]) == {"x", "y"}

        # prove equilance to original eq
        for variable, original in original_expressions.items():
            
            reduced = result["update_expressions"][variable] # Used singular 'variable'

            # Convert the reduced string expression back to a SymPy object for testing
            restored = restore_cse_expression(reduced, replacements) 

            assert sympy.simplify(restored - original) == 0

    def test_cse_disabled_preserves_legacy():

        """
        Existing users upstream should notice no difference in their output. 
        """
        
        # mocking a .json from a model description file 
        indict = {
            "dynamics": [ 
                {
                    "expression":
                        "x' = -x / tau", 
                        "initial_value":
                            "1",
                }   
            ]    
        }

        default_result = odetoolbox.analysis( # default cse is off by default 
            indict,
            disable_stiffness_check=True,
            disable_singularity_detection=True, 
        )

        explicitly_disabled_result = odetoolbox.analysis( # explicitly false cse 
            indict,
            disable_stiffness_check=True,
            disable_singularity_detection=True,
            enable_cse=False
        )

        assert(default_result == explicitly_disabled_result)


    def test_cse_machine_aware_rejected(self):

        """
        Tests that adding ops vs exponential ops have different values and therefore can get rejected / accepted based on this. 
        """

        # low-cost reuse 
        x, y, z = sympy.symbols("x y z",real=True)

        # low-cost reuse 
        expressions = {
            "a": x + y,  
            "b": x + y + z,
        }

        replacements, reduced = eo._run_profitable_cse(
            expressions,
            symbol_prefix="__test_cse_",
            solver_name ="test_solver",
            region_name="cheap_reuse")

        assert replacements == []
        assert reduced == expressions


    def test_cse_machine_aware_accepted(self):

        """
        Tests that adding ops vs exponential ops have different values and therefore can get rejected / accepted based on this. 
        """

        # high-cost reuse 
        x, y, z = sympy.symbols("x y z",real=True)

        common = sympy.exp(x * y + z)


        expressions = { 
            "a": common + x, 
            "b": common + y,
            "c": common + z,
            "d": common + x,
        }

        replacements, reduced = eo._run_profitable_cse(
            expressions,
            symbol_prefix="__test_cse_",
            solver_name ="test_solver",
            region_name="expensive_reuse")

        assert replacements
        assert reduced != expressions


    def test_machine_aware_cost_simple_operations(self):
        """
        Tests to check that the expression weighting is being applied correctly
        """

        x, y = sympy.symbols("x y")

        assert eo.weighted_expression_cost(x) == 0.0
        assert eo.weighted_expression_cost(x + y) == 1.0 
        assert eo.weighted_expression_cost(x * y) == 1.0 

        add_cost = eo.weighted_expression_cost(x + y) 
        exp_cost = eo.weighted_expression_cost(sympy.exp(x + y))

        assert exp_cost > add_cost
        assert exp_cost == 9.0 # with current weighted params. 


    def test_machine_aware_division_costs(self):

        """
        Further tests to check that the expression weighting is being applied correctly. these tests will need to be adapted
        if the parameters are adjusted.
        """


        x,y = sympy.symbols("x y")
        multiply_cost = eo.weighted_expression_cost(x * y)
        division_cost = eo.weighted_expression_cost(x / y)

        print(f"multi cost: {multiply_cost}, division cost: {division_cost}") # multi 1.0 division 5.0 
        assert division_cost > multiply_cost 