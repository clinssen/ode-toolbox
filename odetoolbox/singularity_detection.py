#
# singularity_detection.py
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
from typing import List, Mapping, Tuple

import itertools
import logging
import sympy
import sympy.parsing.sympy_parser


class SingularityDetection:
    r"""
    This class ...
    """

    @staticmethod
    def _is_matrix_defined_under_substitution(A: sympy.Matrix, cond: Mapping) -> bool:
        r"""
        Function to check if a matrix is defined (i.e. does not contain NaN or infinity) after we perform a given set of subsitutions.

        Parameters
        ----------
        A : sympy.Matrix
            input matrix
        cond : Mapping
            mapping from expression that is to be subsituted, to expression to put in its place
        """
        for val in sympy.flatten(A):
            for expr, subs_expr in cond.items():
                if sympy.simplify(val.subs(expr, subs_expr)) in [sympy.nan, sympy.zoo, sympy.oo]:
                    return False
        return True

    @staticmethod
    def _new_system_matrix(A: sympy.Matrix, cond):  # cond is a tuple here (or a tuple of tuples!)
        M = sympy.zeros(3, 3)
        for i in range(3):
            for j in range(3):
                M[i, j] = A[i, j]
                for k in range(len(cond)):  # looping over the tuple
                    M[i, j] = M[i, j].subs(list(cond[k].keys())[0], list(cond[k].values())[0])  # XXX:add check for simplification threshold(later)
        return M

    @staticmethod
    def _flatten_conditions(cond):
        """
        Return a list with conditions in the form of dictionaries
        """
        lst = []
        for i in range(len(cond)):
            if cond[i] not in lst:
                lst.append(cond[i])
        return lst

    @staticmethod
    def _combinations_of_conditions(cond):  # cond is a list
        comb_list = []
        for r in range(len(cond) + 1):
            comb = itertools.combinations(cond, r)
            comb_list = comb_list + list(comb)
        return comb_list

    @staticmethod
    def _filter_valid_conditions(cond, A: sympy.Matrix):
        filt_cond = []
        for i in range(len(cond)):  # looping over conditions
            if SingularityDetection._is_matrix_defined_under_substitution(A, cond[i]):
                filt_cond.append(cond[i])
        return filt_cond

    @staticmethod
    def _generate_singularity_conditions(A: sympy.Matrix):
        r"""
        The function solve returns a list where each element is a dictionary. And each dictionary entry (condition: expression) corresponds to a condition at which that expression goes to zero.
        If the expression is quadratic, like let's say "x**2-1" then the function 'solve() returns two dictionaries in a list. each dictionary corresponds to one solution.
        We are then collecting these lists in our own list called 'condition'.
        """

        condition = []
        for expr in sympy.flatten(A):
            for subexpr in sympy.preorder_traversal(expr):  # traversing through the tree
                if isinstance(subexpr, sympy.Pow) and subexpr.args[1] < 0:  # find expressions of the form 1/x, which is encoded in sympy as x^-1
                    denom = subexpr.args[0]  # extracting the denominator
                    sol = sympy.solve(denom, denom.free_symbols, dict=True)  # 'condition' here is a list of all those conditions at which the denominator goes to zero
                    if sol not in condition:
                        condition.extend(sol)

        return condition

    @staticmethod
    def find_singularities(P: sympy.Matrix, A: sympy.Matrix):
        r"""Find singularities in the propagator matrix :math:`P` given the system matrix :math:`A`.

        Parameters
        ----------
        P : sympy.Matrix
            propagator matrix to check for singularities
        A : sympy.Matrix
            system matrix
        """
        condition = SingularityDetection._generate_singularity_conditions(P)
        condition = SingularityDetection._flatten_conditions(condition)  # makes a list of conditions with each condition in the form of a dict
        condition = SingularityDetection._filter_valid_conditions(condition, A)  # filters out the invalid conditions (invalid means those for which A is not defined)
        return condition
