"""Task 2: build DFA from Regex and build NFA from graph"""

from typing import Set
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
)
from pyformlang.regular_expression import Regex
from networkx import MultiDiGraph


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    """Given regex, builds DFA from it"""
    pfl_regex = Regex(regex)
    eps_nfa = pfl_regex.to_epsilon_nfa()
    return eps_nfa.to_deterministic()


def graph_to_nfa(
    graph: MultiDiGraph, start_states: Set[int], final_states: Set[int]
) -> NondeterministicFiniteAutomaton:
    """Given graph, builds NFA from it"""
    nfa = NondeterministicFiniteAutomaton.from_networkx(graph)
    if len(start_states) == 0:
        start_states = nfa.states
    if len(final_states) == 0:
        final_states = nfa.states
    for start_state in start_states:
        nfa.add_start_state(start_state)
    for final_state in final_states:
        nfa.add_final_state(final_state)
    return nfa
