"""Task 3: Create AdjacencyMatrixFA class with interpeting functions,
implement automata intersection and RPQ for graphs"""

from typing import Set, Iterable, Tuple, Dict, Int
from scipy import sparse
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    Symbol,
    State,
)
from networkx import MultiDiGraph
# from project.task2 import regex_to_dfa, graph_to_nfa


class AdjacencyMatrixFA:
    """Finite automaton represented via sparse adjacency matrix boolean decompositions"""

    boolean_decompositions: Dict[Symbol, sparse.csr_array]
    states: Set[State]
    start_states: Set[State]
    final_states: Set[State]
    state_to_idx: Dict[State, Int]

    def __init__(self, nfa: NondeterministicFiniteAutomaton):
        self.start_states = nfa.start_states
        self.final_states = nfa.final_states
        self.states = nfa.states
        self.state_to_idx = {state: idx for (idx, state) in enumerate(nfa.states)}
        symbols = nfa.symbols
        states_len = len(self.states)
        nfa_dict = nfa.to_dict()
        for symbol in symbols:
            self.boolean_decompositions[symbol] = sparse.csr_array(
                (states_len, states_len), dtype=bool
            )
            for state in self.states:
                next_states = nfa_dict[state][symbol]
                set_next_states = next_states
                if isinstance(next_states, State):
                    set_next_states = set(next_states)
                first_idx = self.state_to_idx[state]
                for next_state in set_next_states:
                    next_idx = self.state_to_idx[next_state]
                    self.boolean_decompositions[symbol][first_idx][next_idx] = True

    def accepts(self, word: Iterable[Symbol]) -> bool:
        """Returns true if automaton accepts word, else false"""
        curr_states = self.start_states
        for symbol in word:
            next_states = set()
            for first_state in self.states:
                for second_state in self.states:
                    first_state_idx = self.state_to_idx[first_state]
                    second_state_idx = self.state_to_idx[second_state]
                    if self.boolean_decompositions[symbol][first_state_idx][
                        second_state_idx
                    ]:
                        next_states.add(second_state)
            curr_states = next_states
        return curr_states & self.final_states != set()

    def is_empty(self) -> bool:
        """Returns true if the automaton's language is empty, else false"""
        return False


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    """Returns automata intersection"""
    return None


def tensor_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> Set[Tuple[int, int]]:
    """Returns all pairs of nodes, connected by a path,
    that forms a word from the language specified by a regex"""
    return set()
