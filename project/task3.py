"""Task 3: Create AdjacencyMatrixFA class with interpeting functions,
implement automata intersection and RPQ for graphs"""

from enum import Enum
from typing import Set, Iterable, Tuple, Dict
from scipy import sparse
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    Symbol,
    State,
)
from networkx import MultiDiGraph
from project.task2 import regex_to_dfa, graph_to_nfa


class MatrixType(Enum):
    """Type of sparse matrix from SciPy"""

    BSR = "bsr"
    COO = "coo"
    CSC = "csc"
    CSR = "csr"
    DIA = "dia"
    DOK = "dok"


MATRIX_CONVERTERS = {
    MatrixType.BSR: lambda m: m.tobsr(),
    MatrixType.COO: lambda m: m.tocoo(),
    MatrixType.CSC: lambda m: m.tocsc(),
    MatrixType.CSR: lambda m: m.tocsr(),
    MatrixType.DIA: lambda m: m.todia(),
    MatrixType.DOK: lambda m: m.todok(),
}


def convert_matrix(matrix: sparse.spmatrix, matrix_type: MatrixType):
    """Converts sparse matrix to a chosen type"""
    return MATRIX_CONVERTERS.get(matrix_type, lambda m: m)(matrix)


class AdjacencyMatrixFA:
    """Finite automaton represented via sparse adjacency matrix boolean decompositions"""

    boolean_decompositions: Dict[Symbol, sparse.spmatrix]
    start_states: Set[State]
    final_states: Set[State]
    states_len: int
    state_to_idx: Dict[State, int]
    matrix_type: MatrixType

    def __init__(
        self,
        nfa: NondeterministicFiniteAutomaton,
        matrix_type: MatrixType = MatrixType.CSR,
    ):
        self.boolean_decompositions = {}
        self.matrix_type = matrix_type
        if nfa is None:
            self.start_states = set()
            self.final_states = set()
            self.states_len = 0
            self.state_to_idx = {}
            return
        self.start_states = nfa.start_states
        self.final_states = nfa.final_states
        states = nfa.states
        self.state_to_idx = {state: idx for (idx, state) in enumerate(states)}
        self.states_len = len(states)
        nfa_dict = nfa.to_dict()
        for symbol in nfa.symbols:
            self.boolean_decompositions[symbol] = sparse.csr_array(
                (self.states_len, self.states_len), dtype=bool
            )
            convert_matrix(self.boolean_decompositions[symbol], matrix_type)
            for state in states:
                transitions = nfa_dict.get(state)
                if transitions is None or symbol not in transitions.keys():
                    continue
                next_states = transitions[symbol]
                set_next_states = next_states
                if isinstance(next_states, State):
                    set_next_states = {next_states}
                first_idx = self.state_to_idx[state]
                for next_state in set_next_states:
                    next_idx = self.state_to_idx[next_state]
                    self.boolean_decompositions[symbol][first_idx, next_idx] = True

    def accepts(self, word: Iterable[Symbol]) -> bool:
        """Returns true if automaton accepts word, else false"""
        curr_states = self.start_states
        states = self.state_to_idx.keys()
        for symbol in word:
            next_states = set()
            for first_state in states:
                for second_state in states:
                    first_state_idx = self.state_to_idx[first_state]
                    second_state_idx = self.state_to_idx[second_state]
                    if self.boolean_decompositions[symbol][
                        first_state_idx, second_state_idx
                    ]:
                        next_states.add(second_state)
            curr_states = next_states
        return (curr_states & self.final_states) != set()

    def get_transitive_closure(self) -> sparse.spmatrix:
        """Returns transitive closure of this adjancency matrix"""
        transitive_closure = sparse.csr_array(
            (self.states_len, self.states_len), dtype=bool
        )
        convert_matrix(transitive_closure, self.matrix_type)
        transitive_closure.setdiag(True)
        for decomposition in self.boolean_decompositions.values():
            transitive_closure += decomposition
        for _ in range(self.states_len):
            transitive_closure = transitive_closure @ transitive_closure
        transitive_closure.astype(bool)
        return transitive_closure

    def is_empty(self) -> bool:
        """Returns true if the automaton's language is empty, else false"""
        transitive_closure = self.get_transitive_closure()
        for start_state in self.start_states:
            start_state_idx = self.state_to_idx[start_state]
            for final_state in self.final_states:
                final_state_idx = self.state_to_idx[final_state]
                if transitive_closure[start_state_idx, final_state_idx]:
                    return False
        return True


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    """Returns automata intersection"""
    result = AdjacencyMatrixFA(None)
    result.states_len = automaton1.states_len * automaton2.states_len
    first_boolean_decomps = automaton1.boolean_decompositions
    second_boolean_decomps = automaton2.boolean_decompositions
    for first_state, first_state_idx in automaton1.state_to_idx.items():
        for second_state, second_state_idx in automaton2.state_to_idx.items():
            result_state = State((first_state, second_state))
            result.state_to_idx[result_state] = (
                first_state_idx * automaton2.states_len + second_state_idx
            )
            if (
                first_state in automaton1.start_states
                and second_state in automaton2.start_states
            ):
                result.start_states.add(result_state)
            if (
                first_state in automaton1.final_states
                and second_state in automaton2.final_states
            ):
                result.final_states.add(result_state)
    result_symbols = set(first_boolean_decomps.keys()) & set(
        second_boolean_decomps.keys()
    )
    result.boolean_decompositions = {
        symbol: sparse.kron(
            first_boolean_decomps[symbol], second_boolean_decomps[symbol], format="csr"
        )
        for symbol in result_symbols
    }
    return result


def tensor_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> Set[Tuple[int, int]]:
    """Returns all pairs of nodes, connected by a path,
    that forms a word from the language specified by a regex"""
    regex_adj = AdjacencyMatrixFA(regex_to_dfa(regex))
    if start_nodes is None:
        start_nodes = set()
    if final_nodes is None:
        final_nodes = set()
    graph_adj = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    intersection_adj = intersect_automata(regex_adj, graph_adj)
    transitive_closure_adj = intersection_adj.get_transitive_closure()
    pairs = set()
    for start_state in intersection_adj.start_states:
        start_state_idx = intersection_adj.state_to_idx[start_state]
        for final_state in intersection_adj.final_states:
            final_state_idx = intersection_adj.state_to_idx[final_state]
            if transitive_closure_adj[start_state_idx, final_state_idx]:
                _, start = start_state.value
                _, final = final_state.value
                pairs.add((start.value, final.value))
    return pairs
