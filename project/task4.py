"""Task 4: Implement reachability function with regular constraints for multiple start vertices"""

from enum import Enum
from typing import Set
from scipy import sparse
from networkx import MultiDiGraph
from project.task2 import regex_to_dfa, graph_to_nfa
from project.task3 import AdjacencyMatrixFA, MatrixType, get_sparse_matrix


class StackType(Enum):
    """Type of sparse stack from SciPy"""

    VSTACK = "vstack"
    HSTACK = "hstack"


STACK_INITIALIZERS = {
    StackType.VSTACK: lambda blocks: sparse.vstack(blocks),
    StackType.HSTACK: lambda blocks: sparse.hstack(blocks),
}


def get_sparse_stack(blocks: list, stack_type: StackType):
    """Initializes sparse matrix of specified type"""
    return STACK_INITIALIZERS[stack_type](blocks)


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: Set[int],
    final_nodes: Set[int],
    matrix_type: MatrixType = MatrixType.CSR,
    stack_type: StackType = StackType.VSTACK,
) -> Set[tuple[int, int]]:
    """Reachabilty function, based on multiple source BFS"""
    regex_adj = AdjacencyMatrixFA(regex_to_dfa(regex), matrix_type)
    if start_nodes is None:
        start_nodes = set()
    if final_nodes is None:
        final_nodes = set()
    graph_adj = AdjacencyMatrixFA(
        graph_to_nfa(graph, start_nodes, final_nodes), matrix_type
    )
    front_blocks = []
    graph_state_to_idx = graph_adj.state_to_idx
    regex_state_to_idx = regex_adj.state_to_idx
    symbols_intersection = (
        graph_adj.boolean_decompositions.keys()
        & regex_adj.boolean_decompositions.keys()
    )
    for graph_start_state in graph_adj.start_states:
        graph_start_state_idx = graph_state_to_idx[graph_start_state]
        front_block = get_sparse_matrix(
            graph_adj.states_len, regex_adj.states_len, matrix_type
        )
        for regex_start_state in regex_adj.start_states:
            regex_start_state_idx = regex_state_to_idx[regex_start_state]
            front_block[graph_start_state_idx, regex_start_state_idx] = True
        front_blocks.append(front_block)
    front = get_sparse_stack(front_blocks, stack_type)
    visited = front.copy()
    graph_boolean_decompositons = graph_adj.boolean_decompositions
    regex_boolean_decompositons = regex_adj.boolean_decompositions
    transposed_graph_boolean_decompositions = {
        symbol: decomposition.transpose().tocsr()
        for symbol, decomposition in graph_boolean_decompositons.items()
    }
    while front.nnz:
        new_front_blocks = {}
        for symbol in symbols_intersection:
            this_symbol_blocks = []
            for i in range(len(graph_adj.start_states)):
                current_block = front[
                    i * graph_adj.states_len : (i + 1) * graph_adj.states_len
                ]
                step = (
                    transposed_graph_boolean_decompositions[symbol]
                    @ current_block
                    @ regex_boolean_decompositons[symbol]
                )
                this_symbol_blocks.append(step)
            new_front_blocks[symbol] = get_sparse_stack(this_symbol_blocks, stack_type)
        front = sum(new_front_blocks.values()) > visited
        visited = visited + front
    result_pairs = set()
    for i, graph_start_state in enumerate(graph_adj.start_states):
        result_block = visited[
            i * graph_adj.states_len : (i + 1) * graph_adj.states_len
        ]
        for graph_final_state in graph_adj.final_states:
            graph_final_idx = graph_state_to_idx[graph_final_state]
            for regex_final_state in regex_adj.final_states:
                regex_final_idx = regex_state_to_idx[regex_final_state]
                if result_block[graph_final_idx, regex_final_idx]:
                    result_pairs.add((graph_start_state.value, graph_final_state.value))
    return result_pairs
