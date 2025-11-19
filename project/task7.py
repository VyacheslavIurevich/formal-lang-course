"""Task 7: Implement reachability function with regular constraints for multiple start vertices, using matrix-based algorithm"""

from pyformlang.cfg import CFG
import networkx as nx
from scipy import sparse
from project.task6 import cfg_to_weak_normal_form


def matrix_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Reachability function, based on linear algebra algorithm"""
    n = len(graph.nodes)
    if start_nodes is None:
        start_nodes = set(graph.nodes)
    if final_nodes is None:
        final_nodes = set(graph.nodes)
    cfg_weak = cfg_to_weak_normal_form(cfg)
    non_terminals = cfg_weak.variables
    node_to_idx = {node: idx for idx, node in enumerate(graph.nodes)}
    epsilon_productions = set()  # nonterminal to epsilon
    terminal_productions = set()  # nonterminal to terminal
    nonterminal_pair_productions = set()  # nonterminal to pair of nonterminals
    for production in cfg_weak.productions:
        match len(production.body):
            case 0:
                epsilon_productions.add(production)
            case 1:
                terminal_productions.add(production)
            case _:
                nonterminal_pair_productions.add(production)
    boolean_decompositions = {
        N: sparse.csr_array((n, n), dtype=bool) for N in non_terminals
    }
    for eps_prod in epsilon_productions:
        N = eps_prod.head
        boolean_decompositions[N].setdiag(True)
    for terminal_prod in terminal_productions:
        N = terminal_prod.head
        terminal = terminal_prod.body[0].value
        for v, u, t in graph.edges(data="label"):
            fst = node_to_idx[v]
            snd = node_to_idx[u]
            if t == terminal:
                boolean_decompositions[N][fst, snd] = True
    upd = True
    while upd:
        upd = False
        for pair_production in nonterminal_pair_productions:
            B = pair_production.body[0].value
            C = pair_production.body[1].value
            N = pair_production.head
            prod = boolean_decompositions[B] @ boolean_decompositions[C]
            for u, v in zip(*prod.nonzero()):
                if boolean_decompositions[N][u, v]:
                    continue
                boolean_decompositions[N][u, v] = True
                upd = True
    answer = set()
    start_symbol = cfg_weak.start_symbol
    for start in start_nodes:
        for finish in final_nodes:
            fst = node_to_idx[start]
            snd = node_to_idx[finish]
            if boolean_decompositions[start_symbol][fst, snd]:
                answer.add((start, finish))
    return answer
