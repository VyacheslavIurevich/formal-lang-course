"""Task 6: Implement transformation of context-free grammar to weak normal form and reachability function with regular constraints for multiple start vertices, using Hellings' algorithm"""

from pyformlang.cfg import CFG, Production, Epsilon
import networkx as nx


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    """Transforms context-free grammar to weak normal form"""
    nf_cfg = cfg.to_normal_form()
    new_productions = nf_cfg.productions
    for nullable_symbol in cfg.get_nullable_symbols():
        new_production = Production(nullable_symbol, [Epsilon()])
        new_productions.add(new_production)
    return CFG(nf_cfg.variables, nf_cfg.terminals, nf_cfg.start_symbol, new_productions)


def hellings_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Reachability function, based on Hellings' algorithm"""
    if start_nodes is None:
        start_nodes = set(graph.nodes)
    if final_nodes is None:
        final_nodes = set(graph.nodes)
    cfg_weak = cfg_to_weak_normal_form(cfg)
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
    r = set()
    for eps_prod in epsilon_productions:
        N_i = eps_prod.head
        for v in graph.nodes:
            r.add((N_i, v, v))
    for terminal_prod in terminal_productions:
        N_i = terminal_prod.head
        terminal = terminal_prod.body[0].value
        for v, u, t in graph.edges(data="label"):
            if t == terminal:
                r.add((N_i, v, u))
    m = r.copy()
    while len(m):
        N_i, v, u = m.pop()
        new = set()
        for N_j, v_prime, v_2 in r:
            if v_2 != v:
                continue
            for pair_production in nonterminal_pair_productions:
                first = pair_production.body[0]
                second = pair_production.body[1]
                if not (first == N_j and second == N_i):
                    continue
                N_k = pair_production.head
                triple = (N_k, v_prime, u)
                if triple in r | new:
                    continue
                m.add(triple)
                new.add(triple)
        for N_j, u_2, v_prime in r:
            if u_2 != u:
                continue
            for pair_production in nonterminal_pair_productions:
                first = pair_production.body[0]
                second = pair_production.body[1]
                if not (first == N_i and second == N_j):
                    continue
                N_k = pair_production.head
                triple = (N_k, v, v_prime)
                if triple in r | new:
                    continue
                m.add(triple)
                new.add(triple)
        r.update(new)
    answer = set()
    start_symbol = cfg_weak.start_symbol
    for N, v, u in r:
        if N == start_symbol and v in start_nodes and u in final_nodes:
            answer.add((v, u))
    return answer
