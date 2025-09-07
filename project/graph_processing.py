"""Task 1: collect graph info & save built 2-cycle graph into DOT file"""

from typing import Set, Tuple
import networkx
import cfpq_data


class GraphInfo:
    """Contains graph's number of nodes, number of edges and set of labels"""

    def __init__(self, num_nodes: int, num_edges: int, labels: Set[any]):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.labels = labels


def get_graph_info(graph_name: str) -> GraphInfo:
    """Takes graph's name as an argument and returns GraphInfo class with its characteristics"""
    graph_path = cfpq_data.download(graph_name)
    graph = cfpq_data.graph_from_csv(graph_path)
    num_of_nodes = graph.number_of_nodes()
    num_of_edges = graph.number_of_edges()
    labels = set(cfpq_data.get_sorted_labels(graph))
    return GraphInfo(num_of_nodes, num_of_edges, labels)


def create_dot_two_cycles_graph(
    cycle_num_nodes: Tuple[int, int], labels_names: Tuple[str, str], path_to_save: str
):
    """Builds 2-cycle graph with given numbers of nodes in each cycle,
    their labels' names and saves it into DOT file with given path"""
    graph = cfpq_data.labeled_two_cycles_graph(
        cycle_num_nodes[0], cycle_num_nodes[1], labels=labels_names
    )
    networkx.nx_pydot.write_dot(graph, path_to_save)
