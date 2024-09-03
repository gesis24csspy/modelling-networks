import logging
from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import graph_tool.all as gt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# import networkx as nx
import numpy as np
import pandas as pd
from graph_tool.all import Graph, GraphView


def save_gt(g: Graph, filename: str) -> None:
    g.save(f"../output/{filename}.gt")


def load_gt(filename: str) -> Graph:
    return gt.load_graph(f"../input/{filename}.gt")


# CONSTRUCT NETWORK FUNCTIONS #


def construct_cooccurrence_edgelist(
    df: pd.DataFrame, node_list_col: str, context_group_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a co-occurrence / co-mention network of nodes in node_list_col
    Defaults to co-occurrences within the row units (e.g., videos, comments)
    If you provide context_group_col, it will create ties within the expanded context
    """
    results: List[Dict[str, Any]] = []

    if context_group_col:
        # group by context column and process each group separately
        grouped = df.groupby(context_group_col)
        for gid, group in grouped:
            # flatten the lists of nodes within each group
            all_nodes = [node for sublist in group[node_list_col] for node in sublist]
            unique_nodes = set(all_nodes)  # Ensure unique nodes

            for node1, node2 in combinations(unique_nodes, 2):
                # undirected graph, so sort the nodes to get the counts right
                if node1 > node2:
                    node1, node2 = node2, node1
                results.append({"i": node1, "j": node2})
                # results.append({"i": node1, "j": node2, "context": gid})
    else:
        # process co-occurrences within row units (e.g., videos, comments)
        for _, row in df.iterrows():  # vectorize this later...
            nodes = row[node_list_col]
            unique_nodes = set(nodes)  # Ensure unique nodes

            for node1, node2 in combinations(unique_nodes, 2):
                # undirected graph, so sort the nodes to get the counts right
                if node1 > node2:
                    node1, node2 = node2, node1
                results.append({"i": node1, "j": node2})
                # results.append({"i": node1, "j": node2, "context": '(row)'})

    edgelist_df = pd.DataFrame(results)
    edgelist_df = edgelist_df.groupby(["i", "j"]).size().reset_index(name="count")

    return edgelist_df


def g_from_weighted_edgelist(
    edgelist: pd.DataFrame,
    node_i_col: str = "i",
    node_j_col: str = "j",
    weight_col: str = "count",
) -> Graph:
    g = gt.Graph(directed=False)

    # add vertex and edge properties
    vprop_name = g.new_vertex_property("string")  # node names
    g.vertex_properties["vprop_name"] = vprop_name  # node names
    eprop_weight = g.new_edge_property("int")  # weight
    g.edge_properties["eprop_weight"] = eprop_weight  # weight

    # add nodes, edges
    vertices: Dict[str, gt.Vertex] = {}
    for _, row in edgelist.iterrows():  # vectorize later
        node_i, node_j, count = row[node_i_col], row[node_j_col], row[weight_col]

        if node_i not in vertices:
            v_i = g.add_vertex()
            vprop_name[v_i] = node_i
            vertices[node_i] = v_i
        else:
            v_i = vertices[node_i]

        if node_j not in vertices:
            v_j = g.add_vertex()
            vprop_name[v_j] = node_j
            vertices[node_j] = v_j
        else:
            v_j = vertices[node_j]

        e = g.add_edge(v_i, v_j)
        eprop_weight[e] = count

    return g


def construct_mention_network(
    df: pd.DataFrame,
    user_id_col: str = "user",
    comment_text_col: str = "comment_text",
    drop_isolates: bool = True,
    include_mentioned_no_comment_nodes: bool = True,
) -> Tuple[Graph, pd.DataFrame]:
    g = gt.Graph(directed=True)

    # extract all mentions
    df["mentioned_users"] = df[comment_text_col].str.findall(r"@(\w+)")
    # explode the df so that each mention becomes a separate row
    exploded_df = df.explode("mentioned_users").dropna(subset=["mentioned_users"])

    users = df[user_id_col].unique()
    user_nodes: Dict[str, gt.Vertex] = {user: g.add_vertex() for user in users}

    if include_mentioned_no_comment_nodes:
        mentioned_users = exploded_df["mentioned_users"].unique()
        for user in mentioned_users:
            if user not in user_nodes:
                user_nodes[user] = g.add_vertex()

    # create edge list with count weights
    edges = exploded_df[exploded_df[user_id_col] != exploded_df["mentioned_users"]]
    edge_list = edges[[user_id_col, "mentioned_users"]].values.tolist()
    edge_weights = Counter((user, mentioned_user) for user, mentioned_user in edge_list)

    # add weighted edges to the graph
    weight_property = g.new_edge_property("int")
    for (user, mentioned_user), weight in edge_weights.items():
        e = g.add_edge(user_nodes[user], user_nodes[mentioned_user])
        weight_property[e] = weight

    g.edge_properties["weight"] = weight_property

    # remove isolates if drop_isolates is true; in and out for directed network
    if drop_isolates:
        isolates = [
            v for v in g.vertices() if v.out_degree() == 0 and v.in_degree() == 0
        ]
        g.remove_vertex(isolates)

    weighted_edges_df = pd.DataFrame(
        [
            (user, mentioned_user, weight)
            for (user, mentioned_user), weight in edge_weights.items()
        ],
        columns=[user_id_col, "mentioned_users", "weight"],
    )

    weighted_edges_df.sort_values("weight", ascending=False, inplace=True)

    return g, weighted_edges_df


def construct_entity_network(
    df: pd.DataFrame,
    id_col: str,
    span_col: str,
    score_col: str,
    threshold: float = 0.5,
    typelist: Optional[List[str]] = None,
    edge_weight_threshold: int = 1,
) -> Tuple[GraphView, List[Tuple[str, str, int]]]:
    filtered_df = df[df[score_col] > threshold]
    if typelist is not None:
        filtered_df = filtered_df[filtered_df["label"].isin(typelist)]

    if filtered_df.empty:
        logging.info(f"Empty graph from threshold {threshold} and typelist {typelist}.")
        return GraphView(Graph()), []
        # return Graph(), []

    g = Graph()
    vprop_name = g.new_vertex_property("string")
    vprop_count = g.new_vertex_property("int")
    eprop_weight = g.new_edge_property("int")

    # property map, entity names to vertex ids
    entity_to_vertex: Dict[str, gt.Vertex] = {}

    # group by comment_id and add edges between co-mentioned entities
    grouped = filtered_df.groupby(id_col)
    for comment_id, group in grouped:
        entities = group[span_col].tolist()
        for i, entity1 in enumerate(entities):
            if entity1 not in entity_to_vertex:
                v1 = g.add_vertex()
                entity_to_vertex[entity1] = v1
                vprop_name[v1] = entity1
                vprop_count[v1] = 1
            else:
                v1 = entity_to_vertex[entity1]
                vprop_count[v1] += 1
            for entity2 in entities[i + 1 :]:
                if entity2 not in entity_to_vertex:
                    v2 = g.add_vertex()
                    entity_to_vertex[entity2] = v2
                    vprop_name[v2] = entity2
                    vprop_count[v2] = 1
                else:
                    v2 = entity_to_vertex[entity2]
                    vprop_count[v2] += 1

                if g.edge(v1, v2) is None:
                    e = g.add_edge(v1, v2)
                    eprop_weight[e] = 1
                else:
                    e = g.edge(v1, v2)
                    eprop_weight[e] += 1

    g.vertex_properties["name"] = vprop_name
    g.vertex_properties["count"] = vprop_count
    g.edge_properties["weight"] = eprop_weight

    gt.remove_self_loops(g)

    # Apply edge weight threshold filter
    edge_filter = g.new_edge_property("bool")
    for e in g.edges():
        edge_filter[e] = eprop_weight[e] >= edge_weight_threshold

    gv = GraphView(g, efilt=edge_filter)

    # Create weighted edgelist from the filtered graph
    weighted_edgelist: List[Tuple[str, str, int]] = []
    for e in gv.edges():
        v1, v2 = e
        weighted_edgelist.append((vprop_name[v1], vprop_name[v2], eprop_weight[e]))

    return gv, weighted_edgelist


def create_bipartite_edgelist(
    df: pd.DataFrame, topic_col: str, entity_list_col: str, weight_counts: bool = True
) -> pd.DataFrame:
    results: List[Dict[str, Any]] = []

    for _, row in df.iterrows():  # vectorize later
        topic = row[topic_col]
        entities = row[entity_list_col]
        unique_entities = set(entities)  # Ensure unique entities

        for entity in unique_entities:
            results.append({"Topic": topic, "Entity": entity})

    edgelist_df = pd.DataFrame(results)

    if weight_counts is True:
        edgelist_df = (
            edgelist_df.groupby(["Topic", "Entity"]).size().reset_index(name="Count")
        )
        return edgelist_df
    else:
        return edgelist_df


def construct_topic_entity_network(
    df: pd.DataFrame,
    topic_col: str = "Topic",
    entity_col: str = "Entity",
    weight_col: str = "Count",
    drop_isolates: bool = True,
) -> Tuple[Graph, pd.DataFrame]:
    g = gt.Graph(directed=False)

    # add vertex properties
    vprop_name = g.new_vertex_property("string")
    # add edge property for weight
    eprop_weight = g.new_edge_property("int")
    # bipartite vertex property map
    vtype = g.new_vertex_property("int")

    # ADD NODES
    node_ids: Dict[str, gt.Vertex] = {}
    for node in pd.concat([df[topic_col], df[entity_col]]).unique():
        v = g.add_vertex()
        vprop_name[v] = node
        node_ids[node] = v

    g.vertex_properties["name"] = vprop_name

    # set node types (for bipartite graph)
    for node in set(df[topic_col]):
        v = node_ids[node]
        vtype[v] = 0  # topic nodes

    for node in set(df[entity_col]):
        v = node_ids[node]
        vtype[v] = 1  # entity nodes

    g.vertex_properties["vtype"] = vtype

    # ADD WEIGHTED EDGES
    for _, row in df.iterrows():
        topic = row[topic_col]
        entity = row[entity_col]
        weight = row[weight_col]

        e = g.add_edge(node_ids[topic], node_ids[entity])
        eprop_weight[e] = weight

    g.edge_properties["weight"] = eprop_weight

    # FILTER / CLEAN UP GRAPH

    # https://graph-tool.skewed.de/static/doc/quickstart.html
    # "For undirected graphs, the “out-degree” is synonym for degree,
    # and in this case the in-degree of a vertex is always zero."

    isolates = [v for v in g.vertices() if v.out_degree() == 0]
    if drop_isolates and isolates:
        g.remove_vertex(isolates, fast=True)
        logging.info(
            (
                f"Removed {len(isolates)} isolates. "
                f"Remaining vertices: {g.num_vertices()}"
            )
        )

    # create wel
    weighted_edges_df = pd.DataFrame(
        [
            (vprop_name[e.source()], vprop_name[e.target()], eprop_weight[e])
            for e in g.edges()
        ],
        columns=[topic_col, entity_col, weight_col],
    )

    weighted_edges_df.sort_values(weight_col, ascending=False, inplace=True)
    return g, weighted_edges_df


# MODEL NETWORK FUNCTIONS #
#
# - Bayesian Planted Partition Models (BPPM)
# - Hierarachical Bayesian Stochastic Blockmodels (HBSBM)


def fit_bppm(
    g: gt.Graph, refine: str = "marginals", return_all_levels: bool = False
) -> Tuple[Any, Optional[pd.DataFrame]]:
    """fit a Bayesian planted partition model for assortative community structure"""
    state = gt.minimize_blockmodel_dl(g=g, state=gt.PPBlockState)
    if refine == "marginals":
        state = get_consensus_partition_from_posterior(state)
        block_data = _get_ppm_results(g=g, block_property_map=state.get_blocks())
        return state, block_data
    elif refine == "basic":
        state = _refine_state_multiflip_mcmc_sweep(state=state)
        block_data = _get_ppm_results(g=g, block_property_map=state.get_blocks())
        return state, block_data
    else:
        return state, None


# THIS VERSION DOES EDGE WEIGHTS DIFFERENTLY THAN BELOW. HAS ALSO BEEN TESTED AND WORKS.
# def fit_hbsbm(
#     graph: gt.Graph,
#     bip: Optional[str] = None,
#     eweight: Optional[str] = None,
#     recs: Optional[List[str]] = None,
#     rec_types: Optional[List[str]] = None,
#     covars: bool = False,
#     refine: str = "basic",  # or 'marginals'
#     return_all_levels: bool = False,
#     vertex_property_key: str = "vprop_name",
# ) -> Tuple[Any, Optional[pd.DataFrame]]:
#     """fit a Hierarchical Bayesian stochastic blockmodel"""
#     global bs
#     bs = []

#     def collect_partitions(s: Any) -> None:
#         global bs
#         bs.append(s.get_bs())

#     if covars:
#         recs = recs
#         rec_types = rec_types
#     else:
#         recs = []
#         rec_types = []

#     state = gt.minimize_nested_blockmodel_dl(
#         graph,
#         state_args=dict(
#             deg_corr=True,
#             eweight=eweight,
#             recs=recs,
#             rec_types=rec_types,
#             clabel=bip,
#             pclabel=bip,
#         ),
#     )

#     if refine == "marginals":
#         state = get_consensus_partition_from_posterior(state, graph)
#         block_data = _get_hbsbm_results(
#             state, all_levels=return_all_levels, vertex_property_key=vertex_property_key
#         )
#         return state, block_data
#     elif refine == "basic":
#         state = _refine_state_multiflip_mcmc_sweep(state)
#         block_data = _get_hbsbm_results(
#             state, all_levels=return_all_levels, vertex_property_key=vertex_property_key
#         )
#         return state, block_data
#     else:
#         return state, None


def fit_hbsbm(
    graph: gt.Graph,
    bip: Optional[str] = None,
    eweight: Optional[str] = None,
    recs: Optional[List[str]] = None,
    rec_types: Optional[List[str]] = None,
    covars: bool = False,
    refine: str = "basic",  # or 'marginals'
    return_all_levels: bool = False,
    vertex_property_key: str = "vprop_name",
) -> Tuple[Any, Optional[pd.DataFrame]]:
    """fit a Hierarchical Bayesian stochastic blockmodel"""
    global bs
    bs = []

    def collect_partitions(s: Any) -> None:
        global bs
        bs.append(s.get_bs())

    if covars:
        recs = recs
        rec_types = rec_types
    else:
        recs = []
        rec_types = []

    state = gt.minimize_nested_blockmodel_dl(
        graph,
        state_args=dict(
            # deg_corr=True,
            # eweight=eweight,
            recs=recs,
            rec_types=rec_types,
            # clabel=bip,
            # pclabel=bip,
        ),
    )

    if refine == "marginals":
        state = get_consensus_partition_from_posterior(state, graph)
        block_data = _get_hbsbm_results(
            state, all_levels=return_all_levels, vertex_property_key=vertex_property_key
        )
        return state, block_data
    elif refine == "basic":
        state = _refine_state_multiflip_mcmc_sweep(state)
        block_data = _get_hbsbm_results(
            state, all_levels=return_all_levels, vertex_property_key=vertex_property_key
        )
        return state, block_data
    else:
        return state, None


def _refine_state_multiflip_mcmc_sweep(state: Any, niter: int = 100) -> Any:
    for i in range(niter):
        state.multiflip_mcmc_sweep(niter=niter, beta=np.inf)
    return state


def get_consensus_partition_from_posterior(state, graph, force_niter=2000):
    global bs
    bs = []

    def collect_partitions(s) -> None:
        """
        Collect block assignments for NestedBlockState if it's a nested model.
        Collect block assignments for BlockState if it's not a nested model.
        """
        global bs
        if isinstance(s, gt.NestedBlockState):
            bs.append(s.get_bs())
        else:
            bs.append(s.b.a.copy())

    # MCMC equilibration
    gt.mcmc_equilibrate(
        state,
        force_niter=force_niter,
        mcmc_args=dict(niter=10),  # no. of iterations per sweep
        callback=collect_partitions,  # collect partitions during sweeps
    )

    # process partitions with PartitionModeState, check if model nested to avoid errors
    pmode = gt.PartitionModeState(
        bs, nested=isinstance(state, gt.NestedBlockState), converge=True
    )

    # get the posterior vertex marginals
    pv = pmode.get_marginal(graph)
    graph.vertex_properties["pv"] = pv

    # obtain the consensus partition
    if isinstance(state, gt.NestedBlockState):
        bs = pmode.get_max_nested()
    else:
        bs = pmode.get_max()

    state = state.copy(bs=bs)

    return state


def _get_ppm_results(g: gt.Graph, block_property_map: Any) -> pd.DataFrame:
    block_property_map = gt.contiguous_map(block_property_map)
    block_data = {g.vp.vprop_name[v]: block_property_map[v] for v in g.vertices()}
    block_data = pd.DataFrame(list(block_data.items()), columns=["Node", "BlockID"])
    return block_data


def _get_hbsbm_results(
    state: Any, all_levels: bool = False, vertex_property_key: str = "vprop_name"
) -> pd.DataFrame:
    """Get block memberships for nodes in a graph-tool state."""
    levels = state.get_levels()
    name_list: List[str] = []
    block_list: List[int] = []
    level_list: List[int] = []

    for level, level_state in enumerate(levels):
        block_map = level_state.get_blocks()
        for v in state.g.vertices():
            name_list.append(state.g.vertex_properties[vertex_property_key][v])
            block_list.append(block_map[v])
            level_list.append(level)

        # break after the first (base) level if not returning all levels
        if not all_levels:
            break

    data_df = pd.DataFrame()
    data_df["Node"] = name_list
    data_df["BlockID"] = block_list
    if all_levels:
        data_df["Level"] = level_list

    return data_df


def rotate_positions(pos, a):
    """Rotate the positions by `a` degrees."""
    theta = np.radians(a)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    x, y = pos.get_2d_array()
    cm = np.array([x.mean(), y.mean()])
    return pos.t(lambda x: R @ (x.a - cm) + cm)


def _sort_models_dict(models_dict):
    sorted_models = sorted(models_dict.items(), key=lambda x: x[1])
    labels, values = zip(*sorted_models)
    return sorted_models, labels, values


def plot_line_comparison(
    models_dict,
    xlabel="\nMinimal Description Length (MDL)",
    title="Model Comparison\n",
    padding=10_000,
    xrange=None,
    print_decimals=False,
    filename=None,
):
    """
    xrange: tuple of start and end for horizontal line and xaxis (e.g., 0, 1)
    """
    sorted_models, labels, values = _sort_models_dict(models_dict)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    if xrange is not None:
        xstart, xend = xrange
    else:  # use padding
        xstart = min(values) - padding
        xend = max(values) + padding

    plt.hlines(1, xstart, xend, color="black", linestyles="solid")
    plt.xlim(xstart, xend)

    # alternate annotations above and below the line for readability
    for i, (label, value) in enumerate(sorted_models):
        ax.plot(value, 1, "o", label=label, c="black")
        if i % 2 == 0:
            ax.text(value, 1.02, f"{label}\n{value:,}", ha="center", va="bottom")
        else:
            ax.text(value, 0.98, f"{label}\n{value:,}", ha="center", va="top")

    plt.yticks([])
    plt.xlabel(xlabel)
    plt.title(title, loc="left", fontsize=14)

    if print_decimals is True:
        plt.gca().xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{round(x, 1):,}")
        )
    else:
        plt.gca().xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
        )

    if filename is not None:
        plt.savefig(filename, dpi=300)


# def graph_tool_to_networkx(gt_graph):
#     """
#     Converts a graph-tool Graph object to a networkx Graph object.

#     Parameters:
#     gt_graph (gt.Graph): The graph-tool graph to convert.

#     Returns:
#     nx.Graph: The converted networkx graph.
#     """
#     nx_graph = nx.DiGraph() if gt_graph.is_directed() else nx.Graph()

#     # Add nodes
#     for v in gt_graph.vertices():
#         nx_graph.add_node(int(v))

#     # Add edges
#     for e in gt_graph.edges():
#         source = int(e.source())
#         target = int(e.target())
#         nx_graph.add_edge(source, target)

#     return nx_graph
