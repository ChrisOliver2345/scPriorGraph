import pandas as pd
import numpy as np
import dgl
import tqdm

def construct_graph():
    file_edge = './lr_pairs.txt'
    f_edge = pd.read_table(file_edge, sep='\t', header=None)

    ligand = set()
    receptor = set()
    for index, row in f_edge.iterrows():
        ligand.add(row[0])
        receptor.add(row[1])

    ligand_ids_index_map = {x: i for i, x in enumerate(ligand)}
    receptor_ids_index_map = {x: i for i, x in enumerate(receptor)}
    ligand_index_id_map = {i: x for i, x in enumerate(ligand)}
    receptor_index_id_map = {i: x for i, x in enumerate(receptor)}

    ligand_receptor_src = []
    ligand_receptor_dst = []
    for index, row in f_edge.iterrows():
        ligand_receptor_src.append(ligand_ids_index_map.get(row[0]))
        ligand_receptor_dst.append(receptor_ids_index_map.get(row[1]))

    print(ligand_ids_index_map)
    print(receptor_ids_index_map)

    li_re = dgl.bipartite((ligand_receptor_src, ligand_receptor_dst), 'ligand', 'li_re', 'receptor')
    re_li = dgl.bipartite((ligand_receptor_dst, ligand_receptor_src), 'receptor', 're_li', 'ligand')
    hg = dgl.hetero_from_relations([li_re, re_li])

    return hg, ligand_index_id_map, receptor_index_id_map


def parse_trace(trace, ligand_index_id_map, receptor_index_id_map, count):
    s = []
    for index in range(trace.size):
        if index % 2 == 0:
            if index == 0:
                s.append("path_" + str(count))
            s.append(ligand_index_id_map[trace[index]])
        else:
            s.append(receptor_index_id_map[trace[index]])
    return '\t'.join(s)


def main():
    count = 0
    hg, ligand_index_id_map, receptor_index_id_map = construct_graph()
    print(hg)
    meta_path = ['li_re', 're_li', 'li_re', 're_li', 'li_re', 're_li']
    num_walk_per_node = 1
    walk_length = 30
    f = open("./output_path4.txt", "w")
    for ligand_idx in tqdm.trange(hg.number_of_nodes('ligand')):
        trace = dgl.contrib.sampling.metapath_random_walk(
            hg=hg, etypes=meta_path * walk_length, seeds=[ligand_idx, ], num_traces=num_walk_per_node)
        tr = trace[0][0].numpy()
        tr = np.insert(tr, 0, ligand_idx)
        res = parse_trace(tr, ligand_index_id_map, receptor_index_id_map, count)
        count += 1
        f.write(res + '\n')
    f.close()


if __name__ == '__main__':
    main()