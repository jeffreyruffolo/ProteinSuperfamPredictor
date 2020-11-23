from itertools import chain
import os

from glob import glob
from tqdm import tqdm
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser

_aa_dict = {
    'A': 0,
    'C': 1,
    'D': 2,
    'E': 3,
    'F': 4,
    'G': 5,
    'H': 6,
    'I': 7,
    'K': 8,
    'L': 9,
    'M': 10,
    'N': 11,
    'P': 12,
    'Q': 13,
    'R': 14,
    'S': 15,
    'T': 16,
    'V': 17,
    'W': 18,
    'Y': 19,
    'Z': 20,
    'X': 21
}

_aa_3_1_dict = {
    'ALA': 'A',
    'CYS': 'C',
    'ASP': 'D',
    'GLU': 'E',
    'PHE': 'F',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LYS': 'K',
    'LEU': 'L',
    'MET': 'M',
    'ASN': 'N',
    'PRO': 'P',
    'GLN': 'Q',
    'ARG': 'R',
    'SER': 'S',
    'THR': 'T',
    'VAL': 'V',
    'TRP': 'W',
    'TYR': 'Y'
}


def get_dataset_stats(data, max_protein_len=300):
    num_proteins = 0
    for pdb_id, chain_residue_range, superfam in tqdm(data):
        pdb_file = os.path.join(pdb_dir, "{}.pdb".format(pdb_id))
        if not os.path.exists(pdb_file):
            continue

        if len(chain_residue_range.split(":")) != 2:
            continue

        chain_id, residue_range = chain_residue_range.split(":")
        residue_range = get_residue_range(residue_range)

        if residue_range == None:
            continue

        _, sequence = get_residue_range_coords(pdb_file, pdb_id, chain_id,
                                               residue_range)
        if len(sequence) > max_protein_len:
            continue

        num_proteins += 1

    return num_proteins


def get_ca_coord(residue):
    if 'CA' in residue:
        return residue['CA'].get_coord()
    else:
        return [0, 0, 0]


def get_residue_range(residue_range):
    residue_range_split = residue_range.split("-")

    try:
        if len(residue_range_split) == 2:
            return (int(residue_range_split[0])), int(residue_range_split[1])
        elif len(residue_range_split) == 3:
            return (-1 * int(residue_range_split[1])), int(
                residue_range_split[2])
        elif len(residue_range_split) == 4:
            return (
                -1 *
                int(residue_range_split[1])), -1 * int(residue_range_split[3])
    except:
        print("Invalid residue range: {}".format(residue_range))

    return None


def get_residue_range_coords(pdb_file, pdb_id, chain_id, residue_range):
    p = PDBParser()
    structure = p.get_structure(pdb_id, pdb_file)

    for chain in structure.get_chains():
        if chain.id == chain_id:
            residues = list(chain.get_residues())

    residues = [
        r for r in residues if residue_range[0] <= r.id[1] <= residue_range[1]
    ]
    ca_coords = torch.tensor([get_ca_coord(r) for r in residues])
    sequence = "".join(
        [_aa_3_1_dict.setdefault(r.get_resname(), "X") for r in residues])

    return ca_coords, sequence


def calc_dist_mat(coords):
    mat_shape = (len(coords), len(coords), 3)

    a_coords = coords.unsqueeze(0).expand(mat_shape)
    b_coords = coords.unsqueeze(1).expand(mat_shape)
    dist_mat = (a_coords - b_coords).norm(dim=-1)

    return dist_mat


pdb_dir = "data/pdb_files"
data = np.genfromtxt("data/scop-simplified.tsv", delimiter="\t",
                     dtype=str)[:500]

max_protein_len = 300
num_proteins = get_dataset_stats(data, max_protein_len)
print("\nCreating dataset of ")

out_file = "data/dataset.h5"
h5_file = h5py.File(out_file, 'w')
id_set = h5_file.create_dataset('id', (num_proteins, ),
                                compression='lzf',
                                dtype='S25',
                                maxshape=(None, ))
superfam_set = h5_file.create_dataset('superfam', (num_proteins, ),
                                      compression='lzf',
                                      dtype='S25',
                                      maxshape=(None, ))
sequence_set = h5_file.create_dataset('sequence',
                                      (num_proteins, max_protein_len),
                                      compression='lzf',
                                      dtype='uint8',
                                      maxshape=(None, max_protein_len),
                                      fillvalue=-1)
sequence_len_set = h5_file.create_dataset('sequence_len', (num_proteins, ),
                                          compression='lzf',
                                          dtype='uint16',
                                          maxshape=(None, ),
                                          fillvalue=-1)
dist_mat_set = h5_file.create_dataset(
    'dist_mat', (num_proteins, max_protein_len, max_protein_len),
    maxshape=(None, max_protein_len, max_protein_len),
    compression='lzf',
    dtype='float',
    fillvalue=-1)

h5_index = 0
for pdb_id, chain_residue_range, superfam in data:
    pdb_file = os.path.join(pdb_dir, "{}.pdb".format(pdb_id))
    if not os.path.exists(pdb_file):
        continue

    if len(superfam) < 2:
        print()

    if len(chain_residue_range.split(":")) != 2:
        continue

    chain_id, residue_range = chain_residue_range.split(":")
    residue_range = get_residue_range(residue_range)

    ca_coords, sequence = get_residue_range_coords(pdb_file, pdb_id, chain_id,
                                                   residue_range)
    if len(sequence) > max_protein_len:
        continue

    dist_mat = calc_dist_mat(ca_coords)
    encoded_sequence = np.array([_aa_dict[s] for s in sequence])

    protein_id = "{}_{}_{}_{}".format(pdb_id, chain_id, residue_range[0],
                                      residue_range[1])

    id_set[h5_index] = protein_id
    superfam_set[h5_index] = superfam
    sequence_set[h5_index, :len(sequence)] = encoded_sequence
    sequence_len_set[h5_index] = len(encoded_sequence)
    dist_mat_set[h5_index, :len(sequence), :len(sequence)] = dist_mat

    h5_index += 1
