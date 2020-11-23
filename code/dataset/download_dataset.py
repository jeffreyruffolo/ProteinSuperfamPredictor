from operator import sub
import os
import subprocess
import numpy as np
from tqdm import tqdm

data = np.genfromtxt("data/scop-cla-latest.txt", skip_header=6, dtype=str)
data = data[:, [1, 7, -1]]

for i in range(len(data)):
    data[i][2] = data[i][2].split(",")[3][3:]

np.savetxt("data/scop-simplified.tsv", data, delimiter='\t', fmt="%s")

pdbs = list(set(data[:, 0].tolist()))
out_path = "data/pdb_files"
for pdb_id in tqdm(pdbs):
    pdb_url = "https://files.rcsb.org/download/" + pdb_id + ".pdb"
    subprocess.call(["wget", pdb_url, "-P", out_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)