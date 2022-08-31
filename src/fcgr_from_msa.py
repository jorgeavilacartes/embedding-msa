import dvc.api
import numpy as np

from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

from utils import count_kmers_from_msa
from fcgr import FCGRFromKmerCount

## params
params=dvc.api.params_show()
k = params['k'] # kmer
FOLDER_MSA=Path(params['fcgr_from_msa']['folder_msa']) # input folder
PATH_FCGR=Path(params['fcgr_from_msa']['path_fcgr'])  # output folder
PATH_FCGR.mkdir(exist_ok=True, parents=True)

# Instantiate FCGR
fcgr_from_kmer_count = FCGRFromKmerCount(k)

# Save number of sequences in each MSA
seqs_by_msa = defaultdict(int)
empty_msa = list()
# Generate FCGR from MSA
list_msa=list(FOLDER_MSA.rglob('*.fa'))
for path_msa in tqdm(list_msa):
    try:
        # count kmers
        kmer_count, nseqs= count_kmers_from_msa(k, path_msa, return_nseqs=True)
        
        # save fcgr 
        fcgr = fcgr_from_kmer_count(kmer_count)   
        np.save(file=PATH_FCGR.joinpath(path_msa.stem+'.npy'), arr=fcgr)
        seqs_by_msa[str(path_msa)] = nseqs

    except:
        empty_msa.append(str(path_msa))

# TODO: save seqs_by_msa and empty msa