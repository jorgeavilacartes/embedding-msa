from pathlib import Path
from collections import defaultdict
from typing import Dict, Union
from Bio import AlignIO, Seq

def clean_seq(seq: Seq) -> Seq:
    "remove `-`"
    seq = seq.replace("-","")
    return seq

def count_kmers_from_msa(k: int, path_msa: Union[str,Path], return_nseqs=False) -> Dict[str,int]: 
    "Return kmer count and number of seqs in the MSA"
    # load MSA
    msa = AlignIO.read(path_msa, format="fasta")
    
    # Count kmers in each row
    kmer_count = defaultdict(int)
    for i,s in enumerate(msa):
        seq = clean_seq(s.seq)
        last_j = len(seq) - k + 1   
        kmers  = (str(seq[j:(j+k)]) for j in range(last_j))
        for kmer in kmers:
            kmer_count[kmer] += 1

    # number of sequences in the alignment
    n_seqs = i + 1 

    if return_nseqs is True:
        return kmer_count, n_seqs
    
    return kmer_count