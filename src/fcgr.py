from typing import Dict
import numpy as np
from pathlib import Path
from complexcgr import FCGR

class FCGRFromKmerCount(FCGR):

    def __init__(self, k: int, bits: int = 8):
        super().__init__(k, bits)

    def __call__(self, kmer_count: Dict[str,int])-> np.ndarray:
        "Given a DNA sequence, returns an array with his FCGR"
       
        # Create an empty array to save the FCGR values
        array_size = int(2**self.k)
        fcgr = np.zeros((array_size,array_size))

        # Assign frequency to each box in the matrix
        for kmer, freq in kmer_count.items():        
            pos_x, pos_y = self.kmer2pixel[kmer]
            fcgr[int(pos_x)-1,int(pos_y)-1] = freq
        return fcgr