import os
import pdb
import time
import torch
import numpy as np
from itertools import groupby
import torch.nn.functional as F


class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0):
        self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        vocab = [chr(x) for x in range(20000, 20000 + num_classes)]
        self.vocab = vocab

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        if self.search_mode == "max":
            return self.MaxDecode(nn_output, vid_lgt)
        else:
            return self.BeamSearch(nn_output, vid_lgt, probs)

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        '''
        Custom Beam Search Implementation:
                - Input: nn_output (B, T, N), which should be passed through a softmax layer
                - Output: decoded results
        '''
        if not probs:
            nn_output = nn_output.softmax(-1).cpu()
        vid_lgt = vid_lgt.cpu()
        ret_list = []
        for batch_idx in range(len(nn_output)):
            beam_candidates = []
            for t in range(nn_output.shape[1]):
                top_k = torch.topk(nn_output[batch_idx, t], k=10)  # Top-k beam search
                beam_candidates.append(top_k.indices.tolist())
            # Decode beam candidates
            first_result = [x[0] for x in groupby([item for sublist in beam_candidates for item in sublist])]
            if len(first_result) != 0:
                # Convert integers to tensors before stacking
                first_result = torch.tensor(first_result, dtype=torch.long)
            # Filter out class IDs that are not in the dictionary (like 0, which is often a blank token)
            result_pairs = []
            for idx, gloss_id in enumerate(first_result):
                gloss_id_int = int(gloss_id)
                if gloss_id_int in self.i2g_dict:
                    result_pairs.append((self.i2g_dict[gloss_id_int], idx))
            ret_list.append(result_pairs)
        return ret_list

    def MaxDecode(self, nn_output, vid_lgt):
        index_list = torch.argmax(nn_output, axis=2)
        batchsize, lgt = index_list.shape
        ret_list = []
        for batch_idx in range(batchsize):
            group_result = [x[0] for x in groupby(index_list[batch_idx][:vid_lgt[batch_idx]])]
            filtered = [*filter(lambda x: x != self.blank_id, group_result)]
            if len(filtered) > 0:
                max_result = torch.stack(filtered)
                max_result = [x[0] for x in groupby(max_result)]
            else:
                max_result = filtered
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(max_result)])
        return ret_list
