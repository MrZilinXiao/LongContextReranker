import torch
from typing import List
import numpy as np

RANKS_BY_DATASET = {
    'sop': [1, 10],  # excluding 100 & 1000 as we only rerank top 100
    'cub200': [1, 2, 4, 8],
    'inshop': [1, 10, 20, 30],
    'cars': [1, 2, 4, 8],
}

def recall_at_k_by_ranks(query_labels, gallery_labels, ranks, ks: List[int]):
    # ranks = cosine_matrix.topk(k=ks[-1], dim=-1, largest=True)[1]  shape: (num_queries, num_galleries)
    # the second dim of `ranks` is reranked. 
    gallery_labels = query_labels if gallery_labels is None else gallery_labels
    if not isinstance(query_labels, torch.Tensor):
        query_labels = torch.from_numpy(query_labels)
    if not isinstance(gallery_labels, torch.Tensor):
        gallery_labels = torch.from_numpy(gallery_labels)
    
    recall_list = []
    for k in ks:
        correct = (gallery_labels[ranks[:, :k]] == query_labels.unsqueeze(-1)).any(dim=-1).float()
        recall_list.append((torch.sum(correct) / len(query_labels)).item())
    
    return recall_list

def mean_average_precision_at_k_by_ranks(query_labels, gallery_labels, ranks, k: int = 100):
    pass

def mean_average_precision_at_r_by_ranks(query_labels, gallery_labels, indices):
    # `ranks` is pre-computed index in (num_query, num_gallery)
    # Calculate MAP@R: mean of AP@R, where R is the number of total positive items for a specific query
    # if query_labels and gallery_labels are numpy arrays, convert to torch tensors
    num_queries = len(query_labels)
    gallery_labels = query_labels if gallery_labels is None else gallery_labels
    
    if not isinstance(query_labels, torch.Tensor):
        query_labels = torch.from_numpy(query_labels)
    if not isinstance(gallery_labels, torch.Tensor):
        gallery_labels = torch.from_numpy(gallery_labels)
    
    average_precisions = []
    for i in range(num_queries):
        # Boolean array of relevant indices (same label as query label)
        relevant_indices = gallery_labels == query_labels[i]   # Bool tensor of relevant items [0, 1, 0, 1,...]
        num_relevant = relevant_indices.sum().item()  # Count of relevant items
        
        if num_relevant > 0:  # have at least one TP! Now compute AP@R
            # Array that is True at indices where retrieved items are relevant
            retrieved_relevant = relevant_indices[indices[i]][:num_relevant]  
            # Doing top index selection over bool tensor. [1,1,0,0,...] but truncated to num_relevant
            precision_at_k = torch.cumsum(retrieved_relevant.float(), dim=0) / torch.arange(1, num_relevant + 1).float()
            average_precision = precision_at_k.sum() / num_relevant
        else:
            average_precision = 0.0
        
        average_precisions.append(average_precision)
    # Mean of average precisions across all queries
    mean_average_precision = torch.tensor(average_precisions).mean().item()
    return mean_average_precision
        

def mean_average_precision_at_r(query_features, query_labels, 
                                gallery_features=None, gallery_labels=None, 
                                return_ranks=False, gallery_trunc=None):
    is_empty_gallery = gallery_features is None
    num_queries = len(query_labels)
    num_gallery = len(gallery_labels) if gallery_labels is not None else num_queries
    gallery_features = query_features if gallery_features is None else gallery_features
    gallery_labels = query_labels if gallery_labels is None else gallery_labels

    # Compute cosine similarity matrix
    cosine_matrix = query_features @ gallery_features.t()

    # Handling self-retrieval scenario
    if is_empty_gallery:
        # print("Warning: gallery_features is None. Overwriting.")
        cosine_matrix.fill_diagonal_(-1e8)  # avoid self-matching
    
    # Retrieve indices sorted by descending similarity
    indices = cosine_matrix.argsort(dim=-1, descending=True)
    
    if gallery_trunc is not None:
        indices = indices[:, :gallery_trunc]
    
    if return_ranks:
        return mean_average_precision_at_r_by_ranks(query_labels, gallery_labels, indices), indices
    
    return mean_average_precision_at_r_by_ranks(query_labels, gallery_labels, indices)

def mean_average_precision(ranks, nres, qcls, cls, kappas=[]):
    """
    Computes (mean) average precision, recall.
    Assumes each image belongs to exactly one class only
    Arguments
    ---------
    ranks : zero-based ranks of positive images QxDB
    nres  : number of positive images for each query Qx1
    qcls  : array/tensor of class ids for each query Qx1
    cls   : array/tensor of class ids for each image Dx1
    kappas: list of kappas for metric @ k
    Returns
    -------
    map    : mean average precision over all queries
    aps    : average precision per query
    apk    : average precision @ given kappas
    rec    : recall @ given kappas
    """
    apk = np.zeros(len(kappas))
    rec = np.zeros(len(kappas))
    mask = (cls[ranks] == qcls).T
    prec = np.cumsum(mask, axis=1) / (np.arange(mask.shape[1]) + 1)
    aps = (prec * mask).sum(1) / np.minimum(ranks.shape[0], nres)
    for j, k in enumerate(kappas):
        apk[j] = np.mean((prec[:, :k] * mask[:, :k]).sum(1) / np.minimum(k, nres))
        rec[j] = np.mean(np.any(mask[:, :k], axis=1))
    map = aps.mean()
    return map, aps, apk, rec

def compute_rectangular_ap(ranks, nres):
    if len(ranks) < 1:
        return 0.
    mask = np.zeros(ranks.max() + 1)
    mask[ranks] = 1
    prec = np.cumsum(mask) / (np.arange(mask.shape[0]) + 1)
    return (mask * prec).sum() / nres

def compute_trapezoidal_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    Arguments
    ---------
    ranks : zero-based ranks of positive images
    nres  : number of positive images
    Returns
    -------
    ap    : average precision
    """
    # number of images ranked by the system
    nimgranks = len(ranks)
    # accumulate trapezoids in PR-plot
    ap = 0
    recall_step = 1. / nres
    for j in np.arange(nimgranks):
        rank = ranks[j]
        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank
        precision_1 = float(j + 1) / (rank + 1)
        ap += (precision_0 + precision_1) * recall_step / 2.
    return ap

def compute_map(ranks, gnd, kappas=[], ap_f=compute_trapezoidal_ap):
    """
    Computes the mAP for a given set of returned results.
         Usage:
           map = compute_map (ranks, gnd)
                 computes mean average precsion (map) only
           map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """
    map = 0.
    nq = len(gnd) # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0
    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])
        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue
        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)
        ###########################################################
        # pp = np.in1d(ranks[:100,i], qgnd).tolist()
        # nn = [not xx for xx in pp]
        # foo = np.concatenate([np.arange(100)[np.array(pp)], np.arange(100)[np.array(nn)]])
        # bar = ranks[:, i][foo]
        # ranks[:100, i] = bar
        ###########################################################
        # sorted positions of positive and junk images (0 based)
        pos  = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgndj)]
        k = 0
        ij = 0
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1
        # compute ap
        ap = ap_f(pos, min(len(qgnd), ranks.shape[0]))
        map = map + ap
        aps[i] = ap
        # compute precision @ k
        if len(pos) > 0:
            pos += 1 # get it to 1-based
            for j in np.arange(len(kappas)):
                kq = min(max(pos), kappas[j]);
                prs[i, j] = (pos <= kq).sum() / kq
            pr = pr + prs[i, :]
    map = map / (nq - nempty)
    pr = pr / (nq - nempty)
    return map, aps, pr, prs


def compute_rectangular_ap(ranks, nres):
    if len(ranks) < 1:
        return 0.

    mask = np.zeros(ranks.max() + 1)
    mask[ranks] = 1
    prec = np.cumsum(mask) / (np.arange(mask.shape[0]) + 1)
    return (mask * prec).sum() / nres