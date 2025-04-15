# see if DINOv2 global features work by themselves?
# prepare topk for ROx RPar and gldv2-test
import argparse
import os.path
from glob import glob

import os.path as osp
import numpy as np
import faiss
import torch

from common_utils.serialization import pickle_save, pickle_load
from common_utils.revisited import compute_metrics

def test_nonzero_features(desc):
    norms = np.linalg.norm(desc, axis=-1)
    print(norms.shape)
    failed_ids = np.where(norms == 0)[0]
    if len(failed_ids):
        print(f"amount failed: {len(failed_ids)}")
    else:
        print("OK")


def load_or_combine(feat_dir, file_name, dim, num_desc, dtype = 'float32'):
    # gldv2-test, revisitop has different patterns
    desc_name = file_name.split('_')[0]
    if 'cvnet_cls' in file_name:
        desc_name = 'cvnet_cls'
    
    if 'gldv2-index' in feat_dir:
        splits = sorted(glob(osp.join(feat_dir, f'{desc_name}?_global.pt')))
    elif 'roxford5k+1m' in feat_dir or 'rparis6k+1m' in feat_dir:
        splits = []   # TODO: revisitop1m
    else:  # default split pattern
        splits = sorted(glob(osp.join(feat_dir, f'{file_name}_xa?.pt')))
        
    if len(splits):
        final_desc = np.memmap(osp.join(feat_dir, f'{file_name}.pt'), dtype=dtype, mode='w+', shape=(num_desc, dim))
        k = 0
        
        # get meta split files
        meta_splits = sorted(glob(osp.join(feat_dir, f'test_gallery_*.txt')))
        assert len(meta_splits) == len(splits), f"Meta split files {len(meta_splits)} != {len(splits)}, {meta_splits}, {splits}"
        for split, meta_split in zip(splits, meta_splits):
            with open(meta_split) as fid:
                num_chunk = len(fid.read().splitlines())

            desc = np.memmap(split, dtype=dtype, mode='r', shape=(num_chunk, dim))
            test_nonzero_features(desc)
            final_desc[k:k+len(desc)] = desc
            k += len(desc)
        # for chunk in splits:
        #     chunk_ext = chunk.split('.')[0][-3:]
        #     with open(osp.join(feat_dir, chunk_ext)) as fid:
        #         num_chunk = len(fid.read().splitlines())

        #     desc = np.memmap(chunk, dtype=dtype, mode='r', shape=(num_chunk, dim))
        #     test_nonzero_features(desc)
        #     final_desc[k:k+len(desc)] = desc
        #     k += len(desc)
        desc = final_desc
    else:
        desc = np.memmap(osp.join(feat_dir, f'{file_name}.pt'), dtype=dtype, mode='r', shape=(num_desc, dim))
        print(f"Loaded {osp.join(feat_dir, file_name)}.pt")

    return desc


def main():
    parser = argparse.ArgumentParser(description='Compute and store image similarities and ranking with global embeddings.')
    parser.add_argument('--dataset', help='Dataset name to load embeddings of.')
    parser.add_argument('--desc_name', default='dinov2_cls', help='Embeddings to load based on name.')
    # parser.add_argument('--server', default='mnt', help='mnt/datagrid')
    parser.add_argument('--ext', nargs='?', const='', default='', help='extension, e.g. _pq8')

    args = parser.parse_args()

    dataset = args.dataset
    # data_dir = f'/{args.server}/personal/sumapave/data/features/'
    data_dir = f"/scratch/zx51/ames/my_data/"
    feat_dir = os.path.join(data_dir, dataset)
    desc_name = args.desc_name
    dim = 768 if 'dinov2' in desc_name else 2048
    m = 1600  # keep top-m results
    ext = args.ext
    dtype = 'float32'

    num_desc = 0
    gnd = None

    if dataset == 'gldv2' or dataset == 'gldv2-train':
        with open(osp.join(feat_dir, 'train_750k.txt')) as fid:
            db_lines = fid.read().splitlines()
        num_desc += len(db_lines)
        # desc = load_or_combine(feat_dir, f'{desc_name}_global{ext}', dim, num_desc)
        desc = load_or_combine(feat_dir, f'{desc_name}_global', dim, num_desc)

        query = desc

    # elif dataset == 'revisitop1m' or dataset[-3:] == '+1m':  # +1m needs further processing
    elif dataset == 'revisitop1m':   # this line not making sense, let's make +1m a separate case
        with open(osp.join(data_dir, 'revisitop1m', 'imlist.txt')) as fid:
            db_lines = fid.read().splitlines()
        num_desc += len(db_lines)

        desc = load_or_combine(osp.join(data_dir, 'revisitop1m'), 
                               f'{desc_name}_global{ext}', dim, num_desc)
        query = desc
        
    # elif dataset.endswith('+1m'):   # +1m needs concat 1m with gallery and get nn_dinov2.pkl
    #     pass

    elif dataset.startswith(('roxford5k', 'rparis6k', 'gldv2-test', 'instre')):
        # small_feat_dir = os.path.join(data_dir, dataset.split('+')[0])   # why exclude +1m?
        small_feat_dir = os.path.join(data_dir, dataset)
        query_feat_dir = os.path.join(data_dir, dataset.split('+')[0])   # query feat is not in +1m
        
        with open(osp.join(small_feat_dir, 'test_query.txt')) as fid:
            query_lines = fid.read().splitlines()
        with open(osp.join(small_feat_dir, 'test_gallery.txt')) as fid:
            db_lines = fid.read().splitlines()
            
        # skip gnd eval for gldv2
        gnd = None
        if not dataset.startswith('gldv2'):
            # gnd_file does not have +1m
            gnd = pickle_load(osp.join(feat_dir.replace('+1m', ''), 
                                       f'gnd_{dataset.replace("+1m", "")}.pkl'))['gnd']
            
        # if doing gldv2-test, query is from gldv2-test, db is from gldv2-index
        if dataset == 'gldv2-test':
            small_feat_dir = os.path.join(data_dir, 'gldv2-test')
            query = load_or_combine(small_feat_dir, f'{desc_name}_global{ext}', dim, len(query_lines))
            db_feat_dir = os.path.join(data_dir, 'gldv2-index')
            db_desc = load_or_combine(db_feat_dir, f'{desc_name}_global{ext}', dim, len(db_lines))
        else:
            query = load_or_combine(query_feat_dir, f'{desc_name}_query_global{ext}', dim, len(query_lines))
            db_desc = load_or_combine(small_feat_dir, f'{desc_name}_gallery_global{ext}', dim, len(db_lines))

        if num_desc > 0:
            final_desc = np.memmap(osp.join(feat_dir, f'{desc_name}_gallery_global{ext}.pt'), dtype=dtype, mode='w+', shape=(num_desc + len(db_lines), dim))
            final_desc[:len(db_lines)] = db_desc
            final_desc[len(db_lines):] = desc
            num_desc += len(db_lines)
            desc = final_desc
        else:
            desc = db_desc
            
    elif dataset.startswith(('met', 'sop', 'food2k', 'inshop', 'rp2k')):
        with open(osp.join(feat_dir, 'test_query.txt')) as fid:
            query_lines = fid.read().splitlines()
        with open(osp.join(feat_dir, 'test_gallery.txt')) as fid:
            db_lines = fid.read().splitlines()

        query = load_or_combine(feat_dir, f'{desc_name}_query_global{ext}', dim, len(query_lines))
        desc = load_or_combine(feat_dir, f'{desc_name}_gallery_global{ext}', dim, len(db_lines))

    test_nonzero_features(desc)

    self = 0
    # if dataset in ['gldv2', 'sop', 'sop_1k', 'food2k', 'rp2k']:
    if any(dataset.startswith(x) for x in ['gldv2-train', 'sop', 'sop_1k', 'food2k', 'rp2k']):
        self = 1

    output_path = osp.join(data_dir, dataset, f'nn_{desc_name}{ext}.pkl')
    # idx = faiss.GpuIndexFlatIP(faiss.StandardGpuResources(), dim)
    idx = faiss.IndexFlatIP(dim)
    # idx = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, idx)
    desc = desc.copy(order='C')
    idx.add(desc)
    scores = idx.search(query, m + self)
    print(scores[0].shape)
    scores = np.stack(scores)
    scores = scores[:, :, self:]

    if dataset.startswith('gldv2'):
        pickle_save(output_path, scores)   # scores then indices
    else:
        pickle_save(output_path, torch.from_numpy(scores))
        print(f"Saved {output_path}")
        class Q:
            def __init__(self):
                self.name = dataset
                self.lines = query_lines
        class D:
            def __init__(self):
                self.name = dataset
                self.lines = db_lines
        compute_metrics(Q(), D(), scores[1].T, gnd)


if __name__ == '__main__':
    main()