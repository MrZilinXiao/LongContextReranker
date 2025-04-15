# this file is an AMES counterpart of gld_data_utils.py
from torch.utils.data import Dataset, ConcatDataset
import os.path as osp
import torch
from common_utils.serialization import pickle_load
from dataclasses import dataclass
import random
import numpy as np
import os
import pandas as pd
from typing import Sequence, Dict, Any, Tuple, List, Union, Optional
import pickle
import h5py

def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def construct_sample_list(txt_path):
    with open(txt_path, 'r') as f:
        samples = f.readlines()
    samples = [(line.split(',')[0], int(line.split(',')[1]), int(line.split(',')[2]), int(line.split(',')[3]))
                for line in samples]
    categories = sorted(list(set([int(entry[1]) for entry in samples])))
    cat_to_label = dict(zip(categories, range(len(categories))))
    samples = [(entry[0], cat_to_label[entry[1]], entry[2], entry[3]) for entry in samples]
    targets = np.asarray([entry[1] for entry in samples])
    
    return samples, targets

class GLDv2ExtractiveDatasetHDF5(Dataset):
    # if global not provided, provide 49 local features; else 48 + 1 global
    def __init__(self, local_hdf5_path, nn_inds_path, sample_txt_path, global_pt_path=None, global_dim=768,
                 num_desc=49, topk=100, max_pos_per_topk=100, min_pos_per_topk=0, 
                 shuffle_indices=False, num_samples=None):
        super(GLDv2ExtractiveDatasetHDF5, self).__init__()
        self.local_hdf5_path = local_hdf5_path
        data_size, data_topk, local_dim = h5py.File(local_hdf5_path, 'r')['features'].shape
        print(f"Data size: {data_size}, local_topk: {data_topk}, local_dim: {local_dim}")
        if global_pt_path is not None:
            self.global_feats = np.memmap(global_pt_path, 
                                        dtype=np.float32, mode='r', shape=(data_size, global_dim))
        else:
            self.global_feats = None
            
        self.nn_inds = pickle_load(nn_inds_path)  # [2, data_size, data_topk]
        assert self.nn_inds.shape[1] == data_size, f"nn_inds shape mismatch! nn_inds: {self.nn_inds.shape}, data_size: {data_size}"
        # just keep the inds-dim in training
        self.nn_inds = self.nn_inds[1, :, :].astype(int)
        self.samples, self.targets = construct_sample_list(sample_txt_path)
        # build helper data structure: target -> other db images with this label
        self.label_indices = [
            np.where(self.targets == i)[0].tolist() 
            for i in range(len(np.bincount(self.targets)))
        ]
        # record other parameters
        self.num_desc = num_desc
        self.topk = topk
        self.max_pos_per_topk = max_pos_per_topk
        self.min_pos_per_topk = min_pos_per_topk
        
        self.shuffle_indices = shuffle_indices or self.max_pos_per_topk == 1
        if self.shuffle_indices:
            print("Warning: shuffle_indices is on. max_pos_per_topk will sample from shuffled indices.")
            
        if num_samples is not None:
            self.samples = self.samples[:num_samples]
            # self.targets = self.targets[:num_samples]
    
    def get_tensor_dict(self, index, local_storage, 
                        fields=["global_feats", "local_feats", "local_scales", "local_mask"]):
        ret_dict = dict()
        if "global_feats" in fields:
            ret_dict["global_feats"] = torch.from_numpy(self.global_feats[index])
        if "local_feats" in fields:
            local_feats = torch.from_numpy(local_storage['features'][index, :self.num_desc])
            # get meta data from local storage previous 5 columns
            local_meta = local_feats[:, :5]
            ret_dict["local_feats"] = local_feats[:, 5:]
            if "local_scales" in fields:
                ret_dict["local_scales"] = local_meta[:, 2].long()
            if "local_mask" in fields:  # 0 -> valid, 1 -> invalid
                ret_dict["local_mask"] = local_meta[:, 3].long()
        return ret_dict
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_path, landmark_id, _, _ = self.samples[index]  # (img_path, landmark_id, w, h / h, w)
        fields_needed = ["global_feats", "local_feats", "local_scales", "local_mask"]
        if self.global_feats is None:
            fields_needed.remove("global_feats")
            
        local_storage = h5py.File(self.local_hdf5_path, 'r')
        query_tensor = self.get_tensor_dict(index, local_storage, fields=fields_needed)
        
        topk_inds = self.nn_inds[index, :]  # top-1600
        topk_inds = topk_inds[topk_inds != index]  # remove query itself
        gallery_labels = [self.targets[ind] for ind in topk_inds]
        positive_mask = [1 if l == landmark_id else 0 for l in gallery_labels]
        
        pos_this_sample = 0
        filtered_topk_inds = []
        filtered_positive_mask = []
        
        # BUG FIX: shuffle_indices should only touch the topk defined in the dataset
        if self.shuffle_indices:
            # to avoid choosing the same positive when max_pos_per_topk is small
            selected_indices = torch.randperm(self.topk).tolist()
            prev_topk_inds = [topk_inds[i] for i in selected_indices]
            prev_positive_mask = [positive_mask[i] for i in selected_indices]
            topk_inds[:self.topk] = prev_topk_inds
            positive_mask[:self.topk] = prev_positive_mask
        
        for ind, pos in zip(topk_inds, positive_mask):
            if pos == 0 or pos_this_sample < self.max_pos_per_topk:  # not a positive or still have room for positive
                filtered_topk_inds.append(ind)
                filtered_positive_mask.append(pos)
                pos_this_sample += pos
            
            if len(filtered_topk_inds) == self.topk:
                break
        
        if index in filtered_topk_inds:
            print("Warning: query image is in the topk candidates. Watch it!")
            
        if len(filtered_topk_inds) != self.topk:  # still not enough, usually impossible
            raise ValueError("Not enough candidates for this query image.")
        
        gallery_tensors = [self.get_tensor_dict(ind, local_storage, fields=fields_needed) 
                           for ind in filtered_topk_inds]
        positive_indices = [i for i, pos in enumerate(filtered_positive_mask) if pos == 1]
        
        # skip negative mining
        
        if len(positive_indices) < self.min_pos_per_topk:
            pos_compensator = self.min_pos_per_topk - len(positive_indices)
            positive_inds = [ni for ni in self.label_indices[landmark_id] if ni != index]
            pos_compensator = min(pos_compensator, len(positive_inds))  # avoid out of index
            index_to_replace = random.sample(positive_inds, pos_compensator)
            
            for i in index_to_replace:
                picked_idx = random.choice(range(self.topk))
                while filtered_positive_mask[picked_idx] == 1:
                    picked_idx = random.choice(range(self.topk))  # avoid replacing positive samples
                gallery_tensors[picked_idx] = self.get_tensor_dict(i, local_storage, fields=fields_needed)
                filtered_positive_mask[picked_idx] = 1
                
        ret = {
            "query_local_features": query_tensor["local_feats"],
            "query_local_scales": query_tensor["local_scales"],
            "query_local_mask": query_tensor["local_mask"],
            "gallery_local_features": [t["local_feats"] for t in gallery_tensors],
            "gallery_local_scales": [t["local_scales"] for t in gallery_tensors],
            "gallery_local_mask": [t["local_mask"] for t in gallery_tensors],
            "positive_mask": filtered_positive_mask,
            "label": landmark_id,
            "index": index
        }
        
        if self.global_feats is not None:
            ret["query_global_features"] = query_tensor["global_feats"]
            ret["gallery_global_features"] = [t["global_feats"] for t in gallery_tensors]
        
        return ret
    
# for other query-db splitted dataset, use QueryGalleryExtractiveDatasetHDF5
# labels needed for evaluation; compatible with rox, rpar and gldv2-test & gldv2-val
class QueryGalleryExtractiveDatasetHDF5(Dataset):
    def __init__(self, query_local_hdf5_path, db_local_hdf5_path, nn_inds_path, 
                 dataset_name,  # choose from 'roxford5k(+1m)', 'rparis6k(+1m)', 'gldv2-test'
                 gnd_pkl_path, 
                 distractor_nn_inds_path=None,  # for +1m 
                 distractor_hdf5_path=None,  # for +1m
                 distractor_global_pt_path=None,  # for +1m
                 query_global_pt_path=None, 
                 db_global_pt_path=None,
                 global_dim=768,
                 num_desc=49, topk=100, max_pos_per_topk=100, min_pos_per_topk=0, 
                 shuffle_indices=True):
        super(QueryGalleryExtractiveDatasetHDF5, self).__init__()
        self.query_local_hdf5_path = query_local_hdf5_path
        query_data_size, data_topk, local_dim = h5py.File(query_local_hdf5_path, 'r')['features'].shape
        print(f"Query data size: {query_data_size}, topk: {data_topk}, local_dim: {local_dim}")
        # record query_data_size for ratio expanding
        self.query_data_size = query_data_size
        
        self.db_local_hdf5_path = db_local_hdf5_path
        gallery_data_size, data_topk, local_dim = h5py.File(db_local_hdf5_path, 'r')['features'].shape
        print(f"DB data size: {gallery_data_size}, topk: {data_topk}, local_dim: {local_dim}")
        
        if query_global_pt_path is not None:
            self.query_global_feats = np.memmap(query_global_pt_path, 
                                        dtype=np.float32, mode='r', shape=(query_data_size, global_dim))
            assert db_global_pt_path is not None, "db_global_pt_path must be provided if query_global_pt_path is provided."
            self.db_global_feats = np.memmap(db_global_pt_path, 
                                        dtype=np.float32, mode='r', shape=(gallery_data_size, global_dim))
        else:
            self.query_global_feats = None
            self.db_global_feats = None
            
        self.nn_inds = pickle_load(nn_inds_path)  # [2, data_size, data_topk]
        assert self.nn_inds.shape[1] == query_data_size, f"nn_inds shape mismatch! nn_inds: {self.nn_inds.shape}, data_size: {query_data_size}"
        # just keep the inds-dim in training
        if isinstance(self.nn_inds, torch.Tensor):
            self.nn_inds = self.nn_inds[1, :, :].long().numpy()
        else:
            self.nn_inds = self.nn_inds[1, :, :].astype(int)
        
        # load query and db samples
        self.dataset_name = dataset_name
        self.gnd = pickle_load(gnd_pkl_path)
        
        if self.dataset_name.endswith('+1m'):
            # load distractor
            pass
        
        # record other parameters
        self.num_desc = num_desc
        self.topk = topk
        self.max_pos_per_topk = max_pos_per_topk
        self.min_pos_per_topk = min_pos_per_topk
        
        self.shuffle_indices = shuffle_indices or self.max_pos_per_topk == 1
        if self.shuffle_indices:
            print("Warning: shuffle_indices is on. max_pos_per_topk will sample from shuffled indices.")
            
    def __len__(self):
        return len(self.gnd['qimlist'])  # number of queries; will be expanded
            
    def get_tensor_dict(self, index, local_storage, 
                        fields=["global_feats", "local_feats", "local_scales", "local_mask"], 
                        is_query=True):
        ret_dict = dict()
        if is_query:
            index = index % self.query_data_size  # for expanding query data
        if "global_feats" in fields:
            ret_dict["global_feats"] = torch.from_numpy(self.query_global_feats[index] if is_query else self.db_global_feats[index])
        if "local_feats" in fields:
            local_feats = torch.from_numpy(local_storage['features'][index, :self.num_desc])
            # get meta data from local storage previous 5 columns
            local_meta = local_feats[:, :5]
            ret_dict["local_feats"] = local_feats[:, 5:]
            if "local_scales" in fields:
                ret_dict["local_scales"] = local_meta[:, 2].long()
            if "local_mask" in fields:  # 0 -> valid, 1 -> invalid
                ret_dict["local_mask"] = local_meta[:, 3].long()
        return ret_dict
    
    def __getitem__(self, index):
        # img_path, landmark_id, _, _ = self.samples[index]  # (img_path, landmark_id, w, h / h, w)
        fields_needed = ["global_feats", "local_feats", "local_scales", "local_mask"]
        if self.query_global_feats is None:
            fields_needed.remove("global_feats")
            
        query_local_storage = h5py.File(self.query_local_hdf5_path, 'r')
        query_tensor = self.get_tensor_dict(index, query_local_storage, 
                                            fields=fields_needed, is_query=True)
        
        topk_inds = self.nn_inds[index % self.query_data_size, :]
        # topk_inds = topk_inds[topk_inds != index]  # remove query itself
        # gallery_labels = [self.targets[ind] for ind in topk_inds]
        # positive_mask = [1 if l == landmark_id else 0 for l in gallery_labels]
        if self.dataset_name.startswith('r'):  # roxford5k or rparis6k
            positive_db_ids = {
                'easy': set(self.gnd['gnd'][index % self.query_data_size]['easy']), 
                'hard': set(self.gnd['gnd'][index % self.query_data_size]['hard'])
            }
            positive_db_ids = positive_db_ids['easy'].union(positive_db_ids['hard'])
        elif self.dataset_name.startswith('gldv2'):
            positive_db_ids = set(self.gnd['gnd'][index % self.query_data_size]['ok'])
        else:
            raise ValueError("Unknown dataset name.")

        positive_mask = [1 if i in positive_db_ids else 0 for i in topk_inds]
        
        pos_this_sample = 0
        filtered_topk_inds = []
        filtered_positive_mask = []
        
        if self.shuffle_indices:
            selected_indices = torch.randperm(self.topk).tolist()
            prev_topk_inds = [topk_inds[i] for i in selected_indices]
            prev_positive_mask = [positive_mask[i] for i in selected_indices]
            topk_inds[:self.topk] = prev_topk_inds
            positive_mask[:self.topk] = prev_positive_mask
        
        for ind, pos in zip(topk_inds, positive_mask):
            if pos == 0 or pos_this_sample < self.max_pos_per_topk:  # not a positive or still have room for positive
                filtered_topk_inds.append(ind)
                filtered_positive_mask.append(pos)
                pos_this_sample += pos
            
            if len(filtered_topk_inds) == self.topk:
                break
        
        # if index in filtered_topk_inds:
        #     print("Warning: query image is in the topk candidates. Watch it!")
            
        if len(filtered_topk_inds) != self.topk:  # still not enough, usually impossible
            raise ValueError("Not enough candidates for this query image.")
        
        db_local_storage = h5py.File(self.db_local_hdf5_path, 'r')
        
        gallery_tensors = [self.get_tensor_dict(ind, db_local_storage, fields=fields_needed, is_query=False) 
                           for ind in filtered_topk_inds]
        positive_indices = [i for i, pos in enumerate(filtered_positive_mask) if pos == 1]
        
        # skip negative mining
        
        if len(positive_indices) < self.min_pos_per_topk:
            pos_compensator = self.min_pos_per_topk - len(positive_indices)
            # positive_inds = [ni for ni in self.label_indices[landmark_id] if ni != index]
            positive_db_inds = list(positive_db_ids)
            pos_compensator = min(pos_compensator, len(positive_db_inds))  # avoid out of index
            index_to_replace = random.sample(positive_db_inds, pos_compensator)
            
            for i in index_to_replace:
                picked_idx = random.choice(range(self.topk))
                while filtered_positive_mask[picked_idx] == 1:
                    picked_idx = random.choice(range(self.topk))  # avoid replacing positive samples
                gallery_tensors[picked_idx] = self.get_tensor_dict(i, db_local_storage, fields=fields_needed, 
                                                                   is_query=False)
                filtered_positive_mask[picked_idx] = 1
                
        ret = {
            "query_local_features": query_tensor["local_feats"],
            "query_local_scales": query_tensor["local_scales"],
            "query_local_mask": query_tensor["local_mask"],
            "gallery_local_features": [t["local_feats"] for t in gallery_tensors],
            "gallery_local_scales": [t["local_scales"] for t in gallery_tensors],
            "gallery_local_mask": [t["local_mask"] for t in gallery_tensors],
            "positive_mask": filtered_positive_mask,
            "index": index % self.query_data_size
        }
        
        if self.query_global_feats is not None:
            ret["query_global_features"] = query_tensor["global_feats"]
            ret["gallery_global_features"] = [t["global_feats"] for t in gallery_tensors]
        
        return ret

        
@dataclass
class ExtractiveCollatorCache:
    topk: int = 100
    shuffle_pos: bool = False
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(instances)
        has_global_features = 'query_global_features' in instances[0]
        # we can help to unsqueeze here instead in the model
        if has_global_features:
            batched_query_global_features = torch.stack([i['query_global_features'] for i in instances], dim=0).unsqueeze(1)
            batched_gallery_global_features = []
            # expected batched_gallery_global_features: [bs, topk, 1, global_dim]
        
        # batched_query_global_features: [bs, 1, global_dim]
        batched_query_local_features = torch.stack([i['query_local_features'] for i in instances], dim=0)
        # batched_query_local_features: [bs, num_descriptor, local_dim]
        # dim projection happens inside the model
        batched_query_local_scales = torch.stack([i['query_local_scales'] for i in instances], dim=0)
        # batched_query_local_scales: [bs, num_descriptor]; in model should be [bs, num_descriptor, hidden_size]
        batched_query_local_mask = torch.stack([i['query_local_mask'] for i in instances], dim=0)
        # batched_query_local_mask: [bs, num_descriptor]; in model should be [bs, num_descriptor, hidden_size]
        
        batched_gallery_local_features = []
        # expected batched_gallery_local_features: [bs, topk, num_descriptor, local_dim]
        batched_gallery_local_scales = []
        # expected batched_gallery_local_scales: [bs, topk, num_descriptor, 1]
        batched_gallery_local_mask = []
        # expected batched_gallery_local_mask: [bs, topk, num_descriptor]
        batched_positive_mask = []
        # expected batched_positive_mask: [bs, topk]
        
        for i in range(batch_size):
            if has_global_features:
                gallery_global_features = instances[i]['gallery_global_features']  # list of k's [global_dim]
            # assert self.topk == len(gallery_global_features), f"Expected {self.topk} gallery features, got {len(gallery_global_features)} instead!"
            gallery_local_features = instances[i]['gallery_local_features']
            gallery_local_scales = instances[i]['gallery_local_scales']
            gallery_local_mask = instances[i]['gallery_local_mask']
            positive_mask = instances[i]['positive_mask']
            if self.shuffle_pos:
                indices = torch.randperm(len(gallery_local_features)).tolist()
                if has_global_features:
                    gallery_global_features = [gallery_global_features[j] for j in indices]
                gallery_local_features = [gallery_local_features[j] for j in indices]
                gallery_local_scales = [gallery_local_scales[j] for j in indices]
                gallery_local_mask = [gallery_local_mask[j] for j in indices]
                positive_mask = [positive_mask[j] for j in indices]
            if has_global_features:
                batched_gallery_global_features.append(
                    torch.stack(gallery_global_features, dim=0).unsqueeze(1)  # [topk, 1, global_dim]
                )
            batched_gallery_local_features.append(
                torch.stack(gallery_local_features, dim=0)  # [topk, num_descriptor, local_dim]
            )
            batched_gallery_local_scales.append(
                torch.stack(gallery_local_scales, dim=0)  # [topk, num_descriptor]
            )
            batched_gallery_local_mask.append(
                torch.stack(gallery_local_mask, dim=0) # [topk, num_descriptor]
            )
            batched_positive_mask.append(
                torch.tensor(positive_mask, dtype=torch.long) #  [topk]
            )
        
        ret = {
            # 'query_global_features': batched_query_global_features,  # torch.Size([4, 512, 1])
            'query_local_features': batched_query_local_features,  # torch.Size([4, 49, 1024])
            'query_local_scales': batched_query_local_scales,
            'query_local_mask': batched_query_local_mask,
            # 'gallery_global_features': torch.stack(batched_gallery_global_features, dim=0),
            'gallery_local_features': torch.stack(batched_gallery_local_features, dim=0),
            'gallery_local_scales': torch.stack(batched_gallery_local_scales, dim=0),
            'gallery_local_mask': torch.stack(batched_gallery_local_mask, dim=0),
            'positive_mask': torch.stack(batched_positive_mask, dim=0)
        }
        if has_global_features:
            ret['query_global_features'] = batched_query_global_features
            ret['gallery_global_features'] = torch.stack(batched_gallery_global_features, dim=0)
        
        return ret
    
# local_hdf5_path, nn_inds_path, sample_txt_path, global_pt_path=None, global_dim=768,
#                  num_desc=49, topk=100, max_pos_per_topk=100, min_pos_per_topk=0, 
# Normal building data modules
def build_extractive_hdf5_data_modules(
    local_hdf5_path: str, 
    nn_ids_path: str, 
    sample_txt_path: str, 
    is_training: bool = True,
    topk: int = 100, 
    max_pos_per_topk: int = 100, 
    min_pos_per_topk: int = 5,
    shuffle_pos: bool = True, 
    num_descriptors: int = 49,  # *added for GLDv2* how many local descriptors to use
    global_dim: int = 768, 
    global_pt_path: Optional[str] = None,  # do not use global features if None
    shuffle_indices: bool = False,  # change after Nov. 6 23:46
    num_samples: Optional[int] = None
):
    dataset = GLDv2ExtractiveDatasetHDF5(
        local_hdf5_path=local_hdf5_path, 
        nn_inds_path=nn_ids_path, 
        sample_txt_path=sample_txt_path, 
        global_pt_path=global_pt_path,
        global_dim=global_dim,
        num_desc=num_descriptors,
        topk=topk,
        max_pos_per_topk=max_pos_per_topk,
        min_pos_per_topk=min_pos_per_topk,
        shuffle_indices=shuffle_indices, 
        num_samples=num_samples
    )
    
    collator = ExtractiveCollatorCache(
        topk=topk, 
        shuffle_pos=shuffle_pos
    )
    
    if is_training:
        return dict(
            train_dataset=dataset, 
            data_collator=collator
        )
    else:
        return dict(
            eval_dataset=dataset, 
            data_collator=collator
        )