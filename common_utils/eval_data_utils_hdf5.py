# HDF5 version of RevisitedExtractiveDataset in `gld_data_utils.py` for compability to evaluate with 
# CVNet & DINOv2 features. 
from torch.utils.data import Dataset
from common_utils.serialization import pickle_load
from dataclasses import dataclass
import numpy as np
import h5py
import torch
from typing import Sequence, Dict


class RevisitopExtractiveEvalDataset(Dataset):
    def __init__(self, dataset_name: str, 
                 query_local_hdf5_path: str, 
                 db_local_hdf5_path: str,
                 nn_inds_path: str,
                 gnd_pkl_path: str, 
                 query_global_pt_path=None, 
                 db_global_pt_path=None,
                 global_dim=768,
                 local_dim=768,
                 num_desc=49,
                 topk=100, 
                 # below for distractor settings
                 distractor_hdf5_path=None,  # for +1m
                 distractor_global_pt_path=None,  # for +1m
                 # below for debug: return some metadata
                 return_indices=True,
                 hard_split=False
                 ):
        super(RevisitopExtractiveEvalDataset, self).__init__()
        self.query_local_hdf5_path = query_local_hdf5_path
        query_data_size, data_topk, local_dim = h5py.File(query_local_hdf5_path, 'r')['features'].shape
        print(f"Query data size: {query_data_size}, topk: {data_topk}, local_dim: {local_dim}")
        # record query_data_size for ratio expanding
        self.query_data_size = query_data_size
        
        self.db_local_hdf5_path = db_local_hdf5_path
        db_data_size, data_topk, local_dim = h5py.File(db_local_hdf5_path, 'r')['features'].shape
        print(f"DB data size: {db_data_size}, topk: {data_topk}, local_dim: {local_dim}")
        
        if query_global_pt_path is not None:
            self.query_global_feats = np.memmap(
                query_global_pt_path, dtype=np.float32, mode='r', shape=(query_data_size, global_dim), 
            )
            assert db_global_pt_path is not None, "db_global_pt_path should be provided"
            self.db_global_feats = np.memmap(
                db_global_pt_path, dtype=np.float32, mode='r', shape=(db_data_size, global_dim), 
            )
        else:
            self.query_global_feats = None
            self.db_global_feats = None
            
        self.nn_inds = pickle_load(nn_inds_path)
        assert self.nn_inds.shape[1] == query_data_size, f"nn_inds shape mismatch! nn_inds: {self.nn_inds.shape}, data_size: {query_data_size}"

        # load ground-truth
        self.dataset_name = dataset_name
        self.gnd = pickle_load(gnd_pkl_path)['gnd']
        # keep an old copy of nn_inds for visualization
        self.old_nn_inds = self.nn_inds[1].clone().long()

        if 'junk' in self.gnd[0]:
            for i in range(self.nn_inds.shape[1]):
                if hard_split:
                    junk_ids = self.gnd[i]['junk'] + self.gnd[i]['easy']
                else:
                    junk_ids = self.gnd[i]['junk']
                is_junk = np.in1d(self.nn_inds[1, i].long(), junk_ids)
                self.nn_inds[:, i] = torch.cat((self.nn_inds[:, i, ~is_junk], self.nn_inds[:, i, is_junk]), dim=1)

        self.nn_sims = self.nn_inds[0]
        self.nn_inds = self.nn_inds[1].long()
        
        self.num_desc = num_desc
        self.topk = topk
        self.return_indices = return_indices
        self.hard_split = hard_split
    
    def __len__(self):
        return len(self.gnd)
    
    def get_tensor_dict(self, index, local_storage, 
                        fields=["global_feats", "local_feats", "local_scales", "local_mask"], 
                        is_query=True):
        ret_dict = dict()
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
    
    def get_visualized_positive_mask(self, topk_ids, index):
        topk_ids = topk_ids.tolist()
        if 'easy' in self.gnd[index]:
            positive_gallery_ids = {
                'easy': set(self.gnd[index]['easy']),
                'hard': set(self.gnd[index]['hard']),
            }
        else:  # sometimes there is 'ok'
            positive_gallery_ids = {
                'ok': set(self.gnd[index]['ok']),
            }
        
        # print(f"Positive gallery ids: {positive_gallery_ids}")
        # print(f"Topk ids: {topk_ids}")
        positive_mask = [0] * len(topk_ids)
        
        for i, idx in enumerate(topk_ids):
            if idx in positive_gallery_ids.get('easy', set()):
                positive_mask[i] = 1
            elif idx in positive_gallery_ids.get('hard', set()):
                positive_mask[i] = 2
            elif idx in positive_gallery_ids.get('ok', set()):
                positive_mask[i] = 3
                
        return positive_mask
    
    def __getitem__(self, index):
        fields_needed = ["global_feats", "local_feats", "local_scales", "local_mask"]
        if self.query_global_feats is None:
            fields_needed.remove("global_feats")
        
        query_local_storage = h5py.File(self.query_local_hdf5_path, 'r')
        query_tensor = self.get_tensor_dict(index, query_local_storage, 
                                            fields=fields_needed, is_query=True)
        
        topk_inds = self.nn_inds[index, :self.topk]  # don't remove self. 
        viz_topk_inds = self.old_nn_inds[index, :self.topk]
        # eval does not need shuffling for now.
        db_local_storage = h5py.File(self.db_local_hdf5_path, 'r')
        gallery_tensors = [self.get_tensor_dict(ind, db_local_storage, fields=fields_needed, is_query=False) for ind in topk_inds]
        # DO NOT need positive_mask
        
        additional_fields = dict()
        if self.return_indices:
            additional_fields['index'] = index
            additional_fields['gallery_indices'] = topk_inds
            
        if self.query_global_feats is not None:
            additional_fields["query_global_features"] = query_tensor["global_feats"]
            additional_fields["gallery_global_features"] = [t["global_feats"] for t in gallery_tensors]
            
        return {
            # 'query_global_features': query_tensor["global_feats"],
            'query_local_features': query_tensor["local_feats"],
            'query_local_scales': query_tensor["local_scales"],
            'query_local_mask': query_tensor["local_mask"],
            # stacking in the worker should be faster
            # 'gallery_global_features': [t["global_feats"] for t in gallery_tensors],  
            'gallery_local_features': [t["local_feats"] for t in gallery_tensors],
            'gallery_local_scales': [t["local_scales"] for t in gallery_tensors],
            'gallery_local_mask': [t["local_mask"] for t in gallery_tensors],
            # 'gallery_img_names': gallery_img_names,
            'visualized_positive_mask': self.get_visualized_positive_mask(viz_topk_inds, index),
            # 'label': current_label,
            **additional_fields
        }
        
@dataclass
class EvalExtractiveCollatorCache:
    return_indices: bool = False
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(instances)
        has_global_features = 'query_global_features' in instances[0]
        # we can help to unsqueeze here instead in the model
        if has_global_features:
            batched_query_global_features = torch.stack([i['query_global_features'] for i in instances], dim=0).unsqueeze(1)
            batched_gallery_global_features = []
        # batched_query_global_features: [bs, 1, global_dim]
        batched_query_local_features = torch.stack([i['query_local_features'] for i in instances], dim=0)
        # batched_query_local_features: [bs, num_descriptor, local_dim]
        # dim projection happens inside the model
        batched_query_local_scales = torch.stack([i['query_local_scales'] for i in instances], dim=0)
        # batched_query_local_scales: [bs, num_descriptor]; in model should be [bs, num_descriptor, hidden_size]
        batched_query_local_mask = torch.stack([i['query_local_mask'] for i in instances], dim=0)
        # batched_query_local_mask: [bs, num_descriptor]; in model should be [bs, num_descriptor, hidden_size]
        
        batched_gallery_global_features = []
        # expected batched_gallery_global_features: [bs, topk, 1, global_dim]
        batched_gallery_local_features = []
        # expected batched_gallery_local_features: [bs, topk, num_descriptor, local_dim]
        batched_gallery_local_scales = []
        # expected batched_gallery_local_scales: [bs, topk, num_descriptor, 1]
        batched_gallery_local_mask = []
        # expected batched_gallery_local_mask: [bs, topk, num_descriptor]
        
        for i in range(batch_size): 
            if has_global_features:
                gallery_global_features = instances[i]['gallery_global_features']  # list of k's [global_dim]
            # assert self.topk == len(gallery_global_features), f"Expected {self.topk} gallery features, got {len(gallery_global_features)} instead!"
            gallery_local_features = instances[i]['gallery_local_features']
            gallery_local_scales = instances[i]['gallery_local_scales']
            gallery_local_mask = instances[i]['gallery_local_mask']
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
        
        additional_fields = {}
        
        if 'visualized_positive_mask' in instances[0]:
            additional_fields['visualized_positive_mask'] = torch.stack(
                [torch.LongTensor(i['visualized_positive_mask']) for i in instances], dim=0
            )
        
        if self.return_indices:
            additional_fields['index'] = [i['index'] for i in instances]
            # additional_fields['gallery_indices'] = torch.stack(
            #     [torch.LongTensor(i['gallery_indices']) for i in instances], dim=0
            # )
        
        ret = {
            # 'query_global_features': batched_query_global_features,  # torch.Size([4, 512, 1])
            'query_local_features': batched_query_local_features,  # torch.Size([4, 49, 1024])
            'query_local_scales': batched_query_local_scales,
            'query_local_mask': batched_query_local_mask,
            # 'gallery_global_features': torch.stack(batched_gallery_global_features, dim=0),
            'gallery_local_features': torch.stack(batched_gallery_local_features, dim=0),
            'gallery_local_scales': torch.stack(batched_gallery_local_scales, dim=0),
            'gallery_local_mask': torch.stack(batched_gallery_local_mask, dim=0),
            **additional_fields
        }
        
        if has_global_features:
            ret['query_global_features'] = batched_query_global_features
            ret['gallery_global_features'] = torch.stack(batched_gallery_global_features, dim=0)
        
        return ret
    
def build_revisitop_eval_data_modules(
    dataset_name: str,
    query_local_hdf5_path: str,
    db_local_hdf5_path: str,
    nn_inds_path: str,
    gnd_pkl_path: str,
    query_global_pt_path=None,
    db_global_pt_path=None,
    global_dim=768,
    local_dim=768,
    num_desc=49,
    topk=100,
    return_indices=True,
    hard_split=False
):
    dataset = RevisitopExtractiveEvalDataset(
        dataset_name=dataset_name,
        query_local_hdf5_path=query_local_hdf5_path,
        db_local_hdf5_path=db_local_hdf5_path,
        nn_inds_path=nn_inds_path,
        gnd_pkl_path=gnd_pkl_path,
        query_global_pt_path=query_global_pt_path,
        db_global_pt_path=db_global_pt_path,
        global_dim=global_dim,
        local_dim=local_dim,
        num_desc=num_desc,
        topk=topk,
        return_indices=return_indices,
        hard_split=hard_split
    )
    
    collator = EvalExtractiveCollatorCache(return_indices=return_indices)
    return dict(
        eval_dataset=dataset,
        data_collator=collator
    )