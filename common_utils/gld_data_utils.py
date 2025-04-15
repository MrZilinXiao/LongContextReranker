# CAUTION: GLDv2 and Revisited datasets share this file to construct data modules
# Read from cache tensors and send to model with `query_features` and `gallery_features`
# first we need a top-1000 nn_ids and a verifier function
# take advantage of sop_data_utils_v2.py
from torch.utils.data import Dataset
import os.path as osp
import torch
from common_utils.serialization import pickle_load
from dataclasses import dataclass
import random
import numpy as np
import os
import pandas as pd
from typing import Sequence, Dict, Any, Tuple, List, Union, Optional
from tqdm import tqdm 
import bisect
from DOLG.revisitop.dataset import configdataset
from DOLG.evaler.util import walkfile
import json

IGNORE_IMAGE_NAMES = {  # corrupted images in rparis6k; just skip. 
    'rparis6k': [
        'paris_louvre_000136',
        'paris_louvre_000146',
        'paris_moulinrouge_000422',
        'paris_museedorsay_001059',
        'paris_notredame_000188',
        'paris_pantheon_000284',
        'paris_pantheon_000960',
        'paris_pantheon_000974',
        'paris_pompidou_000195',
        'paris_pompidou_000196',
        'paris_pompidou_000201',
        'paris_pompidou_000467',
        'paris_pompidou_000640',
        'paris_sacrecoeur_000299',
        'paris_sacrecoeur_000330',
        'paris_sacrecoeur_000353',
        'paris_triomphe_000662',
        'paris_triomphe_000833',
        'paris_triomphe_000863',
        'paris_triomphe_000867'
    ]
}

class RevisitedCacheDataset(Dataset):
    def __init__(self, 
                 dataset='roxford5k', 
                 cache_name='dolg_r50', 
                 dir_main='/scratch/zx51/revisitop/data/datasets/roxford5k', 
                 return_img_path=False, 
                 return_img_name=False, 
                 verify=False, split: str = 'qimlist'
        ) -> None:  # 1m distractors should use `imlist`.
        self.cfg = configdataset(dataset, dir_main)
        # save cached features in the same directory as the dataset
        self.cache_root = os.path.join(dir_main, cache_name)
        # os.makedirs(self.cache_root, exist_ok=True)
        # self.img_path = os.path.join(dir_main, 'jpg')
        # self.img_paths = [path for path in walkfile(self.img_path) if all(
        #     ignore_name not in path for ignore_name in IGNORE_IMAGE_NAMES.get(dataset, [])
        # )]
        self.img_path = os.path.join(dir_main, 'jpg')
        self.tensor_path = os.path.join(dir_main, cache_name)
        
        # self.tensor_paths = [path for path in walkfile(self.tensor_path) if all(
        #     ignore_name not in path for ignore_name in IGNORE_IMAGE_NAMES.get(dataset, [])
        # ) and path.endswith(".pt")]
        if split == 'all':
            self.tensor_paths = [path for path in walkfile(self.tensor_path) if all(
                ignore_name not in path for ignore_name in IGNORE_IMAGE_NAMES.get(dataset, [])
            ) and path.endswith(".pt")]
        else:
            self.tensor_paths = [
                os.path.join(self.tensor_path, f"{img_name.split('/')[-1].replace('.jpg', '')}.pt") for img_name in self.cfg[split]
                if img_name not in IGNORE_IMAGE_NAMES.get(dataset, [])
            ]
        
        self.return_img_path = return_img_path
        self.return_img_name = return_img_name
        if verify:
            self.verify()
        
    def verify(self):
        missing_tensor_path = []
        for tensor_path in tqdm(self.tensor_paths, desc="Verifying cache tensors..."):
            if not os.path.exists(tensor_path):
                missing_tensor_path.append(tensor_path)
        print(f"Missing tensors: {missing_tensor_path}")
        assert len(missing_tensor_path) == 0, f"Warning: {len(missing_tensor_path)} tensors missing!"
        
    def __len__(self):
        return len(self.tensor_paths)
    
    def __getitem__(self, index):
        tensor_path = self.tensor_paths[index]
        try:
            tensor_dict = torch.load(tensor_path)
        except Exception as e:
            print(f"Error loading tensor from {tensor_path}: {e}")
            raise e
        
        if self.return_img_name:
            return tensor_dict, osp.basename(tensor_path).replace(".pt", "")
        elif not self.return_img_path:
            return tensor_dict
        else:  # for visualization
            return tensor_dict, os.path.join(
                self.img_path, osp.basename(tensor_path).replace(".pt", ".jpg")
            )

# only for evaluation usage. since mAP online metric is not gatherible, 
# you have to evaluate the whole dataset to get the final mAP
# since we don't train on revisited dataset, sampling is not needed.
# 11/06 TODO: make it accept global_only
class RevisitedExtractiveDataset(Dataset):
    def __init__(self, 
                 dataset='roxford5k', 
                 cache_name='delg_r50', 
                 dir_main='/scratch/zx51/revisitop/data/datasets/roxford5k',
                 nn_ids_path='/scratch/zx51/revisitop/data/datasets/roxford5k/dolg_r50/nn_ids.npy',
                 num_descriptors=48,  
                 # below are shared between Revisited and GLDv2. 
                 topk=100, 
                 # below not used in Revisited dataset as we don't do training
                 max_pos_per_topk=100, min_pos_per_topk=0,
                 toy_dataset=False, shuffle_indices=False, 
                 scales: List[float] = [0.25, 0.3535, 0.5, 0.7071, 1.0, 1.4142, 2.],
                 distractor_dir_main: Optional[str] = None, 
                 return_indices=False, local_only=False, 
                 ) -> None:
        self.cfg = configdataset(dataset, dir_main)
        self.cache_root = os.path.join(dir_main, cache_name)
        self.cache_nn_ids = np.load(nn_ids_path)
        self.local_only = local_only
        
        if self.local_only:
            print("Warning: local_only is on for RevisitedExtractiveDataset")
        
        if self.cache_nn_ids.dtype != np.int32:
            self.cache_nn_ids = self.cache_nn_ids.astype(np.int32)

        self.query_paths = [
            os.path.join(self.cache_root, f"{img_name}.pt") for img_name in self.cfg['qimlist']
            if img_name not in IGNORE_IMAGE_NAMES.get(dataset, [])
        ]
        
        self.gallery_paths = [
            os.path.join(self.cache_root, f"{img_name}.pt") for img_name in self.cfg['imlist']
            if img_name not in IGNORE_IMAGE_NAMES.get(dataset, [])
        ]
        
        distractor_paths = None
        if distractor_dir_main is not None:
            distractor_cfg = configdataset('revisitop1m', distractor_dir_main)
            distractor_paths = [
                os.path.join(distractor_dir_main, cache_name, f"{img_name.split('/')[-1].replace('.jpg', '')}.pt") 
                for img_name in distractor_cfg['imlist']
            ]
            # TODO: verify direct extension is okay. 
            self.gallery_paths.extend(distractor_paths)
            
        self.topk = topk
        self.max_pos_per_topk = max_pos_per_topk
        self.min_pos_per_topk = min_pos_per_topk
        assert self.min_pos_per_topk <= self.max_pos_per_topk <= self.topk, f"Invalid pos_per_topk settings! Must satisfy min <= max <= topk, {self.min_pos_per_topk} <= {self.max_pos_per_topk} <= {self.topk}"
        self.num_descriptors = num_descriptors
        self.scales = scales
        
        self.toy_dataset = toy_dataset
        self.return_indices = return_indices
        if self.toy_dataset:
            print("Toy dataset mode is on; gallery will always contain an identical query image")
            
        self.shuffle_indices = shuffle_indices or self.max_pos_per_topk == 1
        if self.shuffle_indices:
            print("Warning: shuffle_indices is on. max_pos_per_topk will sample from shuffled indices.")
        
    def __len__(self):
        return len(self.query_paths)
        
    def get_tensor_dict(self, tensor_path, 
                   fields=["global_feats", "local_feats", "local_scales", "local_mask"]):
        # tensor_path = os.path.join(self.cache_root, f"{img_name}.pt")
        tensor_dict = torch.load(tensor_path)
        ret_dict = dict()
        for k, v in tensor_dict.items():
            if k in fields:
                ret_dict[k] = v
                # k: str
                if k.startswith("local"):  # strip every local features to top num_descriptors
                    ret_dict[k] = v[:self.num_descriptors]
                    
        return ret_dict
    
    def get_scales_ids(self, scales):
        return torch.tensor([bisect.bisect_right(self.scales, s) for s in scales], dtype=torch.int32) - 1
    
    def get_visualized_positive_mask(self, topk_ids, index):
        positive_gallery_ids = {
            'easy': set(self.cfg['gnd'][index]['easy']), 
            'hard': set(self.cfg['gnd'][index]['hard']),
            # sometimes there is 'ok'
        }
        
        positive_mask = [0] * len(topk_ids)
        for i, idx in enumerate(topk_ids):
            if idx in positive_gallery_ids['easy']:
                positive_mask[i] = 1
            elif idx in positive_gallery_ids['hard']:
                positive_mask[i] = 2
                
        return positive_mask
    
    def __getitem__(self, index) -> Any:
        fields_needed = ["global_feats", "local_feats", "local_scales", "local_mask"]
        if self.local_only:
            fields_needed.remove("global_feats")
            
        query_tensor = self.get_tensor_dict(self.query_paths[index], fields=fields_needed)
        topk_inds = self.cache_nn_ids[index, :self.topk]
        # posibly has distractor index in it. 
        # topk_inds = topk_inds[topk_inds != index][:self.topk]  # do not exclude as revisited has no overlap.
        
        # no need to provide positive mask as eval happens in the main loop
        gallery_tensors = [self.get_tensor_dict(self.gallery_paths[i], fields=fields_needed) for i in topk_inds]
        
        positive_gallery_ids = {
            'easy': set(self.cfg['gnd'][index]['easy']), 
            'hard': set(self.cfg['gnd'][index]['hard']),
        }
        positive_gallery_ids = positive_gallery_ids['easy'].union(positive_gallery_ids['hard'])
        positive_mask = [1 if i in positive_gallery_ids else 0 for i in topk_inds]
        
        additional_fields = {}
        if self.return_indices:
            additional_fields['index'] = index
            additional_fields['gallery_indices'] = topk_inds
            
        if not self.local_only:
            additional_fields["query_global_features"] = query_tensor["global_feats"]
            additional_fields["gallery_global_features"] = [t["global_feats"] for t in gallery_tensors]
        
        return {
            # 'query_global_features': query_tensor["global_feats"],
            'query_local_features': query_tensor["local_feats"],
            'query_local_scales': self.get_scales_ids(query_tensor["local_scales"]),
            'query_local_mask': query_tensor["local_mask"],
            # stacking in the worker should be faster
            # 'gallery_global_features': [t["global_feats"] for t in gallery_tensors],  
            'gallery_local_features': [t["local_feats"] for t in gallery_tensors],
            'gallery_local_scales': [self.get_scales_ids(t["local_scales"]) for t in gallery_tensors],
            'gallery_local_mask': [t["local_mask"] for t in gallery_tensors],
            # 'gallery_img_names': gallery_img_names,
            'positive_mask': positive_mask,
            'visualized_positive_mask': self.get_visualized_positive_mask(topk_inds, index),
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
            additional_fields['gallery_indices'] = torch.stack(
                [torch.LongTensor(i['gallery_indices']) for i in instances], dim=0
            )
        
        ret = {
            # 'query_img_name': [i['query_img_name'] for i in instances],
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
        
def build_revisited_cache_data_modules(
    dataset='roxford5k', 
    cache_name='delg_r50', 
    dir_main='/scratch/zx51/revisitop/data/datasets/roxford5k',
    nn_ids_path='/scratch/zx51/revisitop/data/datasets/roxford5k/delg_r50/nn_ids.npy',
    num_descriptors=48,
    # below are shared between Revisited and GLDv2. 
    topk=100, 
    # below for with1M evaluation only
    distractor_dir_main: Optional[str] = None,
    return_indices: bool = False, 
    local_only: bool = False,
):
    dataset = RevisitedExtractiveDataset(
        dataset=dataset,
        cache_name=cache_name,
        dir_main=dir_main,
        nn_ids_path=nn_ids_path,
        num_descriptors=num_descriptors,
        topk=topk, 
        distractor_dir_main=distractor_dir_main, 
        return_indices=return_indices, 
        local_only=local_only
    )
    collator = EvalExtractiveCollatorCache(return_indices=return_indices)
    
    return dict(
        eval_dataset=dataset, 
        data_collator=collator
    )


# Only for producing topk nn_ids and data integrity verification
# not for training
class GLDv2CacheDataset(Dataset):
    def __init__(self, data_root, cache_root, 
                 split='train', verify=False, return_img_path=False):
        self.data_root = data_root  # still need a dataroot for metadata
        self.cache_root = cache_root
        
        self.split = split
        self.train = self.split == 'train'
        self.csv = 'train_clean.csv' if self.train else f'{self.split}.csv'
        
        self.return_img_path = return_img_path
        
        if os.path.exists(os.path.join(self.data_root, 'metadata', self.csv)):
            self.data = pd.read_csv(os.path.join(self.data_root, 'metadata', self.csv))
        else:
            raise FileNotFoundError(f"File {self.csv} not found in {os.path.join(self.data_root, 'metadata')}")
            
        self.image_ids = []
        for _, row in self.data.iterrows():
            for img_id in row['images'].split():
                self.image_ids.append((img_id, row['landmark_id']))
                
        if verify:
            self.verify()
        
        # used only for GLDv2 negative generation
        self.labels = np.array([label for _, label in self.image_ids], dtype=np.int32)
            
    def verify(self):
        missing_image_list = []
        for _, row in tqdm(self.data.iterrows(), desc="Verifying cache tensors..."):
            image_ids = row['images'].split()
            for img_id in image_ids:
                cache_path = os.path.join(self.cache_root, self.split, img_id[0], img_id[1], img_id[2], f"{img_id}.pt")
                if not os.path.exists(cache_path):
                    missing_image_list.append(cache_path)

        if missing_image_list:
            raise RuntimeError(f"Warning: {len(missing_image_list)} tensors missing!")
            
        else:
            print("All images are present!")
            
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        img_id, landmark_id = self.image_ids[index]
        tensor_path = os.path.join(self.cache_root, self.split, img_id[0], img_id[1], img_id[2], f"{img_id}.pt")
        tensor_dict = torch.load(tensor_path)
        
        feats, label = {k: v for (k, v) in tensor_dict.items() if k != "label"}, tensor_dict["label"]
        # label should be matching the landmark_id
        assert label.item() == landmark_id, f"Label mismatch: {label} vs {landmark_id}"
        
        if not self.return_img_path:
            return feats, label
        else:  # used for visualization
            return feats, label, os.path.join(self.data_root, self.split, f"{img_id[0]}/{img_id[1]}/{img_id[2]}/{img_id}.jpg")

class GLDv2ExtractiveDataset(GLDv2CacheDataset):
    def __init__(self, data_root, cache_root, nn_ids_path, num_descriptors=49,
                 split='train', topk=100, max_pos_per_topk=10, min_pos_per_topk=5,
                 toy_dataset=False, shuffle_indices=False, 
                 scales: List[float] = [0.25, 0.3535, 0.5, 0.7071, 1.0, 1.4142, 2.], 
                 # added on 05/05: enforce negative training
                 neg_nn_ids_path: Optional[str] = None, local_only=False,
                 ):
        # prepare required meta data
        super().__init__(data_root, cache_root, split=split, verify=False)
        self.local_only = local_only  # add 11/06 for ablation on global
        
        if self.local_only:
            print("Warning: local_only is on for GLDv2ExtractiveDataset")
        
        self.counts = np.bincount(self.labels)  # occurance of each label
        self.cache_nn_ids = np.load(nn_ids_path)
        
        if self.cache_nn_ids.dtype != np.int32:
            self.cache_nn_ids = self.cache_nn_ids.astype(np.int32)
        
        if len(self.cache_nn_ids.shape) == 1:
            self.cache_nn_ids = self.cache_nn_ids.reshape(-1, 1000)
            
        self.neg_cache_nn_ids = None
        if neg_nn_ids_path is not None:
            self.neg_cache_nn_ids = np.load(neg_nn_ids_path)
        
        # make a numpy equivalent
        self.label_indices = [
            np.where(self.labels == i)[0].tolist()
            for i in range(len(self.counts))
        ]
        # self.label_indices = np.array(label_indices, dtype=np.int32)
        
        self.topk = topk
        self.max_pos_per_topk = max_pos_per_topk
        self.min_pos_per_topk = min_pos_per_topk
        assert self.min_pos_per_topk <= self.max_pos_per_topk <= self.topk, f"Invalid pos_per_topk settings! Must satisfy min <= max <= topk, {self.min_pos_per_topk} <= {self.max_pos_per_topk} <= {self.topk}"
        self.num_descriptors = num_descriptors
        self.scales = scales
        
        self.toy_dataset = toy_dataset
        if self.toy_dataset:
            print("Toy dataset mode is on; gallery will always contain an identical query image")
            
        self.shuffle_indices = shuffle_indices or self.max_pos_per_topk == 1
        if self.shuffle_indices:
            print("Warning: shuffle_indices is on. max_pos_per_topk will sample from shuffled indices.")
    
    def get_tensor_dict(self, img_id, 
                   fields=["global_feats", "local_feats", "local_scales", "local_mask"]):
        tensor_path = os.path.join(self.cache_root, self.split, img_id[0], img_id[1], img_id[2], f"{img_id}.pt")
        tensor_dict = torch.load(tensor_path)
        ret_dict = dict()
        for k, v in tensor_dict.items():
            if k in fields:
                ret_dict[k] = v
                # k: str
                if k.startswith("local"):  # strip every local features to top num_descriptors
                    ret_dict[k] = v[:self.num_descriptors]
                    
        return ret_dict
    
    def get_scales_ids(self, scales):
        # if scales == self.scales, then return would be range(len(self.scales))
        # return np.array([bisect.bisect_right(self.scales, s) for s in scales], dtype=np.int32) - 1
        return torch.tensor([bisect.bisect_right(self.scales, s) for s in scales], dtype=torch.int32) - 1
    
    def __getitem__(self, index):
        # needed keys for each image: "global_feats", "local_feats"
        # "local_scales", "local_mask"
        img_id, current_label = self.image_ids[index]
        fields_needed = ["global_feats", "local_feats", "local_scales", "local_mask"]
        if self.local_only:
            fields_needed.remove("global_feats")
        
        query_tensor = self.get_tensor_dict(img_id, fields=fields_needed)
        
        topk_inds = self.cache_nn_ids[index, :]
        topk_inds = topk_inds[topk_inds != index]  # excluded the query itself

        gallery_labels = [self.labels[i] for i in topk_inds]
        positive_mask = [1 if l == current_label else 0 for l in gallery_labels]
        
        pos_this_sample = 0
        filtered_topk_inds = []
        filtered_positive_mask = []
        
        # to avoid choosing the same positive when max_pos_per_topk is small
        if self.shuffle_indices:
            selected_indices = torch.randperm(len(topk_inds)).tolist()
            topk_inds = [topk_inds[i] for i in selected_indices]
            positive_mask = [positive_mask[i] for i in selected_indices]
        
        for ind, pos in zip(topk_inds, positive_mask):
            if pos == 0 or pos_this_sample < self.max_pos_per_topk:  # not a positive or still have room for positive
                filtered_topk_inds.append(ind)
                filtered_positive_mask.append(pos)
                pos_this_sample += pos
            
            if len(filtered_topk_inds) == self.topk:
                break
            
        if index in filtered_topk_inds:
            print("Warning: query image is in the topk candidates. Watch it!")
        
        # there is a chance that no enough negative in top-k when max_pos_per_topk is small
        if len(filtered_topk_inds) != self.topk:
            remain_neg_count = self.topk - len(filtered_topk_inds)
            assert self.neg_cache_nn_ids is not None, "Negative cache nn_ids is not provided!"
            # there should be some strategy when selecting these negative samples
            neg_topk_inds = self.neg_cache_nn_ids[index, :].tolist()
            neg_remain_inds = random.sample(neg_topk_inds, k=remain_neg_count)
            filtered_topk_inds.extend(neg_remain_inds)
            filtered_positive_mask.extend([0] * remain_neg_count)
            
            # the only reason of triggering this is that too many positives!
            # print(f"Warning: not enough topk candidates for {img_id}, got {len(filtered_topk_inds)} instead of {self.topk}!")
            # print(f"len of topk_inds: {len(topk_inds)}: {topk_inds}")
            # print(f"len of positive_mask: {len(positive_mask)}: {positive_mask}")
            # raise RuntimeError
            
        gallery_tensors = [self.get_tensor_dict(self.image_ids[i][0], 
                                                fields=fields_needed) for i in filtered_topk_inds]
        positive_indices = [i for i, p in enumerate(filtered_positive_mask) if p == 1]
        
        if len(positive_indices) < self.min_pos_per_topk:
            pos_compensator = self.min_pos_per_topk - len(positive_indices)
            positive_inds = [ni for ni in self.label_indices[current_label] if ni != index]
            pos_compensator = min(pos_compensator, len(positive_inds))
            index_to_replace = random.sample(positive_inds, k=pos_compensator)  # no place-back sampling
            for i in index_to_replace:
                picked_idx = random.choice(range(self.topk))
                while filtered_positive_mask[picked_idx] == 1:  # rejection sampling
                    picked_idx = random.choice(range(self.topk))
                gallery_tensors[picked_idx] = self.get_tensor_dict(self.image_ids[i][0], fields=fields_needed)
                filtered_positive_mask[picked_idx] = 1
                
        if self.toy_dataset:
            positive_indices = [i for i, p in enumerate(filtered_positive_mask) if p == 1]  # recompute, as it may gets updated
            picked_idx = random.choice(positive_indices)
            gallery_tensors[picked_idx] = query_tensor
            filtered_positive_mask[picked_idx] = 2   # unnecessary; but useful for debugging as we see which one is replaced
        
        additional_fields = {}
        if not self.local_only:
            additional_fields["query_global_features"] = query_tensor["global_feats"]
            additional_fields["gallery_global_features"] = [t["global_feats"] for t in gallery_tensors]
        
        return {
            # 'query_global_features': query_tensor["global_feats"],
            'query_local_features': query_tensor["local_feats"],
            'query_local_scales': self.get_scales_ids(query_tensor["local_scales"]),
            'query_local_mask': query_tensor["local_mask"],
            # stacking in the worker should be faster
            # 'gallery_global_features': [t["global_feats"] for t in gallery_tensors],  
            'gallery_local_features': [t["local_feats"] for t in gallery_tensors],
            'gallery_local_scales': [self.get_scales_ids(t["local_scales"]) for t in gallery_tensors],
            'gallery_local_mask': [t["local_mask"] for t in gallery_tensors],
            'positive_mask': filtered_positive_mask,
            'label': current_label,
            'index': index, 
            **additional_fields
        }
        
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
        
def build_extractive_cache_data_modules(
    data_path: str, 
    split_file: str,   # **for compatibility only; not used**
    cache_nn_ids: str, 
    cache_path: str = None,   # *added for GLDv2*
    crop_size: int = 224,   # **for compatibility only; not used**
    is_training: bool = True,
    topk: int = 100, 
    max_pos_per_topk: int = 10, 
    min_pos_per_topk: int = 0,
    shuffle_pos: bool = True, 
    toy_dataset: bool = False,
    no_train_transform: bool = False,  # **for compatibility only; not used**
    num_descriptors: int = 49,  # *added for GLDv2* how many local descriptors to use
    neg_cache_nn_ids: Optional[str] = None,
    local_only: bool = False
):
    assert cache_path is not None, "cache_path is required!"
    dataset = GLDv2ExtractiveDataset(
        data_root=data_path,
        cache_root=cache_path,
        nn_ids_path=cache_nn_ids,
        num_descriptors=num_descriptors,
        split='train' if is_training else 'test',
        topk=topk, 
        max_pos_per_topk=max_pos_per_topk,
        min_pos_per_topk=min_pos_per_topk,
        toy_dataset=toy_dataset, 
        neg_nn_ids_path=neg_cache_nn_ids,
        local_only=local_only
    )
    collator = ExtractiveCollatorCache(shuffle_pos=shuffle_pos)
    
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