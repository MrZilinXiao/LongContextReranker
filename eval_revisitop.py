# grid-search script that examine multiple checkpoints on ROxford5k and RParis6k
# on 11/11, move the sliding window re-ranking here.
# replace **HARDCODED PATHS** with your own paths to extracted features
import os
import torch
from common_utils.eval_data_utils_hdf5 import build_revisitop_eval_data_modules
from models.longformer_universal_v2 import ExtractiveLongformerForCache
from transformers import AutoConfig
import argparse
from safetensors.torch import load_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from common_utils.eval_utils import compute_map, compute_rectangular_ap
from itertools import product
from copy import deepcopy
import numpy as np
import pickle

config = AutoConfig.from_pretrained("./longformer-base-5120")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default='./ames_logs')
    parser.add_argument("--global_nn_name", type=str, default='superglobal')
    # 01/27: add for rebuttal stage based on custom nn_inds
    parser.add_argument("--custom_nn_path", type=str, default=None)
    parser.add_argument("--custom_nn_file", type=str, default=None)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--sliding_reranking_type", type=str, default=None)
    parser.add_argument("--sliding_topk", type=int, default=400)
    parser.add_argument("--sliding_stride", type=int, default=50)  # 50 / 25
    # choose from: start-to-end, end-to-start, start-to-end-to-start, end-to-start-to-end
    parser.add_argument("--with_1m", action='store_true')
    parser.add_argument("--with_hard", action='store_true')
    parser.add_argument("--gldv2_val", action='store_true')
    # designate alpha and temp for with_1m & sliding re-ranking to save time
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--temp", type=float, default=None)
    # 25/02/06: add f100r50 option: 100 features with 50 topk window
    parser.add_argument("--num_descriptors", type=int, default=48)  # 48 / 98 / 23 for local features
    parser.add_argument("--topk", type=int, default=100)  # 100 / 50 / 200 for the size of long-context reranker
    # 25/02/06: add bottleneck_dim option for fair comparison
    parser.add_argument("--bottleneck_dim", type=int, default=None)
    parser.add_argument("--force_linear", action='store_true')
    return parser.parse_args()

def compute_metrics(ranks, gnd, kappas, dataset_name='default'):
    if dataset_name == 'gldv2-test':
        ranks = ranks[:100]
        map, aps, _, _ = compute_map(ranks[:, :-750], gnd[:-750], ap_f=compute_rectangular_ap)
        priv_map, priv_aps, _, _ = compute_map(ranks[:, -750:], gnd[-750:], ap_f=compute_rectangular_ap)
        comb_map, comb_aps, _, _ = compute_map(ranks, gnd, ap_f=compute_rectangular_ap)

        out = {'pub_map': np.around(map*100, decimals=3), 'priv_map': np.around(priv_map*100, decimals=3),
               'combined': np.around(comb_map*100, decimals=3)}
        info = f'>> {dataset_name}: mAP public: {out["pub_map"]:.3f}, private: {out["priv_map"]:.3f}, combined: {out["combined"]:.3f}'
        
    else:    
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['easy']])
            g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
            gnd_t.append(g)
        mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, kappas)

        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk']])
            gnd_t.append(g)
        mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, kappas)

        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
            gnd_t.append(g)
        mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, kappas)

        out = {
            'E_map': np.around(mapE*100, decimals=3),
            'M_map': np.around(mapM*100, decimals=3),
            'H_map': np.around(mapH*100, decimals=3),
            'E_mp':  np.around(mprE*100, decimals=3),
            'M_mp':  np.around(mprM*100, decimals=3),
            'H_mp':  np.around(mprH*100, decimals=3),
            'apsE': apsE.tolist(),
            'apsM': apsM.tolist(),
            'apsH': apsH.tolist(),
        }
        map = (mapM + mapH) / 2
        aps = np.concatenate((apsM, apsH))
        info = f'>> {dataset_name}: mAP M: {out["M_map"]}, H: {out["H_map"]}'

    print(info)
    return out, map, aps

def sliding_ranks_adjust(curr_indices, this_nn_inds, this_nn_sims, local_sims, st, ed, 
                         alpha=0.5, temp=0.5):
    # adjusting the ranks with extracted st:ed nn_inds and nn_sims
    raw_sim = local_sims.to(this_nn_inds.device)
    select_indices = deepcopy(curr_indices[st:ed])
    
    for k, a, t in product([9999, ], [alpha], [temp]):
        global_sim = this_nn_sims[select_indices].clone()
        s = 1. / (1. + torch.exp(-t * raw_sim))
        # global_sim changes when doing sliding_reranking
        s = a * global_sim + (1 - a) * s
        
        closest_dists, indices = torch.sort(s, dim=-1, descending=True)
        # update select_indices
        # print(f"select_indices: {select_indices}")
        # print(indices.shape)
        select_indices = [select_indices[i] for i in indices.tolist()]
        curr_indices[st:ed] = select_indices
            
    return curr_indices

def compute_metrics_with_sliding(out_ranks, gnd, ks, output_fp=None, dataset_name='default'):
    out = dict()
    for (k, a, t), ranks in out_ranks.items():
        ranks = ranks.cpu().data.numpy().T
        metrics, score, _ = compute_metrics(ranks, gnd, ks, dataset_name=dataset_name)
        out[(k, a, t)] = score
        if dataset_name == 'gldv2-test':
            print(f'>> alpha: {a}, temp: {t}, pub_map: {metrics["pub_map"]}, priv_map: {metrics["priv_map"]}, combined: {metrics["combined"]}', file=output_fp)
            print(f'>> alpha: {a}, temp: {t}, pub_map: {metrics["pub_map"]}, priv_map: {metrics["priv_map"]}, combined: {metrics["combined"]}')
        else:
            print(f'>> alpha: {a}, temp: {t}, M_map: {metrics["M_map"]}, H_map: {metrics["H_map"]}', file=output_fp)
            print(f'>> alpha: {a}, temp: {t}, M_map: {metrics["M_map"]}, H_map: {metrics["H_map"]}')
        
    out_dict = max(out, key=out.get)
    print(f"Best: {out[out_dict]} with {out_dict}")
    return out

def global_local_ensemble(gnd, nn_inds, nn_sims, local_sims, alpha=(0.), temp=(0.5, ), top_k=(100, ), 
                          ks=(1,5,10), output_fp=None, dataset_name='default'):
    assert output_fp is not None, "output_fp must be provided"
    raw_sim = local_sims.to(nn_sims.device)
    
    out = {}
    # alpha = [0. ] + alpha
    for k, a, t in product(top_k, alpha, temp):
        s = 1. / (1. + torch.exp(-t * raw_sim))
        global_sim = nn_sims[:, :k].clone()
        s = a * global_sim + (1 - a) * s  # global with a, local with 1-a
        
        closest_dists, indices = torch.sort(s, dim=-1, descending=True)
        closest_indices = torch.gather(nn_inds, -1, indices)
        ranks = deepcopy(nn_inds)
        ranks[:, :k] = deepcopy(closest_indices)
        ranks = ranks.cpu().data.numpy().T
        metrics, score, _ = compute_metrics(ranks, gnd, ks, dataset_name=dataset_name)
        out[(k, a, t)] = score
        if dataset_name == 'gldv2-test':
            print(f'>> alpha: {a}, temp: {t}, pub_map: {metrics["pub_map"]}, priv_map: {metrics["priv_map"]}, combined: {metrics["combined"]}', file=output_fp)
            print(f'>> alpha: {a}, temp: {t}, pub_map: {metrics["pub_map"]}, priv_map: {metrics["priv_map"]}, combined: {metrics["combined"]}')
        else:
            print(f'>> alpha: {a}, temp: {t}, M_map: {metrics["M_map"]}, H_map: {metrics["H_map"]}', file=output_fp)
            print(f'>> alpha: {a}, temp: {t}, M_map: {metrics["M_map"]}, H_map: {metrics["H_map"]}')
    
    out_dict = max(out, key=out.get)
    print(f"Best: {out[out_dict]} with {out_dict}")
    
    return out

def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    
    checkpoint_path = args.checkpoint_path
    checkpoint_name = os.path.basename(checkpoint_path)
    checkpoint_ext = "default"
    if 'checkpoint' in checkpoint_name:
        checkpoint_ext = checkpoint_name
        checkpoint_name = checkpoint_path.split('/')[-2]
        
    print(f"checkpoint_name: {checkpoint_name}")
        
    if 'dinov2' in checkpoint_name:
        local_dim = global_dim = 768
    elif 'cvnet' in checkpoint_name:
        local_dim = 1024
        global_dim = 2048
        
    if 'scratch_256' in checkpoint_name:
        config.num_hidden_layers = 6
        config.attention_window = [512] * config.num_hidden_layers
        config.num_attention_heads = 8
        config.hidden_size = 256
        config.intermediate_size = 1024
        config.max_position_embeddings = 5122
        
    global_offset = 1
    if 'noglobal' in checkpoint_name:
        global_dim = config.hidden_size  # in this way global_to_lm layer will not be inited
        global_offset = 0
        
    model = ExtractiveLongformerForCache(
        language_model=config, 
        local_dim=local_dim, 
        global_dim=global_dim,
        query_global_attention=True, 
        pos_type='absolute',
        num_layers_kept=6 if 'small' in checkpoint_name else None, 
        bottleneck_dim=args.bottleneck_dim,   # if given, set an intermediate bottleneck layer
        num_features=args.num_descriptors + global_offset, 
        force_linear=args.force_linear
    ).to(device)
    model.eval()
    
    missing_keys, unexpected_keys = load_model(model, os.path.join(checkpoint_path, "model.safetensors"))
    print(missing_keys, unexpected_keys)
    hard_splits = [False, ]
    if args.with_hard:
        hard_splits.append(True)
    
    for hard_split in hard_splits:
        datasets = ['gldv2-test'] if args.gldv2_val else ['roxford5k', 'rparis6k']
        if args.with_1m:
            # datasets.extend(['roxford5k+1m', 'rparis6k+1m'])
            datasets = ['roxford5k+1m', 'rparis6k+1m']   # in rebuttal, do eval on +1m only
        
        # rebuttal: consider roxford5k+1m only when cvnet is used
        if 'cvnet' in checkpoint_name and not args.gldv2_val:
            datasets = ['roxford5k+1m', 'rparis6k+1m']
        
        for dataset_name in datasets:
            alpha_list = list(np.linspace(0, 1, 11)) if args.alpha is None else [args.alpha]
            temp_list = list(np.linspace(0, 1, 11)) if args.temp is None else [args.temp]
            
            local_desc_name = None
            if 'dinov2' in checkpoint_name:
                local_desc_name = 'dinov2'
                # **HARDCODED PATHS**
                query_local_hdf5_path = f"/scratch/zx51/ames/my_data/{dataset_name.replace('+1m', '')}/dinov2_query_local.hdf5"
                db_local_hdf5_path = f"/scratch/zx51/ames/my_data/{dataset_name}/dinov2_gallery_local.hdf5"
                nn_inds_path = f"/scratch/zx51/ames/data/{dataset_name}/nn_{args.global_nn_name}.pkl"
                gnd_pkl_path = f"/scratch/zx51/ames/my_data/{dataset_name.replace('+1m', '')}/gnd_{dataset_name.replace('+1m', '')}.pkl"
                query_global_pt_path = f"/scratch/zx51/ames/my_data/{dataset_name.replace('+1m', '')}/dinov2_query_global.pt"
                db_global_pt_path = f"/scratch/zx51/ames/my_data/{dataset_name}/dinov2_gallery_global.pt"
            elif 'cvnet' in checkpoint_name:
                local_desc_name = 'cvnet'
                # **HARDCODED PATHS**
                query_local_hdf5_path = f"/scratch/zx51/ames/my_data/{dataset_name.replace('+1m', '')}/cvnet_query_local.hdf5"
                db_local_hdf5_path = f"/scratch/zx51/ames/my_data/{dataset_name}/cvnet_gallery_local.hdf5"
                nn_inds_path = f"/scratch/zx51/ames/data/{dataset_name}/nn_{args.global_nn_name}.pkl"
                gnd_pkl_path = f"/scratch/zx51/ames/my_data/{dataset_name.replace('+1m', '')}/gnd_{dataset_name.replace('+1m', '')}.pkl"
                query_global_pt_path = f"/scratch/zx51/ames/my_data/{dataset_name.replace('+1m', '')}/cvnet_query_global.pt"
                db_global_pt_path = f"/scratch/zx51/ames/my_data/{dataset_name}/cvnet_gallery_global.pt"
                
            # override nn_inds_path with custom_nn
            if args.custom_nn_path is not None:
                nn_inds_path = os.path.join(args.custom_nn_path, f"{dataset_name}/{args.custom_nn_file}.pkl")
                
            if not os.path.exists(nn_inds_path):
                nn_inds_path = nn_inds_path.replace('/data/', '/my_data/')
            
            if 'noglobal' in checkpoint_name:
                query_global_pt_path = db_global_pt_path = None
                
            if dataset_name == 'gldv2-test':
                # assert 'dinov2' in checkpoint_name, "Only Dinov2 is supported for GLDv2"
                # **HARDCODED PATHS**
                query_local_hdf5_path = f"/scratch/zx51/ames/my_data/gldv2-test/{local_desc_name}_local.hdf5"
                db_local_hdf5_path = f"/scratch/zx51/ames/my_data/gldv2-index/{local_desc_name}_local.hdf5"
                query_global_pt_path = f"/scratch/zx51/ames/my_data/gldv2-test/{local_desc_name}_global.pt"
                db_global_pt_path = f"/scratch/zx51/ames/my_data/gldv2-index/{local_desc_name}_global.pt"
                
            # build dataloader
            data_module = build_revisitop_eval_data_modules(
                dataset_name=dataset_name,
                query_local_hdf5_path=query_local_hdf5_path,
                db_local_hdf5_path=db_local_hdf5_path,
                nn_inds_path=nn_inds_path,
                gnd_pkl_path=gnd_pkl_path,
                query_global_pt_path=query_global_pt_path,
                db_global_pt_path=db_global_pt_path,
                num_desc=args.num_descriptors,
                topk=args.topk if args.sliding_reranking_type is None else args.sliding_topk,
                global_dim=global_dim,
                local_dim=local_dim,
                hard_split=hard_split, 
            )
            dataloader = DataLoader(data_module['eval_dataset'], batch_size=1, 
                                    shuffle=False, num_workers=0, drop_last=False, 
                                    collate_fn=data_module['data_collator'])
            query_size = data_module['eval_dataset'].nn_inds.shape[0]
            window_size = args.topk  # num of images to rank
            
            # split the sliding window re-ranking here
            if args.sliding_reranking_type is None:
                modes = ['max-sum', 'max-start', 'max-end']
                # local_sims = torch.zeros((len(modes), query_size, 100), dtype=torch.float32)
                local_sims = torch.zeros((len(modes), query_size, args.topk), dtype=torch.float32)
                
                for i, batch in enumerate(tqdm(dataloader)):
                    visualized_positive_mask = batch.pop("visualized_positive_mask")
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(device)
                    query_index = batch.pop('index')
                    assert query_index[0] == i, f"query_index: {query_index[0]}, i: {i}"  # sanity check when bs=1
                    output = model(**batch, with_softmax=False)
                    # local_sims[i] = output.unpack_log['probs'][mode].squeeze(0)
                    
                    for j, mode in enumerate(modes):
                        local_sims[j, i] = output.unpack_log['probs'][mode].squeeze(0)
                        
                for j, mode in enumerate(modes):
                    output_folder = os.path.join(args.output_dir, f"{dataset_name}{'_hard' if data_module['eval_dataset'].hard_split else ''}")
                    os.makedirs(output_folder, exist_ok=True)
                    output_path = os.path.join(output_folder, f'{checkpoint_name}_{checkpoint_ext}_{mode}{"_final" if args.alpha is not None else ""}{"_" + args.custom_nn_file if args.custom_nn_file is not None else ""}.txt')
                    if args.global_nn_name != 'superglobal':
                        output_path += f"_{args.global_nn_name}"
                    
                    if dataset_name.endswith('+1m'):
                        alpha_list = [0.5, ] if args.alpha is None else [args.alpha]
                        temp_list = [0.5, ] if args.temp is None else [args.temp]
                    
                    with open(output_path, 'w') as f:
                        _ = global_local_ensemble(data_module['eval_dataset'].gnd, data_module['eval_dataset'].nn_inds, data_module['eval_dataset'].nn_sims, 
                                                  local_sims[j], 
                                                alpha=alpha_list, 
                                                temp=temp_list, top_k=(args.topk, ), ks=(1,5,10), output_fp=f,
                                                dataset_name=dataset_name)
            else:
                # sliding-window re-ranking can not afford grid-search as
                # each permutation costs one forward pass
                mode = 'max-end'
                sliding_topk = args.sliding_topk
                sliding_stride = args.sliding_stride
                local_sims = torch.zeros((query_size, args.topk), dtype=torch.float32)
                alpha_list = [0.5, ] if args.alpha is None else [args.alpha]
                temp_list = [0.5, ] if args.temp is None else [args.temp]
                ranks_list = []  # for final eval; need torch.cat before eval
                
                for i, batch in enumerate(tqdm(dataloader)):
                    visualized_positive_mask = batch.pop("visualized_positive_mask")
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(device)
                    query_index = batch.pop('index')
                    assert query_index[0] == i, f"query_index: {query_index[0]}, i: {i}"
                    
                    # if args.sliding_reranking_type == 'start-to-end':
                    window_left_st = 0
                    window_right_ed = sliding_topk - sliding_stride
                    step = sliding_stride
                    start_to_end_range = list(range(window_left_st, window_right_ed, step))
                    # elif args.sliding_reranking_type == 'end-to-start':
                    window_left_st = sliding_topk - window_size
                    window_right_ed = -sliding_stride
                    step = -sliding_stride
                    end_to_start_range = list(range(window_left_st, window_right_ed, step))
                    
                    curr_indices = list(range(sliding_topk))
                    
                    if args.sliding_reranking_type == 'start-to-end':
                        range_selected = start_to_end_range
                    elif args.sliding_reranking_type == 'end-to-start':
                        range_selected = end_to_start_range
                    elif args.sliding_reranking_type == 'start-to-end-to-start':
                        range_selected = start_to_end_range + end_to_start_range[1:]
                    elif args.sliding_reranking_type == 'end-to-start-to-end':
                        range_selected = end_to_start_range + start_to_end_range[1:]
                        
                    if i == 0:
                        print(f"forward passes needed: {len(range_selected)}: {range_selected}")
                    
                    for curr_start in range_selected:
                        curr_end = curr_start + window_size
                        select_indices = curr_indices[curr_start: curr_end]
                        curr_batch = {k: v[:, select_indices] for k, v in batch.items() if k.startswith('gallery')}
                        curr_batch.update({k: v for k, v in batch.items() if not k.startswith('gallery')})
                        
                        output = model(**curr_batch, with_softmax=False)
                        local_sims[i] = output.unpack_log['probs'][mode].squeeze(0)
                        # local_sims[i] /= 50. if mode == 'max-sum' else 1.
                        local_sims[i] /= float(model.language_model.num_local_features + 1) if mode == 'max-sum' else 1.
                        
                        curr_indices = sliding_ranks_adjust(curr_indices, 
                                                            data_module['eval_dataset'].nn_inds[i],
                                                            data_module['eval_dataset'].nn_sims[i],
                                                            local_sims[i], curr_start, curr_end, 
                                                            alpha=alpha_list[0], temp=temp_list[0])
                    # after the loop, use curr_indices to rebuild the ranks?
                    this_ranks = data_module['eval_dataset'].nn_inds[i].tolist()
                    this_ranks[:len(curr_indices)] = [this_ranks[i] for i in curr_indices]
                    ranks_list.append(this_ranks)
                
                out_ranks = {
                    (9999, alpha_list[0], temp_list[0]): torch.tensor(ranks_list, dtype=torch.long)
                }
                # adjust should be completed
                output_folder = os.path.join(args.output_dir, f"{dataset_name}{'_hard' if data_module['eval_dataset'].hard_split else ''}")
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(output_folder, f'{args.sliding_reranking_type}_top{sliding_topk}_stride{sliding_stride}_{checkpoint_name}_{checkpoint_ext}{"_final" if args.alpha is not None else ""}{"_" + args.custom_nn_file if args.custom_nn_file is not None else ""}.txt')
                if args.global_nn_name != 'superglobal':
                    output_path += f"_{args.global_nn_name}"
                with open(output_path, 'w') as f:
                    _ = compute_metrics_with_sliding(out_ranks, data_module['eval_dataset'].gnd, ks=(1,5,10), output_fp=f, 
                                                     dataset_name=dataset_name)
                        

if __name__ == "__main__":
    main()