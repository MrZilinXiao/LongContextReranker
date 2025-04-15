import torch
from tqdm import tqdm
import torch.nn.functional as F
from models.resnet import ResNet, extract_feat_res_pycls
import os

RF = 291.0
STRIDE = 16.0
PADDING = 145.0


def generate_coordinates(h, w):
    '''generate coorinates
    Returns: [h*w, 2] FloatTensor
    '''
    x = torch.floor(torch.arange(0, float(w * h)) / w)
    y = torch.arange(0, float(w)).repeat(h)

    coord = torch.stack([x, y], dim=1)
    return coord


def calculate_receptive_boxes(height, width, rf, stride, padding):
    coordinates = generate_coordinates(height, width)
    point_boxes = torch.cat([coordinates, coordinates], dim=1)
    bias = torch.FloatTensor([-padding, -padding, -padding + rf - 1, -padding + rf - 1])
    rf_boxes = stride * point_boxes + bias
    return rf_boxes


def non_maxima_suppression_2d(heatmap):
    hmax = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    keep = (heatmap == hmax)
    return keep


def calculate_keypoint_centers(rf_boxes):
    '''compute feature centers, from receptive field boxes (rf_boxes).
    Args:
        rf_boxes: [N, 4] FloatTensor.
    Returns:
        centers: [N, 2] FloatTensor.
    '''
    xymin = torch.index_select(rf_boxes, dim=1, index=torch.LongTensor([0, 1]).cuda())
    xymax = torch.index_select(rf_boxes, dim=1, index=torch.LongTensor([2, 3]).cuda())
    return (xymax + xymin) / 2.0


@torch.no_grad()
def extract_feature(model, test_loader):
    # with torch.no_grad():
    img_feats = [[] for i in range(3)]

    for i, (im_list, scale_list) in enumerate(tqdm(test_loader, mininterval=10)):
        for idx in range(len(im_list)):
            im_list[idx] = im_list[idx].cuda()
            desc = model(im_list[idx])[0]
            if len(desc.shape) == 1:
                desc.unsqueeze_(0)
            desc = F.normalize(desc, p=2, dim=1)
            img_feats[idx].append(desc.detach().cpu())

    for idx in range(len(img_feats)):
        img_feats[idx] = torch.cat(img_feats[idx], dim=0)
        if len(img_feats[idx].shape) == 1:
            img_feats[idx].unsqueeze_(0)

    img_feats_agg = F.normalize(
        torch.mean(torch.cat([img_feat.unsqueeze(0) for img_feat in img_feats], dim=0), dim=0), 
        p=2, dim=1)
    img_feats_agg = img_feats_agg.cpu().numpy()

    return img_feats_agg

@torch.no_grad()
def extract_feature_single(model, im_list):
    img_feats = [[] for _ in range(len(im_list))]
    # our gldv2_extract_dataset has no batchify (collate_fn) in dataloader, 
    # so we need to build a batch dim for each image.

    for idx in range(len(im_list)):
        im_list[idx] = im_list[idx].cuda()  # Move image to GPU
        if len(im_list[idx].shape) == 3:  # no batch dimension due to no dataloader
            im_list[idx] = im_list[idx].unsqueeze(0)
        desc = model(im_list[idx])[0]       # Get feature descriptor from the model
        if len(desc.shape) == 1:
            desc = desc.unsqueeze(0)        # Ensure batch dimension
        desc = F.normalize(desc, p=2, dim=1)  # L2 normalize
        img_feats[idx].append(desc.detach().cpu())  # Detach and move to CPU

    for idx in range(len(img_feats)):
        img_feats[idx] = torch.cat(img_feats[idx], dim=0)  # Concatenate all descriptors
        if len(img_feats[idx].shape) == 1:
            img_feats[idx] = img_feats[idx].unsqueeze(0)  # Ensure batch dimension

    # Aggregate across scales and normalize
    img_feats_agg = F.normalize(
        torch.mean(torch.cat([img_feat.unsqueeze(0) for img_feat in img_feats], dim=0), dim=0), 
        p=2, dim=1
    )
    img_feats_agg = img_feats_agg.cpu().numpy()

    return img_feats_agg


def get_local(local_features, local_weights, scales, topk=700, 
              return_raw_scale=False):
    # scales: [tensor([0.7071], dtype=torch.float64), tensor([0.7071], dtype=torch.float64)], 
    # [tensor([1.], dtype=torch.float64), tensor([1.], dtype=torch.float64)], 
    # [tensor([1.4142], dtype=torch.float64), tensor([1.4142], dtype=torch.float64)]]
    feature_list, feature_w, scale_limits, boxes, keeps = [], [], [], [], []
    last_scale_limit = 0
    raw_scale_list = []

    for j, local_feature in enumerate(local_features):
        w = local_weights[j]

        keep = torch.ones_like(w).bool().squeeze(0).squeeze(0)
        # calculate receptive field boxes.
        rf_boxes = calculate_receptive_boxes(
            height=local_feature.size(2),
            width=local_feature.size(3),
            rf=RF,
            stride=STRIDE,
            padding=PADDING)

        # re-projection back to original image space.
        # print(scales[j])  # 0.7071, 0.7071
        # print(type(scales[j]))
        rf_boxes = rf_boxes / torch.stack(scales[j], dim=1).repeat(1, 2)
        boxes.append(rf_boxes.cuda()[keep.flatten().nonzero().squeeze(1)])

        local_feature = local_feature.squeeze(0).permute(1, 2, 0)[keep]
        feature_list.append(local_feature)
        feature_w.append(w.squeeze(0).squeeze(0)[keep])
        last_scale_limit += local_feature.shape[0]
        scale_limits.append(last_scale_limit)
        # problem: scale for x, y might be different for DINOv2?
        # raw_scale_list.append(scales[j][0])
        raw_scale_list.extend([scales[j][0]] * local_feature.shape[0])

    feats = torch.cat(feature_list, dim=0)
    boxes = torch.cat(boxes, dim=0)
    norms = torch.cat(feature_w, dim=0)
    raw_scales = torch.cat(raw_scale_list, dim=0).to(norms.device)
    
    seq_len = min(feats.shape[0], topk)
    weights, ids = torch.topk(norms, k=seq_len)
    # print('feats:', feats.shape)  # feats: torch.Size([5972, 1024])
    # print('boxes:', boxes.shape)  # boxes: torch.Size([5972, 4])
    # print('raw_scales:', raw_scales.shape)  # raw_scales: torch.Size([3]) -> torch.Size([3 * num_local_features])
    # print('ids:', ids.shape)  # ids: torch.Size([topk])

    top_feats = feats[ids]
    top_scales = raw_scales[ids]
    
    locations = calculate_keypoint_centers(boxes.cuda()[ids])  # CUDA exception triggered here
    
    if return_raw_scale:
        return top_feats, weights, top_scales, locations, seq_len
    else:
        scale_enc = torch.bucketize(ids, torch.asarray(scale_limits).cuda(), right=True)
        return top_feats, weights, scale_enc, locations, seq_len


@torch.no_grad()
def extract_local_feature(model, detector, test_loader, feature_storage, topk=700, chunk_size=5000):
    with torch.no_grad():
        img_feats = []

        for i, (im_list, scale_list) in enumerate(tqdm(test_loader, mininterval=10)):
            local_features, local_weights = [], []
            for idx in range(len(im_list)):
                im_list[idx] = im_list[idx].cuda()
                feats = extract_feat_res_pycls(im_list[idx], model)[0]
                if detector:
                    feats, weights = detector(feats)
                else:
                    weights = torch.linalg.norm(feats, dim=1).unsqueeze(1)
                local_features.append(feats)
                local_weights.append(weights)

            top_feats, weights, scale_enc, locations, seq_len = get_local(local_features, local_weights, scale_list, topk=topk)

            local_info = torch.zeros((topk, 1029))
            local_info[:seq_len] = torch.cat(
                (locations, scale_enc[:, None], torch.zeros_like(weights)[:, None], weights[:, None], top_feats),
                dim=1).cpu()
            local_info[seq_len:, 3] = 1
            img_feats.append(local_info)

            if (i + 1) % chunk_size == 0:
                if len(img_feats) > 0:
                    feature_storage.save(torch.stack(img_feats, dim=0), 'local')
                    feature_storage.update_pointer(len(img_feats))
                    img_feats = []
        if len(img_feats) > 0:
            feature_storage.save(torch.stack(img_feats, dim=0), 'local')

@torch.no_grad()    
def extract_local_feature_single(model, detector, im_list, scale_list, topk=700):
    local_features, local_weights = [], []
    
    for idx in range(len(im_list)):
        im_list[idx] = im_list[idx].cuda() # Move image to GPU; no need to unsqueeze again
        feats = extract_feat_res_pycls(im_list[idx], model)[0]  # Extract features
        
        if detector:
            feats, weights = detector(feats)  # Use detector for feature selection
        else:
            weights = torch.linalg.norm(feats, dim=1).unsqueeze(1)  # Calculate weights if no detector
        
        local_features.append(feats)
        local_weights.append(weights)
        
    top_feats, weights, top_scales, locations, seq_len = get_local(local_features, 
                                                                  local_weights, 
                                                                  scale_list, 
                                                                  topk=topk, 
                                                                  return_raw_scale=True)
    # Use dictionary to store local information: 
    # local_feats: [topk, 512]
    # local_scores, local_scales, local_mask: [topk]
    # local_locations: [topk, 2]
    local_feats = top_feats.cpu()
    local_scores = weights.cpu().squeeze()
    local_scales = top_scales.cpu().squeeze()
    local_mask = torch.ones_like(local_scores).cpu()
    local_mask[:seq_len] = 0
    local_locations = locations.cpu()
    
    return {
        'local_feats': local_feats,  # used in training
        'local_scores': local_scores,
        'local_scales': local_scales,  # used in training
        'local_mask': local_mask,  # used in training
        'local_locations': local_locations
    }
    

def load_cvnet(weight_path, model_type='cvnet_rn101'):
    if model_type == 'cvnet_rn101':
        model = ResNet(101, 2048)
    elif model_type == 'cvnet_rn50':
        model = ResNet(50, 2048)

    weight = torch.load(weight_path)
    weight_new = {}
    for i, j in zip(weight['model_state'].keys(), weight['model_state'].values()):
        weight_new[i.replace('encoder_q.', '')] = j

    mis_key, unexpect_keys = model.load_state_dict(weight_new, strict=False)
    print(f"Missing keys: {mis_key} for {model_type}")
    return model


# added by Zilin: replace extract function in orginal extractor setting
class CVNetExtractor:
    """
    Extractor w/ extractive saving strategy
    """
    def __init__(self, model, detector, top_k=100, device=None):
        self.model = model
        self.model.eval()
        
        self.detector = detector
        self.detector.eval()
        
        self.top_k = top_k
        self.device = device
        if not self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    @torch.no_grad()
    def extract(self, dataset, name="", show_progress=False):
        dataloader = dataset
        if show_progress:
            progress_bar = tqdm(dataloader, ncols=120, desc=f"Extract {name}")
        else:
            progress_bar = dataloader
            
        for im_list, scale_list, cache_path in progress_bar:   # why other process can not enter this dataloader?
            if isinstance(cache_path, list):
                assert len(cache_path) == 1, "Only support single cache path for now."
                cache_path = cache_path[0]
                
            if os.path.isfile(cache_path):
                print(f"Found cache for {cache_path}, skipping...")
                continue
            # print(im_list)
            # print(scale_list)
            # print(cache_path)
            # exit(0)
            # unlike DELG, extraction for CVNet takes two forward step to get global and local features.
            # 1. get global_feats with extract_feat_single
            global_feats = extract_feature_single(self.model, im_list)
            # 2. get local_feats with extract_local_feature_single
            local_info_dict = extract_local_feature_single(self.model, 
                                                           self.detector, 
                                                           im_list, 
                                                           scale_list, 
                                                           topk=self.top_k)
            data = {
                'global_feats': global_feats,
                **local_info_dict
            }
            parent_path = os.path.dirname(cache_path)
            os.makedirs(parent_path, exist_ok=True)
            torch.save(data, cache_path)
            
        if show_progress:
            progress_bar.close()
            


def extract(model, detector, feature_storage, dataloader, topk):
    if 'global' in feature_storage.save_type:
        features = extract_feature(model, dataloader)
        feature_storage.save(torch.from_numpy(features), 'global')
    if 'local' in feature_storage.save_type:
        extract_local_feature(model, detector, dataloader, feature_storage, topk=topk)
