# grid search on gldv2-val for best alpha and temp
python eval_revisitop.py \
    --checkpoint_path ./checkpoints/dinov2_locore_base \
    --device cuda:0 --gldv2_val

# run sliding window reranking
python eval_revisitop.py \
    --checkpoint_path ./checkpoints/dinov2_locore_base \
    --device cuda:0 --alpha 0.5 --temp 0.1 --with_hard --with_1m --sliding_reranking_type end-to-start