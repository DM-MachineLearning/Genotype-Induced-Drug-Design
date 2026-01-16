nohup stdbuf -oL -eL python pretrain_encoder.py --data_dir data/chembl/tokenized \
    --lr 0.0009808126266739308 \
    --dropout 0.18184389629895298 \
    --weight_decay 0.0008589569128923666 \
    --n_layers 12 \
    --latent_size 512 \
    --embedding_dim 1024 \
    --batch_size 16 \
    --warmup_steps 1000 \
    --grad_clip 1.2812360187852758 \
    --epochs 5 \
    --max_steps 10000000 \
    > logs/training_$(date +%Y%m%d_%H%M%S).log