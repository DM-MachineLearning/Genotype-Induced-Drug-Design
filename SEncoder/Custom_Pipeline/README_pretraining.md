Command to pre-train on backround
```
nohup python pretrain_encoder.py --data_dir data/chembl/tokenized --checkpoint_dir checkpoints/encoder_pretrain --epochs 10 2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log
```