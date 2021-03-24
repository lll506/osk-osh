export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch  --master_port 8088  --nproc_per_node=2  tools/inference.py --config-file configs/glide/dota_keypoint.yaml
