export CUDA_VISIBLE_DEVICES=0,1,2
python -m torch.distributed.launch --master_port 8088  --nproc_per_node=3  tools/train_net.py  --config-file configs/glide/dota_keypoint.yaml

