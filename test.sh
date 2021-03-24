export CUDA_VISIBLE_DEVICES=0,1,2
rm -rf test_result
mkdir test_result
python -m torch.distributed.launch --master_port 8088  --nproc_per_node=3  tools/test_net.py --config-file configs/glide/dota_keypoint.yaml
python maskrcnn_benchmark/DOTA_devkit/ResultMerge.py
