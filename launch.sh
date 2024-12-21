CUDA_VISIBLE_DEVICES=0 nohup wandb agent rcjkhhgf >> output1.log &
CUDA_VISIBLE_DEVICES=1 nohup wandb agent rcjkhhgf >> output2.log &
CUDA_VISIBLE_DEVICES=2 nohup wandb agent rcjkhhgf >> output3.log &
CUDA_VISIBLE_DEVICES=3 nohup wandb agent rcjkhhgf >> output4.log &