# conda env
source activate /home/bro/miniconda3/envs/wan22

# working directory
cd /home/bro/Wan2.2-motion

python train_motion.py \
  --ckpt_dir ../Wan2.2-TI2V-5B \
  --driver_video examples/kling_20260512.mp4 \
  --ref_image examples/cat.png \
  --prompt "the cute cat raises its right paw and licking the paw repeatedly, satisfied and gently" \
  --frame_num 81 \
  --steps 500 \
  --output checkpoints/motion_transfer.pt

python generate_motion.py \
  --ckpt_dir ../Wan2.2-TI2V-5B \
  --motion_ckpt checkpoints/motion_transfer.pt \
  --image examples/corgi.png \
  --motion_ref_image examples/cat.png \
  --motion_video examples/kling_20260512.mp4 \
  --prompt "the cute corgi raises its right paw and licking the paw repeatedly, satisfied and gently" \
  --size 1280*704 \
  --frame_num 81 \
  --offload_model True \
  --sample_steps 50 \
  --convert_model_dtype \
  --t5_cpu \
  --save_file output/corgi_motion.mp4