
# Print current date
date

# Print debug info
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Number of available GPUs: $(nvidia-smi --list-gpus | wc -l)"

cd
source anaconda3/bin/activate
conda activate neurips
cd NeurIPS/2dcavity/

# Simple approach with DataParallel
export PYTHONUNBUFFERED=1  # This ensures print statements are shown immediately

#python3 pinn_cavity.py --config config_cavity.yaml
#python3 utils_cavity_data.py

nohup python3 pinn_cavity.py --config config_cavity.yaml > output.txt 2>&1 &

#nohup python3 pinn_cavity.py --config config_cavity2.yaml > output_og_May14_8.txt 2>&1 &

#nohup python3 pinn_cavity_new.py --config config_cavity_new.yaml > output_May1_test2.txt 2>&1 &
#nohup python3 utils_cavity_data.py > valid_data_output.txt 2>&1 &
date
