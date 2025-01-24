#!/bin/bash



#Source conda enviroment
source /home/desmondhe/anaconda3/bin/activate ad

#Check the CUDA version of the GPU
echo "Cuda version using nvidia-smi"
nvidia-smi

#Confirm the environment is active
echo "Virtual environment"
which python
python --version

CONFIG_FILE="config.toml"
#Run the training script
cd /home/desmondhe/ADwithNE
echo "Start the training"
# python -u train_embedder.py
sed -i '/^\[Transformer\]/,/^$/s/output_dim = 2/output_dim = 4/' "$CONFIG_FILE"
python -u embed.py

sed -i '/^\[Transformer\]/,/^$/s/output_dim = 4/output_dim = 8/' "$CONFIG_FILE"
python -u embed.py

sed -i '/^\[Transformer\]/,/^$/s/output_dim = 8/output_dim = 16/' "$CONFIG_FILE"
python -u embed.py

sed -i '/^\[Transformer\]/,/^$/s/output_dim = 16/output_dim = 32/' "$CONFIG_FILE"
python -u embed.py

#Deactivate the environment
conda deactivate