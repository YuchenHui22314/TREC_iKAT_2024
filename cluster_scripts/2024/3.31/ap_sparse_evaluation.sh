#!/bin/bash

#SBATCH --account=def-jynie
#SBATCH --job-name=ance_ap_dense_indexing           # will appear when sq
#SBATCH --nodes=1                  # min numbre of nodes used  1-5 -> min 1 max 5
#SBATCH --gpus-per-node=v100l:4     # or --gpus=p100:1 or --gres=gpu:1
#SBATCH --ntasks-per-node=32        # max thread number
#SBATCH --cpus-per-gpu=6           # 6 is safe for all servers.
#SBATCH --exclusive                # not share node with others
#SBATCH --mem=64G
#SBATCH --output /home/yuchenxi/projects/def-jynie/yuchenxi/IR_LLM_user_intent_alignment/cluster_scripts/outputs/%x_%j.out

#SBATCH --mail-user=yuchen.hui.udem+cc@gmail.com
#SBATCH --mail-type=ALL

  
module purge
module load python/3.10
module load StdEnv/2023
module load gcc/12.3
module load openmpi/4.1.5
module load cuda/12.2
module load faiss
module load arrow/15.0.1
module load java/17.0.6
module load rust/1.70.0

#module list

cd /home/yuchenxi/projects/def-jynie/yuchenxi/IR_LLM_user_intent_alignment
source ./alignir_env/bin/activate

cd /home/yuchenxi/projects/def-jynie/yuchenxi/IR_LLM_user_intent_alignment/src/evaluate
#pip list
chmod +x run_AP_sparse_evaluation.sh

echo "beginning evaluation"
./run_AP_sparse_evaluation.sh
