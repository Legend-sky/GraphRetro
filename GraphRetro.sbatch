#!/bin/zsh                    
#提交单个作业
#SBATCH --job-name=GraphRetro      
#SBATCH --partition=mixed       
#SBATCH --nodes=2
#SBATCH --cpus-per-task=1       
#SBATCH -n 32       
#SBATCH --ntasks-per-node=16

#SBATCH --output=output.out      
#SBATCH --error=error.out       

### 作业命令 

## 2. 执行python脚本
python scripts/benchmarks/run_model.py --config_file configs/lg_ind/defaults.yaml