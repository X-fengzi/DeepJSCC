import os
import yaml
import subprocess
from itertools import product

original_yaml = "./train_yaml/MNIST_IBSC.yaml"

param_grid = {
    'beta': [0.000001,0.00001,0.0001,0.0002,0.0004,0.0006,0.0008,0.001,0.002,0.004,0.006,0.008,0.01,0.1],
}

output_dir = "./train_yaml/"+"MNIST_IBSC"
os.makedirs(output_dir, exist_ok=True)

# original_yaml = "./train_yaml/MNIST_DT_Gumbel.yaml"
# param_grid = {
#     'embd_dim': [48]
# }
# output_dir = "./train_yaml/"+"MNIST_DT_Gumbel"
# os.makedirs(output_dir, exist_ok=True)

with open(original_yaml, 'r') as f:
    base_config = yaml.safe_load(f)

keys = param_grid.keys()
values = param_grid.values()

for idx, combo in enumerate(product(*values)):
    config = {k: v for k, v in base_config.items()}  
    for key, val in zip(keys, combo):
        config[key] = val
    name_parts = [f"{k}{v}".replace('.', '_') for k, v in zip(keys, combo)]
    filename = f"run_{'_'.join(name_parts)}.yaml"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    cmd = ["python", "main.py", "--config_file_path", filepath]
    print(f"[{idx+1}] Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"⚠️  Job failed: {filepath}")
