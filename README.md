# DeepJSCC
The baseline implementation of the paper Variable-Length End-to-End Joint Source-Channel Coding for Semantic Communication submitted to IEEE ICC 2026.


## Usage

### Quick Start

Run training and evaluation with default settings:

1. optimal rate-distortion tradeoff
```bash
python main.py --config_file_path "./train_yaml/MNIST_IBSC.yaml"
```

2. Discrete DeepJSCC with gumbel softmax trick

```bash
python main.py --config_file_path "./train_yaml/MNIST_DT_Gumbel.yaml"
```

### Custom Training

You can customize hyperparameters via  changing the config.py file or yaml file