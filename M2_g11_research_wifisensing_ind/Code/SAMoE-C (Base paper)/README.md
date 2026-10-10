# SAMoE-C Paper Reproduction

This folder contains my reproduced code for the SAMoE-C baseline model. I implemented this based on the original research paper to understand and test how continual learning works in dynamic environments. 

## Requirements
You just need a standard PyTorch setup to run the code. Here are the main libraries I used:
- `torch`
- `numpy`
- `argparse`

## Files Overview
Here is a quick summary of what each file does:
- `main_samoec.py`: The main script you need to run to train and evaluate the baseline.
- `config_samoec.py`: I put all the hyperparameters and domain settings in here so they're easy to change.
- `dataset_samoec.py`: Handles loading the data. It can generate synthetic data or load the real MM-Fi/CSI-Bench dataset.
- `models_samoec.py`: Contains the actual model architectures (the SAMoEC framework with the router and specialists).
- `buffer_samoec.py`: My implementation of the replay buffer.
- `train_samoec.py`: The training loops for the specialists and the router.
- `ssl_pretrain_samoec.py` & `continual_ssl_samoec.py`: The self-supervised learning (SSL) parts and adapters.
- `utils_samoec.py`: Some helper functions for things like setting random seeds.

## How to Run the Code

You can run everything through `main_samoec.py`. I've set up a few different ways to run it depending on what you want to test.

### 1. Quick Test
If you just want to check if the code runs without errors, you can do a quick smoke test. It uses synthetic data and only trains for 1 epoch.
```bash
python main_samoec.py --smoke-test
```

### 2. Full Synthetic Data Run
If you run the script normally, it will train the model sequentially across synthetic domains.
```bash
python main_samoec.py
```

### 3. Running with Real Data (MM-Fi)
If you have the actual MM-Fi dataset downloaded, you can pass the path to it using `--mmfi-root`. You also need to add the `--real-data` flag.
```bash
python main_samoec.py --real-data --mmfi-root /path/to/mmfi/dataset
```

### 4. Changing Domain Order
I added a flag to test how the model performs if we change the curriculum (the order of domains). Just add `--alt-order`:
```bash
python main_samoec.py --alt-order
```
You can also combine this with the real data flag.

## What to Expect
When you run the code, it will print out the progress in the console. First, it trains on the initial domain, and then it sequentially adds new specialists for the following domains.
After every domain, it prints out the validation accuracy for that specific domain and the router's self-discrimination accuracy. At the very end, it runs a final test on a combined, balanced dataset of all domains and gives you the final overall accuracy.
