# Rotational Equilbrium
This repository provides the implementation and experiment scripts for our ICML 2024 paper [Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks](https://arxiv.org/abs/2305.17212)

## Repository Structure
* **[experiments](https://github.com/epfml/REQ/tree/main/experiments)**: Scripts to run the experiments, as reported in the paper. The experiments are seperated into folder analogously to the experiment sections and paragraphs in the paper. We order them by (dataset, architecture, optimizer, special hyper-parameters). The code has undergone additional changes since we ran some scripts so they may not repreduce our results exactly anymore, but should give a good starting point.
* **[shared/optimizers](https://github.com/epfml/REQ/tree/main/shared/optimizers)**: Contains implementation of the rotational variant of the baseline optimizers (AdamW, SGD, Lion). We use the implementation from rotational_wrapper.py for our experiments but also provide simpler implementations of individual algorithms in, rotational_sgd.py, rotational_adamw.py and rotational_lion.py.
* **[submodules](https://github.com/epfml/REQ/tree/main/submodules)**: Contains the **FairSeq**, **LLM-Baselines**, **TIMM** and **nanoGPT** libraries, that are used to run experiments with the baseline and rotationl variants of the optimizers.

## Experiments
The scripts to run the experiments as reported in the paper are provided in experiments folder.
The bash scripts require `DATA_SET_DIR` to be set.
Note that the scripts are based on existing conda environments. How to set up the individual libraries, can be found in the respective libraries.

## Citation
```
@inproceedings{
  kosson2024rotational,
  title={Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks},
  author={Atli Kosson and Bettina Messmer and Martin Jaggi},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024},
  url={https://openreview.net/forum?id=MQirNNU2pC},
  note={\href{https://arxiv.org/abs/2305.17212}{arXiv:2305.17212}}
}
```
