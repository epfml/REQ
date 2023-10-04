# Rotational Optimizers
This repository provides the implementation and experiment scripts for the paper [Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks](https://arxiv.org/abs/2305.17212)

## Repository Structure
* **[experiments](https://github.com/epfml/rotational-optimizers/tree/main/experiments)**: Scripts to run the experiments, as reported in the paper. The experiments are seperated into folder analogously to the experiment sections and paragraphs in the paper. We order them by (dataset, architecture, optimizer, special hyper-parameters).
* **[shared/optimizers](https://github.com/epfml/rotational-optimizers/tree/main/shared/optimizers)**: Contains implementation of the rotational variant of the baseline optimizers (AdamW, SGD, Lion)
* **[submodules](https://github.com/epfml/rotational-optimizers/tree/main/submodules)**: Contains the **FairSeq**, **LLM-Baselines** and **TIMM** libraries, that are used to run experiments with the baseline and rotationl variants of the optimizers.

## Experiments
The scripts to run the experiments as reported in the paper are provided in experiments folder.
The bash scripts require `DATA_SET_DIR` to be set.
Note that the scripts are based on existing conda environments. How to set up the individual libraries, can be found in the respective libraries.

## Citation
```
@misc{kosson2023rotational,
    title={Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks},
    author={Kosson, Atli and Messmer, Bettina and Jaggi, Martin},
    journal={arXiv preprint arXiv:2305.17212},
    year={2023}
}
```
