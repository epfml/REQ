# Rotational Optimizers
This repository provides the implementation and experiment scripts for the paper [Rotational Optimizers: Simple & Robust DNN Training](https://arxiv.org/abs/2305.17212)

## Repository Structure
* **[experiments](https://github.com/epfml/rotational-optimizers/tree/main/experiments)**: Scripts to run the experiments, as reported in the paper
* **[shared/optimizers](https://github.com/epfml/rotational-optimizers/tree/main/shared/optimizers)**: Contains implementation of the rotational variant of the baseline optimizers (AdamW, SGD, Lion)
* **[submodules](https://github.com/epfml/rotational-optimizers/tree/main/submodules)**: Contains the **FairSeq**, **LLM-Baselines** and **TIMM** libraries, that are used to run experiments with the baseline and rotationl variants of the optimizers.

## Experiments
The scripts to run the experiments as reported in the paper are provided in experiments folder.
In order to run the experiments the environment variables `TMUX_SESSION`, `EXPERIMENT`, `DATA_DIR`, `PPATH` and `CONDA_ENV` need to be set.
Note that the scripts are based on existing conda environments. How to set up the individual libraries, can be found in the respective libraries.

## Citation
```
@misc{kosson2023rotational,
      title={Rotational Optimizers: Simple & Robust DNN Training}, 
      author={Atli Kosson and Bettina Messmer and Martin Jaggi},
      year={2023},
      eprint={2305.17212},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
