<div align="center">

## Joint Training of Deep Ensembles Fails Due to Learner Collusion
</div>

<p align="center">
<img src="figure.jpg" width="500">
</p>

This repository contains the code associated with [our NeurIPS 2023 paper](https://arxiv.org/abs/2301.11323) titled "Joint Training of Deep Ensembles Fails Due to Learner Collusion". This work investigates why training deep ensembles *jointly* results in degenerate behavior. For further details, please see our paper. In this repository, we include the code to reproduce the image-based experiments from the paper. 


### Experiments
**Setup**

Clone this repository and navigate to the root folder.
```
git clone https://github.com/alanjeffares/joint-ensembles.git
cd joint-ensembles
```
Ensure PYTHONPATH is also set to the root folder.
```
export PYTHONPATH="/your/path/to/joint-ensembles"
```
Using conda, create and activate a new environment. 
```
conda create -n <environment name> pip python
conda activate <environment name>
```
Then install the repository requirements.
```
pip install -r requirements.txt
```

Then the experiments are split across three files. `src/sweep.py` runs the experiments that sweep across values of $\beta$ from the augmented objective in Sec. 5. `src/imagenet.py` runs the ImageNet experiments reported in Table 1. `src/diagnostics.ipynb` is a notebook containing the three diagnostic experiments (1) Diversity explosion; (2) Debiased diversity; (3) Learner codependence. 

**Sweep**

Set the path to the folder in which the data is stored (or should be downloaded to) in `src/configs/data.json`.

Next, select the configuration of the experiment. The high-level experimental parameters are set in `src/configs/sweep/experiment.json` and the more fine-grained optimization hyperparameters are set in `src/configs/sweep/optim.json`. The values implemented for the arguments that the user might wish to change are listed in `src/tests/implemented.json`.

Once the configs are set, the experiment can be run with the following command (where `<tag>` is an optional keyword to save the experiment under).
```
python src/sweep.py <tag>
```
After the experiment is complete the results can be found in the `results/` folder.

**ImageNet**

Download and process the ImageNet data following the instructions from [this blog post](https://towardsdatascience.com/downloading-and-using-the-imagenet-dataset-with-pytorch-f0908437c4be). Then set the path to the folder in which the data is stored in `src/configs/data.json`.

Next, select the configuration of the experiment. The high-level experimental parameters are set in `src/configs/imagenet/experiment.json` and the more fine-grained optimization hyperparameters are set in `src/configs/imagenet/optim.json`. The values implemented for the arguments that the user might wish to change are listed in [`src/tests/implemented.json`](https://github.com/alanjeffares/joint-ensembles/blob/master/src/tests/implemented.json).

Once the configs are set, the experiment can be run with the following command (where `<tag>` is an optional keyword to save the experiment under).
```
python src/imagenet.py <tag>
```
After the experiment is complete the results can be found in the `results/` folder.

**Diagnostics**

Download the results of the Bayesian optimization sweep from [the following link](https://drive.google.com/drive/folders/1WN8uEkxRbyV5DAnJuZWfvAN0BRyJkbLl?usp=sharing).

Then simply run the cells in `src/diagnostics.ipynb` corresponding to each of the three post-hoc experiments.

**Plotting**

A plotting notebook is provided in `src/plotting/generate_figure.ipynb` illustrating how the outputs of *sweep* and *ImageNet* experiments can be plotted. 

### Citation

If you use this code, please cite the associated paper.

```
@inproceedings{
  jeffares2023joint,
  title={Joint Training of Deep Ensembles Fails Due to Learner Collusion},
  author={Alan Jeffares and Tennison Liu and Jonathan Crabb{\'e} and Mihaela van der Schaar},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
  url={https://openreview.net/forum?id=WpGLxnOWhn}
  }
```
