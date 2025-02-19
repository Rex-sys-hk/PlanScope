# PlanScope

This is the official repository of

**PlanScope: Learning to Plan Within Decision Scope Does Matter**

[Ren Xin](https://rex-sys-hk.github.io), [Jie Cheng](https://jchengai.github.io/), and [Jun Ma](https://personal.hkust-gz.edu.cn/junma/index.html)


<p align="left">
<a href="https://rex-sys-hk.github.io/pub_webs/PlanScope/">
<img src="https://img.shields.io/badge/Project-Page-blue?style=flat">
</a>
<a href='https://arxiv.org/abs/2411.00476' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/arXiv-PDF-red?style=flat&logo=arXiv&logoColor=wihte' alt='arXiv PDF'>
</a>
</p>

![PlanScopeConcept](https://github.com/user-attachments/assets/c622cb18-8ebe-4b70-94c7-6d7a4c443260)


## Setup Environment

### Setup dataset

Setup the nuPlan dataset following the [offiical-doc](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html)

### Setup conda environment

```
conda create -n planscope python=3.9
conda activate planscope

# install nuplan-devkit
git clone https://github.com/motional/nuplan-devkit.git && cd nuplan-devkit
pip install -e .
pip install -r ./requirements.txt

# setup planscope
cd ..
git clone https://github.com/Rex-sys-hk/PlanScope && cd planscope
sh ./script/setup_env.sh
```

## Feature Cache

Preprocess the dataset to accelerate training. It is recommended to run a small sanity check to make sure everything is correctly setup.

```
 python run_training.py \
    py_func=cache +training=train_pluto \
    scenario_builder=nuplan_mini \
    cache.cache_path=/nuplan/exp/sanity_check \
    cache.cleanup_cache=true \
    scenario_filter=training_scenarios_tiny \
    worker=sequential
```

Then preprocess the whole nuPlan training set (this will take some time). You may need to change `cache.cache_path` to suit your condition

```
 export PYTHONPATH=$PYTHONPATH:$(pwd)

 python run_training.py \
    py_func=cache +training=train_pluto \
    scenario_builder=nuplan \
    cache.cache_path=/nuplan/exp/cache_pluto_1M \
    cache.cleanup_cache=true \
    scenario_filter=training_scenarios_1M \
    worker.threads_per_node=40
```

## Training


```
sh train_scope.sh
```

- you can remove wandb related configurations if your prefer tensorboard.


## Checkpoint

Copy your the chekpoint path to ```sim_scope.sh``` or ```sim_pluto.sh``` and replace the value of ```CKPT_N``` to run the evaluation. 

| Model            | Download |
| ---------------- | -------- |
| Pluto-aux-nocil-m6-baseline  | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/EYkVd-OcOTFLlP5KE7ZnG-0BrluObe4vd7jNAhHeKtmcjw?e=UBmqf1)|
| PlanScope-h10-m6    | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/EcxJsqO4QgxJt2HeyfmDEssBelkGmMqzq3pFkk2w5OgQDQ?e=bUem3P)|
| PlanScope-h20-m6    | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/EbdjCkpdTEBKhwnz4VFv0R8BDD0C76zHsV7BedgYlytV5g?e=9BA7ft)|
<!---| Pluto-1M-aux-cil-m12-original | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jchengai_connect_ust_hk/EaFpLwwHFYVKsPVLH2nW5nEBNbPS7gqqu_Rv2V1dzODO-Q?e=LAZQcI)    | -->

## Run PlanScope-planner simulation

Run simulation for a random scenario in the nuPlan-mini split

```
sh ./sim_scope.sh
```


## To Do

The code is under cleaning and will be released gradually.

- [ ] improve docs
- [x] training code
- [x] visualization
- [x] Scope-planner & checkpoint
- [x] feature builder & model
- [x] initial repo & paper

## Citation

If you find this repo useful, please consider giving us a star 🌟 and citing our related paper.

```bibtex
@misc{planscope,
      title={{PlanScope:} Learning to Plan Within Decision Scope Does Matter}, 
      author={Ren Xin and Jie Cheng and Jun Ma},
      year={2024},
      eprint={2411.00476},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2411.00476}, 
}
```

## Thanks
- [tuplan](https://github.com/autonomousvision/tuplan_garage)
- [pluto](https://github.com/jchengai/pluto)


## Special Announcement

This work investigates a technique to enhance the performance of planning models in a pure learning framework. We have deliberately omitted the rule-based pre- and post-processing modules from the baseline approach to mitigate the impact of artificially crafted rules, as claimed in our paper. A certain unauthorized publication led to **inaccuracies in the depiction of its state-of-the-art (SOTA) capabilities**. We hereby clarify this to prevent misunderstanding.

Nevertheless, the method introduced in our article is worth trying and could potentially serve as an add-on to augment the performance of the models you are developing, especially when the dataset is small. We are open to sharing and discussing evaluation results to foster a collaborative exchange.

## A common problem of nuPlan Dataset
https://github.com/motional/nuplan-devkit/issues/379 