# PlanScope

This is the official repository of

**PlanScope: Learning to Plan Within Decision Scope Does Matter**

[Ren Xin](https://rex-sys-hk.github.io), [Jie Cheng](https://jchengai.github.io/), [Hongji Liu](http://liuhongji.site) and [Jun Ma](https://personal.hkust-gz.edu.cn/junma/index.html)


<p align="left">
<a href="https://rex-sys-hk.github.io/pub_webs/PlanScope/">
<img src="https://img.shields.io/badge/Project-Page-blue?style=flat">
</a>
<a href='https://arxiv.org/abs/2411.00476' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/arXiv-PDF-red?style=flat&logo=arXiv&logoColor=wihte' alt='arXiv PDF'>
</a>
</p>

![PlanScopeConcept](https://github.com/user-attachments/assets/c622cb18-8ebe-4b70-94c7-6d7a4c443260)

## TL;NR
Based on PLUTO, we study the integrating method of long and short-term decision making, and the Time Dependent Normalization achieves the most significant improvement to 91.32% in the nuPlan Val4 CLS-NR score.

## Performance Comparison with SOTA methods
w/o post-processing
|  Model Name  | Val14 CLS-NR Score  | Val14 CLS-R Score  |
|  ----  | ----  | ----  |
| [PLUTO](https://github.com/jchengai/pluto)  | 89.04 | 80.01 |
| [STR2-CPKS-800M](https://github.com/Tsinghua-MARS-Lab/StateTransformer?tab=readme-ov-file)| 65.16 | - |
| [Diffusion Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner)  | 89.87 | **82.80** |
| PlanScope (ours)  | **91.32** | 80.96 |

Hybrid Mode
|  Model Name  | Val14 CLS-NR Score  | Val14 CLS-R Score  |
|  ----  | ----  | ----  |
| [PLUTO](https://github.com/jchengai/pluto)  | 93.21 | 92.06 |
| [STR2-CPKS-800M](https://github.com/Tsinghua-MARS-Lab/StateTransformer?tab=readme-ov-file)| 93.91 | 92.51 |
| [Diffusion Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner)  | **94.26** | **92.90** |
| PlanScope (ours)  | 93.59 | 91.07 |

## Setup Environment

### Setup dataset

Setup the nuPlan dataset following the [official-doc](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html)

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

Copy your chekpoint path to ```sim_scope.sh``` or ```sim_pluto.sh``` and replace the value of ```CKPT_N``` to run the evaluation. 

<!-- | PlanScope-h10-m6    | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/EcxJsqO4QgxJt2HeyfmDEssBelkGmMqzq3pFkk2w5OgQDQ?e=bUem3P)|
| PlanScope-h20-m6    | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/EbdjCkpdTEBKhwnz4VFv0R8BDD0C76zHsV7BedgYlytV5g?e=9BA7ft)| -->

| Model            | Download |
| ---------------- | -------- |
| Pluto-aux-nocil-m6-baseline  | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/EYkVd-OcOTFLlP5KE7ZnG-0BrluObe4vd7jNAhHeKtmcjw?e=UBmqf1)|
| PlanScope-Ih10-DWT | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/EXjVIgwKh3hCmMfJ-rQArcABRn3tH1RZhptPOLYRJjkS2A?e=scYt4e)    |
| PlanScope-Mh10-DWH | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/EXVaD_lc3kJBtUxGSQBBgPwBl8isEQzRaDtfrJ-geDB-XQ?e=pnbSPy)    |
| PlanScope-Mh20-DWT | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/EajN1DzBjKhMg4GiqkuuHuoBGilZzJbkK5QiPD9_GuoDLQ?e=BgidZM)    |
| --- |
| PlanScope-Th20 | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/EcHd8CFgBH1JqKT9yMyPsr0BukUsXTjfJpNSik_vQQrsLw?e=48VbzA)    |
| PlanScope-timedecay | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/EdMfIvFKuFlLh-SyHVvMB74Bs3TxH5hEp3HCSU34b6yAjg?e=KmVDGh)    |
| PlanScope-timenorm | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/EUMawRA-i-NIimhVp_I_Ft8BeuHWrCJzsVXb-E4BEMMQuA?e=0uRrDN)    |
| --- |
| Pluto-1M-aux-cil-m12-original | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jchengai_connect_ust_hk/EaFpLwwHFYVKsPVLH2nW5nEBNbPS7gqqu_Rv2V1dzODO-Q?e=LAZQcI)    |
| PlanScope-timenorm-cil-m12 | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/Ed863-9h9ZtFm145JyWGjCIBbF-rInj8P2smuXeG0SAPsg?e=g860Ho)    |

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


<!-- ## Special Announcement (Updated on 4 March 2025)

Our approach has achieved a CLS-NR score of 91.32% without rule-based post-processing, which currently is the highest score in pure-model-mode. 
However, the main objective is to find a general method for addressing horizon fusing problem, thus enhance the performance of planning models during execution. -->

<!-- This work investigates a technique to enhance the performance of planning models in a pure learning framework. We have deliberately omitted the rule-based pre- and post-processing modules from the baseline approach to mitigate the impact of artificially crafted rules, as claimed in our paper. A certain unauthorized publication led to **inaccuracies in the depiction of its state-of-the-art (SOTA) capabilities**. We hereby clarify this to prevent misunderstanding.

Nevertheless, the method introduced in our article is worth trying and could potentially serve as an add-on to augment the performance of the models you are developing, especially when the dataset is small. We are open to sharing and discussing evaluation results to foster a collaborative exchange. -->

## Others
- Please mind the common problem of nuPlan Dataset setup: https://github.com/motional/nuplan-devkit/issues/379 
- Advised NATTEN Version: 0.14.6+torch1121cu116
- Please mind your Linux system version, Ubuntu 18.04.6 LTS is prefered. Debian may lead to some unexpected error in closed-loop simulation.
- When training on the 20% dataset, the random selection of data splits during training possibly cause fluctuations of about 2% CLS-NR score on Random14, the training on partial dataset should only be used as reference during development.
- This repo is updated on 5 March 2025, the previous version can be found by checkout branch archived_1.
