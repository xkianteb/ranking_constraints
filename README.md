# Ranking with Long-Term Constraints

## Enviroment Setup
Python 3.10
- Create virtual Enviroment with virtualenv, conda or ...

- Install required libraries

```shell
pip install -r requirements.txt
```

## Environment Configs

```shell
usage: simulate.py [-h] --conf CONF --dataset DATASET --ctrl {mc,sc,schinge,oracle,pc,pchinge,base} [--seed SEED]
                   [--lmbda LMBDA] [--gamma GAMMA] [--output_dir OUTPUT_DIR] [--c C [C ...]]
                   [--b B] [--bo BO] [--relevance_type {offline_relevance,online_relevance,sequence_relevance}]
                   [--lr LR] [--init {one,zero}] [--metrics_file_name METRICS_FILE_NAME]
                   [--targets TARGETS [TARGETS ...]] [--dev] [--hinge_min HINGE_MIN]
                   [--shuffle_bootstraps SHUFFLE_BOOTSTRAPS] [--eps EPS] [--beta BETA]

Simulate the online ranking systemunder differenct control policies

options:
  -h, --help            show this help message and exit
  --conf CONF           config file
  --dataset DATASET     dataset
  --ctrl {mc,sc,schinge,oracle,pc,pchinge,base}
                        controller
  --seed SEED           random seed
  --lmbda LMBDA         lambda used in P-Controller
  --gamma GAMMA         gamma used in BP-Controller
  --output_dir OUTPUT_DIR
                        Output of mterics
  --c C [C ...]         C in weighted objective
  --b B                 B offline in PC with bootstrap
  --bo BO               B online in PC with bootstrap
  --relevance_type {offline_relevance,online_relevance,sequence_relevance}
  --lr LR       learning rate
  --init {one,zero} init value
  --metrics_file_name METRICS_FILE_NAME
  --targets TARGETS [TARGETS ...]
  --dev
  --hinge_min HINGE_MIN
  --shuffle_bootstraps SHUFFLE_BOOTSTRAPS
  --eps EPS
  --beta BETA
```

## Datasets `<dataset>`:
KuaiRec: `kuai` <br />
Linear television dataset Tv Audience: `zf_tv` <br />
Last.fm: `lastfm_len_top_10` <br />
Fully-synthetic: `early_and_late` <br />

## Datasets-Configurations `<conf>`:
KuaiRec: `experiments/multi_group.yml` <br />
Linear television dataset Tv Audience: `experiments/zf_tv_single.yml` <br />
Last.fm: `experiments/multi_lastfm_len_top_10.yml.yml` <br />
Fully-synthetic: `experiments/early_and_late` <br />

## Controllers `<ctrl>`:
Myopic Controller: `mc` <br />
Stationary Controller: `sc` <br />
Oracle: `oracle` <br />
Predictive Controller: `pc` <br />
Myopic Controller with out constraints: `base` <br />

## Example:

The below command works for the `mc`, `sc`, `oracle`, and `base` controllers:
```shell
python -u simulate.py 
    --conf <conf>
    --dataset <dataset>  
    --ctrl <ctrl>
    --output_dir <output-directory>
    --c 10. 10. 
    --metrics_file_name <job-name>  
    --shuffle_bootstraps true
```


The below command works for `pc` controller:
```shell
python -u simulate.py 
    --conf  <conf>
    --dataset <dataset> 
    --ctrl <ctrl> 
    --output_dir  <output-directory> 
    --c 10. 10.  
    --metrics_file_name <job-name> 
    --shuffle_bootstraps false 
    --relevance_type offline_relevance 
    --b <B offline in PC with bootstrap>
    --bo <B online in PC with bootstrap>
```

## Citation

```
@article{Brantley2023RankingWL,
  title={Ranking with Long-Term Constraints},
  author={Kiant{\'e} Brantley and Zhichong Fang and Sarah Dean and Thorsten Joachims},
  journal={ArXiv},
  year={2023},
}
```


