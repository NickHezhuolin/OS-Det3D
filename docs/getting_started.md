# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test (example: split2/5cls)

**Stage 1**

Train bevformer with 8 GPUs in closed-set setting (18 epoch)
```
bash tools/dist_train_bev.sh projects/configs/bevformer/bevformer_base_5cls.py 8
```

Train LG-RPN with 8 GPUs in closed-set setting (20 epoch)
```
./tools/dist_train.sh projects/configs/lgrpn/lgrpn_5cls.py 8
```

Eval LG-RPN with 8 GPUs for 3D Region Proposals
```
./tools/dist_test.sh projects/configs/lgrpn/lgrpn_5cls.py ./path/to/ckpts.pth 8 
```

**Stage 2**

Train LOUD with 8 GPUs (6 epoch)
```
bash tools/dist_train.sh projects/configs/loud/bevformer_base_loud_5cls.py 8
```

Eval LOUD with 8 GPUs
```
bash tools/dist_test.sh projects/configs/loud/bevformer_base_loud_5cls.py ./path/to/ckpts.pth 8
```
Note: the result_nusc.json file will be output in test/. ( code detail : see [test.py](../tools/test.py) line 246-248 )


# Visualization 

See [visual.py](../tools/analysis_tools/visual.py)

# Open-set eval

See [eval_nusc_ar.py](../tools/eval_nusc_ar.py). Needs result_nusc.json file.