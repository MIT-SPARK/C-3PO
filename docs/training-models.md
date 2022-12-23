# Training Models

### The Shapenet Experiment

**Self-Supervised Training on Real Data.** To train the models run:
```bash
cd scripts/expt_shapenet
bash self_supervised_train.sh
```
This trains models for all objects in the ShapeNet dataset.
If you want to train a model for a specific object, say chair, then run:
```bash
cd c3po/expt_shapenet
python training.py "point_transformer" "chair" "self_supervised"
```
Note that running self-supervised training will overwrite the trained and saved models.

**Pre-training on Simulated Data** To run supervised training on simulated data: 
```bash
cd scripts/expt_shapenet
bash supervised_train.sh
```
This trains models for all objects in the ShapeNet dataset.
If you want to train a model for a specific object, say bottle, then run:
```bash
cd c3po/expt_shapenet
python training.py "point_transformer" "bottle" "supervised"
```


### The YCB Experiment

**Self-Supervised Training on Real Data.** To train the models run:
```bash
cd scripts/expt_ycb
bash self_supervised_train.sh
```
This trains models for all objects in the YCB dataset.
If you want to train a model for a specific object, say 004_sugar_box, then run:
```bash
cd c3po/expt_ycb
python training.py "point_transformer" "004_sugar_box" "self_supervised"
```
Note that running self-supervised training will overwrite the trained and saved models.

**Pre-training on Simulated Data** To run supervised training on simulated data: 
```bash
cd scripts/expt_ycb
bash supervised_train.sh
```
This trains models for all objects in the ShapeNet dataset.
If you want to train a model for a specific object, say 021_bleach_cleanser, then run:
```bash
cd c3po/expt_ycb
python training.py "point_transformer" "021_bleach_cleanser" "supervised"
```

### Categoryless Self-Supervised Training Experiment

To train the models, run:
```bash
cd scripts/expt_categoryless
bash train_shapenet.sh
bash train_ycb.sh
```