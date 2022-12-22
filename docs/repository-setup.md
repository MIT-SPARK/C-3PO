# Setup Repository

Download and setup the dataset and the pre-trained models.

## Datasets

Our experiments rely on  the [ShapeNet](https://shapenet.org/), [KeypointNet](https://github.com/qq456cvb/KeypointNet), 
and the [YCB](https://www.ycbbenchmarks.com/object-models/) datasets. Please view ShapeNet's terms of use [here](https://shapenet.org/terms). 

There's no need to download the datasets seperately. Our experiments use a processed version of these datasets. 
Follow the steps below to download and save the relevant dataset. 

1. Download our processed dataset files on Google Drive [here](https://drive.google.com/drive/folders/1EYa8B0dID1vk9bze93pzil8rVj2-fYb5?usp=sharing) and move all the files to the same directory this README is in (the C3PO repo). We've provided the dataset as a zip archive split into 1GB chunks of the format ```c3po-data.z**```.

2. Combine the archives into one zip file: 
	```bash 
	zip -F c3po-data.zip --out data.zip
	```

3. Unzip the file:
	```bash
	unzip data.zip
	```

4. Verify your directory structure looks as follows:

	```
	C3PO
	│   README.md
	│   c3po   
	│   setup.py
	└───data
	│   │   learning-objects
	│   |   └───...
	│   │
	│   │   KeypointNet
	│   |   └───...
	│   │
	│   └───ycb
	│       │   models
	│       └───...
	│   
	└───...
	``` 
 

## Pre-trained Models

We provide trained models for you to either use or reproduce our results. 
Download these from Google Drive 
[here](https://drive.google.com/file/d/1bOyMsT0xTZDhX8L-PjA41KdSz5ZLlgXj/view?usp=sharing) 
and unzip the file. The unziped folder 
has the structure:

```
c3po
│   
└───expt_corrector
│   │    
│   └───runs
│       └───...
│      
└───expt_shapenet
│   │  
│   └───...
│   
└───expt_shapenet
    │  
    └───...
``` 

Place contents of folder c3po/expt_corrector in 
C-3PO/c3po/expt_corrector, in the repo; and similarly for
c3po/expt_shapenet, c3po/expt_ycb, c3po/expt_categoryless, and c3po/expt_time.

