# ProteinSuperfamPredictor

## Dataset Generation
Files related to generation of the dataset, including data download and pre-processing, are located in the `psfpred/dataset` directory.

The dataset class is defined in `psfpred/dataset/ProteinDataset.py`.  The following commands can be used to download, pre-process, and create an H5 data file.

To download the PDB files corresponding to the entities in the SCOP2 dataset file located in `data/scop-cla-latest.txt`), run:
```
python psfpred/dataset/download_dataset.py
```
Next, to pre-process and create the H5 file, run:
```
python psfpred/dataset/generate_h5_file.py
```

## Model Training
Model training requires an existing `ProteinDataset` H5 file located at `data/dataset.h5`.  Training is configured to utilize a CUDA device if present or CPU as a backup.  Before training, paths to store the model and loss plots should be configured within `psfpred/train.py`  To begin training, run:
```
python psfpred/train.py
```

## Model Evaluation
Model evaluation requries a trained model file as well as an existing `ProteinDataset` H5 file located at `data/dataset.h5`.  Evaluation will run on the held-out test dataset consisting on 20% of the dataset and will produce a `figures/results.csv` and confusion matrices for each model configured in `psfpred/evaluate_models.py`.  To evaluate models, run:
```
python psfpred/evaluate_models.py
```