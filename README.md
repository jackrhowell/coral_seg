# Coral Segmentation Project

Jack Howell and Linnea Wolniewicz

example_data/ has 3 examples. image.png and seg_r10.png are the reference image and dilated radius 10 segmentation mask.

All other files are in the scripts/ folder

scripts/dataset.py contains the PyTorch Dataset and DataModule classes for loading the data. It splits data by image to avoid data leakage.

scripts/model.py contains the segmentation model architecture (SegFormer with MiT-B0 backbone) and training code.

scripts/train.ipynb is a Jupyter notebook to train the segmentation model using the Dataset and DataModule from dataset.py and the model from model.py

scripts/data_load.ipynb pulled .svg files from KOA and loaded them, and combines layers so that we have only five layers per svg file:
- nv # healthy
- nh # unhealthy
- hy # parasite1
- st/zo # parasite2
- image # background image

scripts/data_process.ipynb rasterized .svg files to .png files and dilated masks to have radius 10, to prepare for the segmentation model

scripts/deprecated_train.ipynb was used to train the previous model, SegFormer, on hand-labelled png files

