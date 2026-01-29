# Coral Segmentation Project

Jack Howell and Linnea Wolniewicz

data/ has 3 examples. image.png and seg_r10.png are the reference image and dilated radius 10 segmentation mask.

data_load.ipynb pulled .svg files from KOA and loaded them, and combines layers so that we have only five layers per svg file:
- nv # healthy
- nh # unhealthy
- hy # parasite1
- st/zo # parasite2
- image # background image

data_process.ipynb rasterized .svg files to .png files and dilated masks to have radius 10, to prepare for the segmentation model

coral_seg.ipynb was used to train the previous model, SegFormer, on hand-labelled png files

