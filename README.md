# FrameGraph

## Introduction
This repository contains the training code for the door frame example in the paper "FrameGraph: A Scalable Performance Evaluation Method for Frame Structure Designs Using Graph Neural Network." The required environment and dataset can be downloaded from Google Drive.
<p align="center">
  <img src="https://github.com/Li-Yongcheng/FrameGraph/blob/main/FrameGraph.png" />
</p>

## Have a try!
* Download the environment [py39.7z](https://drive.google.com/file/d/1-vCPz8M1Si4HfFvVaL2u3eT23cEVEQfP/view?usp=sharing) from Google Drive, Copy the extracted **py39** folder into the directory of your conda environment.
* Download the code repository and extract it to your local disk.
* Download the dataset [TU_DOOR_DATA.7z](https://drive.google.com/file/d/1-0_lkwfZGaPwFUQ-P5nTvGLtd7eLajIi/view?usp=sharing) from Google Drive, Copy the extracted **TU_DOOR_DATA** folder into the **FrameGraph/datasets** folder of the code.
* To run the training code, open a **CMD** window in the **FrameGraph** folder and execute:
```
conda activate py39
bash run_single.sh
```
If everything goes well, the training code will start running and display the prediction precision at the end of each epoch!

## Acknowledgment
We would like to express sincere thanks to the authors of the following tools and packages:
* [PyG](https://github.com/pyg-team/pytorch_geometric)
* [GraphGym](https://github.com/snap-stanford/GraphGym)

