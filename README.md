# Vox-Fusion

### [Project Page](https://xingruiy.github.io/vox-fusion/) | [Video](https://youtu.be/Prp28y1b2Qs) | [Paper](https://arxiv.org/abs/2210.15858)


> Vox-Fusion: Dense Tracking and Mapping with Voxel-based Neural Implicit Representation 
> 
> Xingrui Yang, Hai Li, Hongjia Zhai, Yuhang Ming, Yuqian Liu, Guofeng Zhang. 
> 
> ISMAR 2022
> 


## Installation

It is recommended to install [Pytorch](https://pytorch.org/get-started/locally/) manually for your hardware platform first. You can then install all dependancies using `pip` or `conda`:

```
pip install -r requirements.txt
```

After you have installed all third party libraries, run the following script to build extra Pytorch modules used in this project.

```bash
sh install.sh
```

## Demo

It is simple to run Vox-Fusion on datasets that already have dataloaders. `src/datasets` list all existing dataloaders. You can of course build your own, we will come back to it later. For now, we use the replica dataset as an example. 

First you have to modify `configs/replica/room_0.yaml` so the `data_path` section points to the real dataset path. Now you are all set to run the code:

```
python demo/run.py configs/replica/room_0.yaml
```

You should now see a progress bar and some output indicating the system is now running. For now you have to rely on the progress bar to estimate the running time as we are still working on a working GUI.

## Custom Datasets

You can use virtually any RGB-D dataset with Vox-Fusion including self-captured ones. Make sure to adapt the config files and dataloaders and put them in the correct folder. Make sure to implement a `get_init_pose` function for your dataloader, please refer to `src/datasets/tum.py` for an example.

## Acknowledgement

Some of our codes are adapted from [Nerual RGB-D Surface Reconstruction](https://dazinovic.github.io/neural-rgbd-surface-reconstruction/) and [BARF: Bundle Adjusted NeRF](https://github.com/chenhsuanlin/bundle-adjusting-NeRF/blob/main/camera.py).

## Citation

If you find our code or paper useful, please cite

```bibtex
@inproceedings{yang2022voxfusion,
  author    = {Xingrui Yang and Hai Li and Hongjia Zhai and Yuhang Ming and Yuqian Liu and Guofeng Zhang},
  title     = {{Vox-Fusion}: Dense Tracking and Mapping with Voxel-based Neural Implicit Representation},
  booktitle = {{IEEE} International Symposium on Mixed and Augmented Reality, {ISMAR}},
  pages     = {80--89},
  year      = {2021},
}
```

## Contact
Contact [Xingrui Yang](mailto:xingruiy@gmail.com) and [Hai Li](mailto:gary_li@zju.edu.cn) for questions, comments and reporting bugs.
