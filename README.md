
This project contains the code for the paper `Pose estimation for objects with rotational symmetry`. More information about the project in our [project page link](http://www.cs.utoronto.ca/~ecorona/symmetry_pose_estimation/) or the [paper link](http://www.cs.utoronto.ca/~ecorona/symmetry_pose_estimation/paper.pdf)

# REQUIREMENTS

Generating the dataset and running experiments requires PyTorch 0.4. The CAD models are available for download [here link](https://drive.google.com/file/d/1yNPIlFaR0YE-FTyjupJxYCNc0ideg_dq/view?usp=sharing). They are based by a realistic database of textures which can be download from [here link](https://drive.google.com/file/d/18PuNQgZ1PKmF2pp4y9RRIxWq6HPyKZaC/view?usp=sharing)

# RUN CODE

To generate a synthetic dataset from the object CAD models please see the [dataset generation folder link](https://github.com/enriccorona/PoseSym/tree/master/Synthetic_simulation)

The training code is based on PyTorch and more documented in the [training folder link](https://github.com/enriccorona/PoseSym/tree/master/Training)

# CITATION

If you use this code or ideas from the paper in your research, please cite our paper:

```
@inproceedings{corona2018pose,
  title={Pose Estimation for Objects with Rotational Symmetry},
  author={Corona, Enric and Kundu, Kaustav and Fidler, Sanja},
  booktitle={2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={7215--7222},
  year={2018},
  organization={IEEE}
}
```
