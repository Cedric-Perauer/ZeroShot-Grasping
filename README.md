# ZeroShot-Grasping
Repository for ZeroShot-Grasping project 

## Subproblems
    - 1. Feature Extraction of Dinov2 on textureless objects -> Baris
    - 2. Parameterization/features for transferring correspondence -> Cedric
        - How to deal with missing correspondence on grasping points ? 
        - Feature space, grasp points 
    - 3. Datasets -> Zhehuan
        - CO3D, LineMod, GraspNet 1 Billion, HouseCAD60  
        - Find optimal view
        
## Roadmap 

0. set up Github, share Code 
1. Research on Datasets, correspondence between features & grasp points 
2. Build on zero-shot category level object pose estimation & other research
    1. Use Dinov2
    2. Choose specific dataset
    3. Transfer of the Grasping representation


## Relevant Papers 

0. [Dinov2](https://arxiv.org/abs/2304.07193) | [Github](https://github.com/facebookresearch/dinov2)
1. [Zero-Shot Category Level Object Pose Estimation](https://arxiv.org/abs/2204.03635) | [Github](https://github.com/applied-ai-lab/zero-shot-pose)
2. [Lightweight Convolutional Neural Network with
Gaussian-based Grasping Representation](https://arxiv.org/pdf/2101.10226.pdf)
3. [SurfEmb](https://arxiv.org/pdf/2111.13489.pdf) | [Github](https://github.com/rasmushaugaard/surfemb)
4. [TransGrasp](https://arxiv.org/pdf/2207.07861.pdf) | [Github](https://github.com/yanjh97/TransGrasp)
5. [A Tale of Two Features : SD-Dino](https://arxiv.org/pdf/2305.15347.pdf) | [Github](https://github.com/facebookresearch/dinov2)


## Dataset Links
1. [GraspNet](https://graspnet.net)
2. [HouseCAT6D]()
3. [CO3D](https://ai.facebook.com/datasets/CO3D-dataset/)
4. [LineMod](https://bop.felk.cvut.cz/datasets/)
