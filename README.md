# SA3D-Segmentation
3D Gaussian Splatting Segmentation
Technical Details: The proposed framework operates in two stages to achieve high-fidelity 3D segmentation.
Stage 1 (Robust 2D Masking): Leverages a SAM2-based foundation model for initial proposals, enhanced by a novel multi-view attention mechanism (combining Self and Cross-Attention) to retrieve and fuse features from historically consistent views. A matte-based refinement step is applied for boundary precision.
Stage 2 (3D Consistent Lifting): Lifts 2D consistencies to 3D space using an adaptive view-weighted loss that prioritizes reliable observation angles. Finally, a boundary-aware Gaussian densification strategy is employed to optimize geometry and refine edge details effectively.
<img width="1620" height="886" alt="image" src="https://github.com/user-attachments/assets/7fc3cb23-e88c-40fb-acd3-74479060da08" />
<img width="2437" height="433" alt="image" src="https://github.com/user-attachments/assets/3e9b58c0-2d86-4daf-a07f-257aaa4de15d" />

# Installation
The installation is similar to [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).

```bash
git clone https://github.com/bwv1007/SA3D-Segmentation.git
```

```bash
cd SA3D-Segmentation;
```

Then install the dependencies:
```bash
conda env create --file environment.yml
conda activate gaussian_splatting_sa3d
```

Install SAM:
```bash
cd third_party;
git clone git@github.com:facebookresearch/segment-anything.git 
cd segment-anything; pip install -e .
mkdir sam_ckpt; cd sam_ckpt
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

