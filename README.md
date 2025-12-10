# SA3D-Segmentation
3D Gaussian Splatting Segmentation
Technical Details: The proposed framework operates in two stages to achieve high-fidelity 3D segmentation.
Stage 1 (Robust 2D Masking): Leverages a SAM2-based foundation model for initial proposals, enhanced by a novel multi-view attention mechanism (combining Self and Cross-Attention) to retrieve and fuse features from historically consistent views. A matte-based refinement step is applied for boundary precision.
Stage 2 (3D Consistent Lifting): Lifts 2D consistencies to 3D space using an adaptive view-weighted loss that prioritizes reliable observation angles. Finally, a boundary-aware Gaussian densification strategy is employed to optimize geometry and refine edge details effectively.
<img width="1620" height="886" alt="image" src="https://github.com/user-attachments/assets/7fc3cb23-e88c-40fb-acd3-74479060da08" />
<img width="1115" height="448" alt="image" src="https://github.com/user-attachments/assets/60ec664c-255b-4a8c-b15b-bec4d6352117" />
<img width="1246" height="570" alt="image" src="https://github.com/user-attachments/assets/9002f697-fca2-4f39-af80-4fdd2bfcfb4a" />

