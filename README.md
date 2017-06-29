# Image-Inpainting 
Image inpainting is the process of reconstructing lost or deteriorated parts of images and videos.
This project implements two algorithms. 
1. Examplar based image inpainting:
The basic algorithm used in this project was proposed by Criminisi et al., whose idea was to restore the missing information  at  patch  level and encourage linear structures to  be synthesized first. One important observation of this algorithm is the criticality of patch filling order in the contour. 

2. Sparsity based image inpainting 
Examplar based image inpainting has some limitations.To address these limitations, sparsity was introduced. This concept was learnt from Xu and Sun’s work in 2010 and it represents the sparseness of patch similarities with its neighboring patches. The main theory is based on the sparser distribution of structures than textures in an image, due to structures are distributed in 0-D points or 1-D curves while textures are spread in the 2-D sub-region.

