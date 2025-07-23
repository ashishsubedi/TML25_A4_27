# Assignment 4 - Explainability
## Task 1 - Network Dissection

The goal of this task is to figure out which specific units of a model correspond with specific concepts that were learned by the model using Network Dissection.

For this assignment the entire process of network dissection was done automatically by the tool [CLIP-Dissect](https://github.com/Trustworthy-ML-Lab/CLIP-dissect).
Network dissection was performed on the last three layers of two different models:
- ResNet18 trained on ImagNet
- ResNet18 trained on places 365

The code that was used is found in `task3/analyze_models.ipynb`.
The full report for this task is available in `reports/task1.pdf`.


## Task 2 – Grad‑CAM, AblationCAM & ScoreCAM

The goal of Task 2 is to visualize and analyze which parts of each of the 10 provided images influenced the ResNet‑50 classifier’s top‑1 decision by comparing three complementary attribution methods:

1. **Grad‑CAM** – uses the gradients flowing into the last convolutional layer to produce a coarse localization heatmap.  
2. **AblationCAM** – systematically “ablates” (masks out) each feature map in the target layer to measure its effect on the output score, producing a more diffuse attribution map.  
3. **ScoreCAM** – generates attribution by scoring the effect of each upsampled activation map on the output, yielding sharper, higher‑contrast explanations.


The report of the task is in `reports/task2.pdf`
The output of the task is in `task2`. It contains 2 folder: `output_gradcam_mask_v2` which contains greyscale heatmap of the `GradCAM` and `output_gradcam_v2` contains output of each CAM methods separted by the directory.

## Task 3 - LIME

The goal of this task is to derive an explanation for image classification using Local Interpretable Model-agnostic Explanations (LIME). Given for the task are an image classifier (ResNet 50 trained on ImageNet) and 10 images of which the classification should be explained.
For this task LIME was performed according to this tutorial: [LIME-Tutorial](https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb).

The code that was used is found in `task3/lime.ipynb`.
The full report for this task is available in `reports/task2.pdf`.
The cropped imaes highlighting the parts of the image the classifier deemed to be most important can be found in `task3/explanation`.


## Task 4 - Comparative Analysis of Grad-CAM and LIME with IoU Scores

The goal of Task 4 is to compare and analyze the alignment between Grad-CAM and LIME, two fundamentally different explainability methods applied to the same 10 ImageNet images. The analysis is supported by quantitative Intersection over Union (IoU) scores to precisely measure the degree of agreement between the two methods' explanations.

The notebook is present in `task4` directory.
The full report is available in `reports/task4.pdf`.
