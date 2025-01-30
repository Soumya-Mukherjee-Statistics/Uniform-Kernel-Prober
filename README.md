# Uniform Kernel Prober (UKP)

* This repository contains the code to replicate the results in the paper: "Uniform Kernel Prober".

* There are two sets of experiments - One on ReLU networks trained on MNIST data and the other on pretrained ImageNet networks. Pairwise distances between representations are presaved in this GitHub repository, but if one wishes to recompute these distances or access the network representations directly, the corresponding network architectures are required to be loaded or trained before the distances can be recomputed. Loading/training models and recomputing distances are expensive from both computational and memory perspectives.

* Code files are a mix of .py and .ipynb files. The most compute-intensive components were run on Google Colab using a single A100 GPU.

* The main definitions of different distance measures, including UKP, is in distance_functions_final.py . Note that the UKP_dist and GULP_dist functions in distance_functions actually return the squared UKP and GULP distances.

* The folders Figures and Appendix figures contains the figures used in the main paper and the Appendix.

* The two sets of experiments are separated out in terms of code into two folders - imagenet_experiments and mnist_experiments.

* Remaining files at the outermost level are LaTeX/PDF files corresponding to the ICML paper. The paper is provided in PDF format as icml.pdf and in tex format as icml.tex . 

## MNIST Experiments (`mnist_experiments/`)
* Distances between fully-connected ReLU networks of varying widths and depths are saved in `distances/widthdepth/`.
* To recompute these distances, first clear the folder `distances/widthdepth/`
* Run fit_model.ipynb.to train ReLU networks
* Run compute_reps.ipynb to compute the representations for these networks.
* Run compute_dists.ipynb to compute pairwise distances based on various distance measures.
* Run separating_distance_into_separate_files.ipynb to categorize distances based on distance measures for use in further code files.
* Run Generalization part 1.ipynb to perform first part of computations to obtain Figure 1 in the paper.
* Run Generalization part 2.ipynb to perform second part of computations to obtain Figure 1 in the paper.
* Run plots.ipynb to generate all plots in Section 8.1 of the Supplementary Materials (Note: Running the correlation plots takes a long time, one should pick and choose the required plots).

## ImageNet Experiments (`imagenet_experiments/`)
* Distances between pretrained state-of-the-art ImageNet networks taken from https://pytorch.org/vision/stable/models.html#classification are saved in `distances_torch/val/pretrained/`.
* To recompute these distances, first clear the folders `distances_torch/val/pretrained`.
* Go to https://image-net.org/challenges/LSVRC/2012/2012-downloads.php and download "Validation images (all tasks). 6.3GB. MD5: 29b22e2961454d5413ddabcf34fc5622". Extract this and then cd into the val subfolder. Paste this into the terminal "wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash" to categorize validation images into folders named by categories. Paste this path into the file `compute_reps.ipynb` in the ' dataset = datasets.ImageFolder(f"/content/drive/MyDrive/UKP/imagenet_experiments/datasets/ImageNet/{subset}/", transform=transform) ' line. 
* Run compute_reps.ipynb to compute the representations for these networks.
* Run compute_dists.ipynb to compute pairwise distances based on various distance measures.
* Run separating_distance_into_separate_files.ipynb to categorize distances based on distance measures for use in further code files.
* Run plots.ipynb to generate Figure 2 in the main paper and all plots in Section 8.2 of the Supplementary Materials (Note: Running the correlation plots takes a long time, one should pick and choose the required plots).
