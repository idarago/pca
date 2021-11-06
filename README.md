# Principal Component Analysis

Principal Component Analysis (PCA) is a dimension-reduction algorithm. The idea is to use the singular value decomposition of a data matrix to obtain the directions that explain the most of the variance in the data. This boils down to finding the eigenvalue decomposition of the covariance matrix.

# Applications

A toy case is shown to illustrate the fact that PCA are the directions which explain most of the variance, in order (```toycase.py```).

<p align="center">
  <img src="https://github.com/idarago/pca/blob/main/toydata.png" />
</p>

An example where PCA can be used for dimension reduction and classification is shown for the Iris dataset (```irisdataset.py```).

<p align="center">
  <img src="https://github.com/idarago/pca/blob/main/irisdataset.png" />
</p>

We also show how to use PCA for image compression (```imagecompression.py```).

<p align="center">
  <img src="https://github.com/idarago/pca/blob/main/pca_image_compression.png" />
</p>
