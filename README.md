# Sorghum Semantic Segmentation using machine learning and hyperspectral imaging
sorghum semantic segmentation project related code and data

- sorghum pixel classification dataset for background, leaf, stalk, and panicle
  - `sorghum_features.csv` (7560*243)
  - `sorghum_labels.csv`  (7560*1)

- maize pixel classification dataset for background, leaf, stalk, and tassel
  - `maize_features.csv` (4000*243)
  - `maize_labels.csv`  (4000*1)

- feature selection results
  - `feature_selection.xlsx`

- code for training and testing 7 machine learning methods in R
  - `Analysis_7methods.R`

- code for training Artificial neural networks (ANNs) in python
  - `Analysis_ANNs.py` (training)
  - `Analysis_ANNs_predict.py` (prediction)

- traits extracted from segmented sorghum images
  - `pheno_height.csv` (phenotypes for plant height)
  - `pheno_panicle.csv` (phenotypes for panicle size)
  - `pheno_stalk.csv` (phenotypes for stalk size)
  - `pheno_leaf.csv` (phenotpes for leaf size)
  - `pheno_panicleleaf.csv` (phenotypes of the ratio of panicle and leaf size)
  - `pheno_stalkleaf.csv` (phenotypes of the ratio of stalk and leaf size)
  - `pheno_paniclestalk.csv` (phenotypes for the ratio of panicle stalk size)

- Sorhgum and maize hyperspectral cubes in numpy array for testing
  - maize_hyperspectral_cube.npy (495*320*243)
  - sorghum_hyperspectral_cube1.npy (561*320*243)
  - sorghum_hyperspectral_cube2.npy (560*320*243)
  
