Machine Learning Lab
====================

A collection of small, focused machine learning experiments using popular algorithms and real-world datasets
Each script is self-contained and intended for learning and lab work.

Folder Structure
----------------

<pre>
.
├── DATASETS/
│   ├── Iris.csv
│   ├── heart.csv
│   ├── winequality-red.csv
│   ├── kc_house_data.csv
│   └── house_data_cleaned.csv
└── PROGRAMS/
    ├── decision_tree.py
    ├── Feature-wise histograms by species.py
    ├── kmeans_heart.py
    ├── knn_iris.py
    ├── knn_wineclass.py
    ├── lr_heart_cleave.py
    ├── mlp_iris.py
    ├── mlr_california.py
    ├── mlr_syn.py
    ├── mlr.py
    ├── Species-wise petal length by petal width.py
    ├── stats_basic.py
    └── stats_ulr.py
</pre>

What’s Included
---------------

Regression
- mlr_syn.py – Multiple Linear Regression on synthetic house prices
- mlr.py, mlr_california.py – Multiple Linear Regression on real housing data
- ulr.py / stats_ulr.py – Simple/univariate linear regression demos

Classification
- knn_iris.py – k-NN on Iris dataset
- knn_wineclass.py – k-NN on wine quality data
- lr_heart_cleave.py – Logistic regression on heart disease data
- mlp_iris.py – Simple MLP (neural network) on Iris dataset

Clustering
- kmeans_heart.py – K-Means clustering on heart dataset

Visualization / EDA
- decision_tree.py – Decision tree example
- Feature-wise histograms by species.py – Histograms for Iris features
- Species-wise petal length by petal width.py – Scatter plots by species
- stats_basic.py – Basic statistics utilities

Each script typically:
- Loads a CSV from DATASETS/
- Preprocesses data
- Trains a model
- Prints metrics and/or shows plots

Setup (Windows)
---------------

1. Create and activate virtual environment

    python -m venv myenv
    myenv\Scripts\activate

2. Check if pip is installed

    pip --version

   If pip is not installed:

    python -m ensurepip --upgrade

3. Install required packages

    pip install numpy
    pip install scipy
    pip install pandas
    pip install matplotlib
    pip install seaborn
    pip install scikit-learn

   (Optionally, you can also install "statsmodels" if some regression scripts use it.)

Make sure the corresponding dataset CSVs are present in the DATASETS folder with the expected filenames.
