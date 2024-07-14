# Large-Scale-Computing

## 1. Parallelizing Laplace with MPI
The goal is to solve the Laplace equation for temperature distribution across a 2D grid using parallel computing.
-  parallelize the given serial python code by running on 4PE's on **Bridges-2**
-  Distributed work among PEs
-  Initialize Temperature Grid
-  Broadcasted Input
-  Iterated computation to synchronize PEs at each iteration using barriers, and exchanged boundary rows between adjacent PEs using **MPI** send and receive functions.

## 2. Pulsar Data Analysis with PySpark and MPI
- Analyzed pulsar signal data: Using **PySpark** and **MPI** for signal processing, data analysis, and clustering.
- Used PySpark for distributed processing: Utilizing **PySpark** and **Hadoop** for **RDD** **operations**, distributed computing, and map-reduce.
- Utilized **MPI** for **parallel computation**: Leveraging **MPI** and **OpenMPI** for parallel processing, message passing, and synchronization.
- Preprocess data: Employing **PySpark** and **NumPy** for data cleaning, data transformation, and ETL.
- Identify the most significant pulsar signals: Employing PySpark and KMeans from pyspark.mllib.clustering for signal detection, clustering, and significance testing.
- Employ PySpark for big data processing: Utilizing **PySpark**, **Hadoop**, and **Apache Spark** for big data, distributed processing, and RDD operations.

## 3. Identifying High Order Volume Postal Codes Around Cambridge using MySQL
- Identified postal codes around Cambridge, Massachusetts with the highest order volumes: Using **MySQL** for database querying and spatial functions for geographic calculations.
- Calculated the centroid of Cambridge: Utilizing MySQL's AVG function to determine the average longitude and latitude.
- Find postal codes within a 100 km radius: Using MySQL's ST_Distance_Sphere function for geographic distance calculations.
- Identified the top three postal codes with the highest number of orders: Leveraging SQL queries for grouping, ordering, and limiting results.
- Tools used include MySQL for database querying and spatial functions for geographic calculations.

## 4. Discovering the Structure of Higher Dimensional Data using Spark
- Performed clustering on a dataset of spatial coordinates to identify distinct groups and analyze their geometric properties: Using K-Means clustering to determine optimal clusters, filter outliers, and calculate distances.
- Utilized K-Means clustering to determine the optimal number of clusters: Leveraging **PySpark**'s pyspark.mllib.clustering.KMeans for clustering analysis.
- Filtered outliers and calculate distances to identify and characterize different shapes (e.g., cylinders, rectangular prisms) within the data: Employing PySpark for outlier detection and distance calculations.
- Used PySpark's distributed computing capabilities for efficient processing and clustering: Utilizing PySpark for big data processing and distributed computation.

## 5. Fashion MNIST Classification with Optimized CNN Architectures
- Imported **TensorFlow** and loaded the Fashion MNIST dataset.
- Reshaped and normalized the training and test images.
- Defined an initial model with Dropout layers.
- Created a model with BatchNormalization layers to reduce overfitting.
- Added additional convolutional layers and increased the number of neurons.
- Optimized the model by increasing filters and adding MaxPooling layers.
- Changed the optimizer to SGD to improve accuracy.
- Compiled the model with different parameters and optimizers **(SGD, Adam, RMSprop)**.
- Trained the model with various architectures and settings.
- Used learning rate schedules for fine-tuning the optimizer parameters.
- Achieved the best test accuracy by using SGD with a specific learning rate and momentum.
