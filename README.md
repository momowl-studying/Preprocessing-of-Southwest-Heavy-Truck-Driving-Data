# Preprocessing-of-Southwest-Heavy-Truck-Driving-Data
This includes the driving data of the 18-ton heavy-duty trucks produced by Dongfeng Liuzhou Automobile Co., Ltd. during their testing in the southwest region, as well as the preprocessing of these driving data and the clustering analysis of driving conditions.
The steps of data preprocessing are as follows:
1. Pulse processing (pulsedata.py)
2. Detect missing values and perform linear interpolation (DataPreprocess.py)
3. Handle abnormal gradients (DataPreprocess.py)
4. Mark and set values of vehicle speed that are less than 10 as 0 (DataPreprocess.py)
5. Delete consecutive zero data at idle speed (DataPreprocess.py)
6. Moving average filtering (Meanfilter.py)
7. Segment data and extract features (characterindex.py)
8. Merge features of dataset 1 and dataset 2
9. Optimize initial clustering centers using PSO algorithm and cluster using K-Means algorithm (PSO_KMeans.py)
Use REF_parameter_selection.py to determine the optimal number of feature subsets for all standard features and fluctuation features (REF parameter selection)
The training process of the LVQ model is carried out using the "lvqnetwork.m" file.
