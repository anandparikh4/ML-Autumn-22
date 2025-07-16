'''
	Machine Learning Assignment-2
	Question-1
	Team members:
		Soni Aditya Bharatbhai (20CS10060)
		Anand Manojkumar Parikh (20CS10007)
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

# This function standardizes the 2d feature matrix of size 4177 * 8
# After standardizing the data, each column vector will have mean = 0 and standard deviation = 1
# The formula used is as follows:
# arr[i][j] = (arr[i][j] - mean of values in jth column)/(standard deviation of values in jth column)
def standardize_data(features):
	column_means = np.mean(features,axis=0)
	column_stddev = np.std(features,axis=0)
	rows,cols = features.shape
	for i in range(rows):
		for j in range(cols):
			features[i][j] = (features[i][j] - column_means[j])/column_stddev[j]
	return features

# This function uses the in-built function from sklearn library to perform PCA
# Input is the standardized feature matrix of size 4177*8
# The number of components is selected such that 95% of total variance is preserved 
def perform_pca(x_data):
	pca = PCA(0.95)
	pca.fit(x_data)
	x_data = pca.transform(x_data)
	return x_data

# This function is called during each iteration of the k-means algorithm
# It assigns each data point to the closest centroid
# Closeness is measured using the 2-norm of the vector features[i] - centroids[j]
# For each of the data points in the features matrix we compute its distance from each of the k centroids
# Then we assign the index of the closest centroid to this data point
# centroids: Matrix containing k centroids 
# features: Matrix containing the 4177 data points
# # cluster_assgn: cluster_assgn[i] = the index of the centroid closest to data point features[i]
def update_cluster_assignment(centroids,features,cluster_assgn):
	num_samples = features.shape[0]
	k = centroids.shape[0]
	for i in range(num_samples):
		min_dist_cluster = -1
		min_dist = -1*np.inf
		for j in range(k):
			dist = np.linalg.norm(features[i] - centroids[j])
			if (min_dist_cluster == -1) or (min_dist > dist):
				min_dist_cluster = j
				min_dist = dist
		cluster_assgn[i] = min_dist_cluster
	return cluster_assgn

# This function is called during each iteration of the k-means algorithm
# Based on the updated cluster assignment we now update the centroids of each cluster
# The updated centroid of cluster j is computed by taking the mean of all the data points belonging to this cluster
# So to compute the i-th component of centroid of cluster j we take arithmetic of mean
# of the i-th component of all the data points for which cluster_assgn is j
# We do this to compute each component of the centroid, for each of the k centroids
def update_centroids(centroids,features,cluster_assgn):
	k = centroids.shape[0]
	for i in range(k):
		idx = np.where(cluster_assgn == i)
		cluster_points = features[idx]
		if len(cluster_points)==0:
			centroids[i] = np.zeros(features.shape[1])
		else:
			centroids[i] = np.mean(cluster_points,axis=0)
	return centroids

# This function runs the iterative k-means algorithm where the initial k centroids are
# obtained by choosing k data-points randomly from the features matrix.
# The convergence criterion is chosen that the clusters centroids do not update much during centroid update
# Convergence criterion: sqrt(sum over k centroids(square of 2-norm(updated centroid - old centroid))) < epsilon
# epsilon is taken as 1e-7
# k: number of clusters
# labels: the true number of rings obtained from the abalone dataset
# features: the 2d matrix containing 4177 data points after application of PCA
# After the k-means algorithm converges this function computes the NMI value and returns it
def k_means_clustering(k,features,labels):
	num_samples = features.shape[0]
	k_indices = np.random.randint(0,num_samples,k)
	centroids = features[k_indices].copy()
	cluster_assgn = np.zeros(num_samples,dtype=np.int64)
	flag = False
	epsilon = 1e-7 #can be changed as per different convergence criterion
	while(1):
		old_centroids = centroids.copy()
		cluster_assgn = update_cluster_assignment(centroids,features,cluster_assgn)
		centroids = update_centroids(centroids,features,cluster_assgn)
		if(not flag):
			flag = True
		else:
			dist = np.linalg.norm(old_centroids - centroids)

			if dist < epsilon:
				break
	H_c = compute_entropy(cluster_assgn)
	H_y = compute_entropy(labels)
	H_y_c = 0
	cluster_labels = np.unique(cluster_assgn)
	for i in cluster_labels:
		indices = np.where(cluster_assgn == i)
		num_points = labels[indices].shape[0]
		H_y_c += (num_points/cluster_assgn.shape[0])*compute_entropy(labels[indices])
	I_y_c = H_y - H_y_c
	NMI = (2*(I_y_c))/(H_c + H_y)
	print("Value of k =",k)
	print("NMI value =",NMI)
	print()
	return NMI

# This function computes the entropy of the 1d array labels
# The entropy is calculated using the formula: sum of all classes(-pi*log(pi))
# where pi = probability that a chosen label belongs to class i
def compute_entropy(labels):
	total_samples = labels.shape[0]
	elements,freq = np.unique(labels,return_counts=True)
	entropy = 0
	for i in range(elements.shape[0]):
		prob_i = freq[i]/total_samples
		entropy += (-1*prob_i)*np.log2(prob_i)
	return entropy

def main():
	column_names = ['sex','length','diameter','height','whole_weight','shucked_weight','viscera_weight', \
					'shell_weight','rings']
	data = pd.read_csv('abalone.data',names=column_names)
	
	# features is a 2d matrix which consists of the first 8 columns of the dataframe 'data'
	features = data.drop('rings',axis=1).to_numpy()
	# labels is a 1d array consisting of the number of rings corresponding to each abalone
	labels = data['rings'].to_numpy()
	
	# since sex is a categorical variable with values M,F or I we map these values to the numbers 0,1,2 respectively
	features[features == 'M'] = 0
	features[features == 'F'] = 1
	features[features == 'I'] = 2
	features = features.astype(np.float64)
	
	# standardize the feature matrix and then perform pca on it
	features = standardize_data(features)
	features = perform_pca(features)
	
	# randomly permute the data points of the feature matrix and their corresponding labels
	permute = np.random.permutation(np.arange(features.shape[0]))
	features = features[permute]
	labels = labels[permute]

	# plot the 3-dimensional data obtained from the 2d matrix features
	# the shape of features matrix is 4177*3 implying that each data point has 3 components
	fig = plt.figure()
	my_cmap = plt.get_cmap('twilight_r')
	ax = fig.add_subplot(111,projection='3d')
	x = features[:,0]
	y = features[:,1]
	z = features[:,2]
	plt.title('PCA Output')
	plot = ax.scatter3D(x,y,z,c=labels,cmap=my_cmap)
	ax.set_xlabel('x-axis',fontweight='bold')
	ax.set_ylabel('y-axis',fontweight='bold')
	ax.set_zlabel('z-axis',fontweight='bold')
	cbar = fig.colorbar(plot, ax = ax, shrink=0.5,aspect = 5)
	cbar.ax.set_title('No. of Rings')
	plt.show()
	# plt.savefig('pca_output.png',orientation='landscape')
	plt.close(fig)

	# run k means algorithm for k in range 2<=k<=8.
	# We plot k vs NMI graph and then report the value of k for which NMI is maximum
	k_vs_nmi = []
	max_k = 2
	for i in range(2,9):
		nmi_val = k_means_clustering(i,features,labels)
		k_vs_nmi.append(nmi_val)
		if i>2 and nmi_val > k_vs_nmi[max_k-2]:
			max_k = i
	
	print('The value of k for which NMI is maximum =',max_k)
	print('Corresponding maximum NMI value =',k_vs_nmi[max_k-2])
	
	plt.plot(range(2,9),k_vs_nmi,'bx-')
	plt.xlabel('Value of k') 
	plt.ylabel('Normalized Mutual Information (NMI)') 
	plt.show()
	# plt.savefig('k_vs_nmi_graph.png')

if __name__ == "__main__":
	main()