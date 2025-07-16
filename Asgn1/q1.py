'''
	Question-1
	Machine Learning Assignment
	Group members:
		Soni Aditya Bharatbhai (20CS10060)
		Anand Manojkumar Parikh (20CS10007)
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
'''
	Node class:
		feature_index: index of feature the node tests
		threshold: threshold value of feature
		left: left child of node
		right: right child of node
		output_val: The value predicted by this node if it is a leaf node
'''
class Node():
	# constructor of node class
	def __init__(self, feature_index=None, threshold=None, left=None, right=None, output_val=None):
		self.feature_index = feature_index
		self.threshold = threshold
		self.left = left
		self.right = right
		self.output_val = output_val
	
class RegTree():
	# constructor of RegTree Class
	# root: root of the tree
	def __init__(self,root=None):
		self.root = root
		self.depth = 0
	
	'''
	find_best_split function: 
		samples- a numpy array of the training samples
		computes the best attribute and its corresponding threshold value
		that reduces variance in y by the maximum amount
	'''
	def find_best_split(self,samples):
		attr = samples[:,:-1]
		y_values = samples[:,-1]
		num_row, num_col = attr.shape
		split_dict = {}
		max_variance = 0.0
		# for all possible attributes we try to find the best threshold
		for col in range(num_col):
			idx_order = samples[:,col].argsort()
			samples = samples[idx_order]
			# sort samples with respect to the current attribute value
			attr = attr[idx_order]
			y_values = y_values[idx_order]
			
			# since we have continuous valued attributes we sort the samples
			# with respect to current attribute and take mean of
			# all possible adjacent attribute values in the sorted list of samples
			# we take this average value as the threshold
			for row in range(num_row-1):			
				val1 = attr[row][col]
				val2 = attr[row+1][col]
				if val1==val2:
					continue
				split_val = (val1+val2)/2.0

				var_tot = np.var(y_values) # initial variance in output labels
				var_left = np.var(y_values[0:row+1])*((row+1)/num_row)
				var_right = np.var(y_values[row+1:])*(1-((row+1)/num_row))
				# take weighted mean of variance of left and right children and compare with initial variance
				var_reduction = var_tot - (var_left+var_right)
				if var_reduction>max_variance:
					max_variance = var_reduction
					# split_dict stores the max_variance reduction,
					# the index of the feature chosen to split,
					# the threshold value of this feature,
					# the training samples in the left and right children after splitting
					split_dict['variance'] = max_variance
					split_dict['feature_index'] = col
					split_dict['threshold'] = split_val
					split_dict['left_samples'] = samples[:row+1]
					split_dict['right_samples'] = samples[row+1:]
		
		return split_dict

	'''
		build_tree:
			builds the tree if the training samples are stored in the numpy array samples
			returns root of the tree
	'''
	def build_tree(self,samples):
		num_rows = samples.shape[0] # num_rows stores number of samples

		# if no samples are there to split return None, base case		
		if num_rows == 0:
			return None	

		# single training sample present, return leaf node, base case
		if num_rows == 1:
			return Node(output_val=samples[0][-1])

		# split_dict stores the result of the best possible split computed
		# by the find_best_split function
		split_dict = self.find_best_split(samples)

		# if no splitting needed then return leaf node with output being mean of output labels of samples
		if len(split_dict)==0 or split_dict['variance']==0:
			return Node(output_val=np.mean(samples[:,-1]))

		# since we are creating a decision node we need to recursively build the left and right subtrees
		left_samples = split_dict['left_samples']
		right_samples = split_dict['right_samples']
		# recursive calls to build left and right subtrees
		l_node = self.build_tree(left_samples)	
		r_node = self.build_tree(right_samples)
		
		# return the decision node
		return Node(feature_index=split_dict['feature_index'],threshold=split_dict['threshold'],left=l_node,right=r_node)
	
	# recursive function that computes the depth of the tree
	# depth is defined as total number of nodes in the tree (decision nodes as well as leaf nodes)
	def find_depth(self,node):
		if node==None:
			return 0
		return 1+self.find_depth(node.left)+self.find_depth(node.right)

	# trains the regression tree on training samples
	# stores the root and depth of the tree
	def train_data(self,samples):
		self.root = self.build_tree(samples)
		self.depth = self.find_depth(self.root)

	# predict_y function outputs the prediction of the tree rooted at root_node
	# on the example	
	def predict_y(self,root_node,example):
		# if leaf node return output_val of leaf node as predicted value
		if root_node.left==None and root_node.right==None:
			return root_node.output_val

		# go to left or right child depending on the spliting condition
		# recursively call predict_y to predict the output label of example
		if example[root_node.feature_index] <= root_node.threshold:
			return self.predict_y(root_node.left,example)
		else:
			return self.predict_y(root_node.right,example)

	# predict_values function takes as argument a numpy array test_data (test examples)
	# it returns a numpy array containing the model's prediction of output value for these test examples
	def predict_values(self,test_data):
		predictions = np.empty(test_data.shape[0],dtype=np.float64)
		for i in range(test_data.shape[0]):
			predictions[i] = self.predict_y(self.root,test_data[i])
		return predictions
	
	# prune_tree performs pruning operation on the overfitted tree
	# takes as the input current node, its parent node, those training examples that will reach
	# the current node if we start from root and split the training data,  
	# all the test examples, a list containing the pairs (accuracy,depth) for different depths
	def prune_tree(self,node,par_node,train_samples,test_data,acc_vs_depth):

		# if current node is a leaf node or we are left with no training samples we return
		if node.output_val != None or train_samples.size==0:
			return

		# compute the output labels and error for the test samples with respect to the current tree
		curr_preds = self.predict_values(test_data)
		curr_err = self.find_error(test_data,curr_preds)

		# the approach is that we try and prune the current node by
		# making it a leaf node and assigning the output value of
		# this leaf node as average of output values of the training examples
		# that reach the current node. We find test accuracy of modified tree.
		# If test accuracy is improved we permanently prune of this node.
		# Else we recursively try and prune left and right subtrees.

		new_node = Node(output_val=np.mean(train_samples[:,-1]))
		old_root = self.root
		old_left = old_right = None
		#  change the tree if current node is made leaf node
		if par_node!=None:
			old_left = par_node.left
			old_right = par_node.right
		if par_node==None:
			self.root = new_node
		elif par_node.left == node:
			par_node.left = new_node
		elif par_node.right == node:
			par_node.right = new_node
		# find test error of this pruned tree
		new_preds = self.predict_values(test_data)
		new_err = self.find_error(test_data,new_preds)
		# if accuracy improves simply prune this node
		if new_err <= curr_err:
			new_depth = self.find_depth(self.root)
			# since depth of tree changes we store the new pair of accuracy,depth
			acc_vs_depth.append((new_err,new_depth)) 
			return
		else: 
			# since accuracy did not improve we restore initial tree structure
			# and then recursively prune left and right subtrees
			self.root = old_root
			if par_node != None:
				par_node.left = old_left
				par_node.right = old_right
			idx = node.feature_index
			threshold = node.threshold
			# spilt training examples based on attribute threshold
			l_train = np.array([row for row in train_samples if row[idx] <= threshold])
			r_train = np.array([row for row in train_samples if row[idx] > threshold])
			
			self.prune_tree(node.left,node,l_train,test_data,acc_vs_depth)
			self.prune_tree(node.right,node,r_train,test_data,acc_vs_depth)
	
	# find_error function: computes root mean square error of predicted output values
	# the test error is computed as sqrt (sum ((yi - fi)^2) / number of samples)
	# where yi is actual output label and fi is predicted output label 
	def find_error(self,test_samples,predictions):
		err = np.sqrt(np.sum(np.square(test_samples[:,-1]-predictions))/predictions.shape[0])
		return err

	# print_tree: performs breadth first search on the tree
	# prints the tree in top down fashion 
	# where each level is printed from leftmost node to rightmost node
	# attr_name contains names of attributes	
	# The invariant we maintain in the tree is that if the condition of decision node is true we go to left child
	# else we go to right child.  
	def print_tree(self,attr_name):
		node = self.root
		if node==None:
			return

		q = []
		q.append(node)
		level = 0
		while len(q)!=0:
			new_q = []
			idx = 0
			print('Level',end=' ')
			print(level,end='  ')
			while idx<len(q):
				curr = q[idx]
				if curr.left==None and curr.right==None:
					# leaf node just printed as output value
					print(curr.output_val,end='\t')
				else:
					print(attr_name[curr.feature_index],end='')
					if attr_name[curr.feature_index] == ' Size by Inch':
						print(' <= ',end='')
						print(curr.threshold,end='\t')
					else:
						print(' = yes',end='\t')
					
					new_q.append(curr.left)
					new_q.append(curr.right)
				idx += 1
			level += 1
			q = new_q
			print()

# main function
def main():
	
	df = pd.read_csv("Train_D_Tree.csv")
	df = df.iloc[:,1:] # remove restaurant names' column from df
	attr_name = []
	
	for col in df:
		attr_name.append(col)
		if type(df[col][0])==str: 
			df[col] = df[col].map(dict(yes=1,no=0)) 
			# yes is mapped to 1 and no is mapped to 0

	np_arr = df.to_numpy()
	min_error_root = None	# stores the root of the tree which gives best accuracy	
	min_error = np.inf	# stores the min error
	min_err_test = None # stores the test set for which we get best accuracy
	min_err_train = None # stores the training set for which we get best accuracy

	# train the tree by randomly splitting the samples as 70% train and 30% test
	for i in range(10):
		np.random.shuffle(np_arr)
		train_rows = (int)(0.7*np_arr.shape[0])
		train_data = np_arr[0:train_rows,:]
		test_data = np_arr[train_rows:,:]
		tree = RegTree()
		tree.train_data(train_data)
		pred_values = tree.predict_values(test_data)
		mse_error = tree.find_error(test_data,pred_values)
		if mse_error<min_error:
			min_error=mse_error
			min_error_root = tree.root
			min_err_test = np.array(test_data)
			min_err_train = np.array(train_data)

	# tree denotes the best accuracy tree we obtained in above loop
	tree = RegTree(root=min_error_root)
	# print the tree and the error
	print('Tree obtained:')
	tree.print_tree(attr_name) 
	print("\nTest Error = ",end='')
	print(min_error)
	print('\nDepth of tree = ',end='')
	dep = tree.find_depth(tree.root)
	print(dep)
	
	print('Depth of tree for which the model overfits = ',end='')
	print(dep)
	# prune the tree and store the pairs of accuracy,depth as depth changes during pruning
	acc_vs_depth = []
	# store initial accuracy,depth value
	acc_vs_depth.append((min_error,tree.find_depth(tree.root)))
	tree.prune_tree(tree.root,None,min_err_train,min_err_test,acc_vs_depth)


	print("Pruned Tree:")
	tree.print_tree(attr_name)
	pred_values = tree.predict_values(min_err_test)
	mse_error = tree.find_error(min_err_test,pred_values)
	print('Test error for pruned tree = ',end='')
	print(mse_error)
	print('Depth of the pruned tree = ',end='')
	print(tree.find_depth(tree.root))

	acc = np.array([i[0] for i in acc_vs_depth])
	depth = np.array([i[1] for i in acc_vs_depth])
	plt.scatter(depth,acc)
	plt.plot(depth,acc,color='red')
	plt.xlabel('Depth(number of nodes)')
	plt.ylabel('Root Mean Square Error')
	plt.savefig('output_plot.jpg')
if __name__ == "__main__":
	main()