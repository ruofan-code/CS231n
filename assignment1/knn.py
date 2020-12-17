import random
import numpy as np
from assignment1.cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from assignment1.cs231n.classifiers.k_nearest_neighbor import KNearestNeighbor

# set default size of plots
plt.rcParams['figure.figsize'] = (10.0, 8.0) #设置图像显示的最大范围
plt.rcParams['image.interpolation'] = 'nearest' #差值方式
plt.rcParams['image.cmap'] = 'gray' #灰度空间

cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
# enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
for y, cls in enumerate(classes):
    # 输入一个矩阵，返回了其中非零元素的位置.
    idxs = np.flatnonzero(y_train == y)
    # numpy.random.choice(a, size=None, replace=True, p=None)
    # 从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
    # replace:True表示可以取相同数字，False表示不可以取相同数字
    # 数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

#取样，前5000
num_training = 5000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
#该reshape函数返回带有修改形状的参数，而该 ndarray.resize方法会修改数组本身
#如果在 reshape 操作中将 size 指定为-1，则会自动计算其他的 size 大小
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)




# Create a kNN classifier instance.
# Remember that training a kNN classifier is a noop:
# the Classifier simply remembers the data and does no further processing
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

# Test your implementation:
dists = classifier.compute_distances_two_loops(X_test)
print(dists.shape)

# We can visualize the distance matrix: each row is a single test example and
# its distances to training examples
plt.imshow(dists, interpolation='none')
plt.show()

# Now implement the function predict_labels and run the code below:
# We use k = 1 (which is Nearest Neighbor).
y_test_pred = classifier.predict_labels(dists, k=1)

# Compute and print the fraction of correctly predicted examples
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

y_test_pred = classifier.predict_labels(dists, k=5)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

# 现在通过使用部分矢量化来加快距离矩阵的计算一圈。 实现函数compute_distances_one_loop，然后运行下面的代码：
dists_one = classifier.compute_distances_one_loop(X_test)

# 为确保矢量化实现正确无误，请确保同意朴素的实现。 有很多方法可以决定是否两个矩阵相似； 最简单的方法之一就是Frobenius规范。
# 以防万您之前从未见过，两个矩阵的Frobenius范数是平方所有元素之差的平方和的根；
# 换句话说，重塑将矩阵转换为向量，并计算它们之间的欧式距离。
difference = np.linalg.norm(dists - dists_one, ord='fro')
print('Difference was: %f' % (difference, ))
if difference < 0.001:
  print('Good! The distance matrices are the same')
else:
  print('Uh-oh! The distance matrices are different')


# Now implement the fully vectorized version inside compute_distances_no_loops
# and run the code
dists_two = classifier.compute_distances_no_loops(X_test)

# check that the distance matrix agrees with the one we computed before:
difference = np.linalg.norm(dists - dists_two, ord='fro')
print('Difference was: %f' % (difference, ))
if difference < 0.001:
  print('Good! The distance matrices are the same')
else:
  print('Uh-oh! The distance matrices are different')


# Let's compare how fast the implementations are
def time_function(f, *args):
  """
  Call a function f with args and return the time (in seconds) that it took to execute.
  """
  import time
  tic = time.time()
  f(*args)
  toc = time.time()
  return toc - tic

two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
print('Two loop version took %f seconds' % two_loop_time)

one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
print('One loop version took %f seconds' % one_loop_time)

no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
print('No loop version took %f seconds' % no_loop_time)

# you should see significantly faster performance with the fully vectorized implementation


num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
# 将训练数据分成多个折叠。 拆分后，X_train_folds和y_train_folds每个都应是长度为num_folds的列表，
# 其中y_train_folds [i]是X_train_folds [i]中各点的标签向量
# 提示：查找numpy array_split函数。
pass
# split self.X_train to 5 folds
avg_size = int(X_train.shape[0] / num_folds)  # will abandon the rest if not divided evenly.
for i in range(num_folds):
    X_train_folds.append(X_train[i * avg_size: (i + 1) * avg_size])
    y_train_folds.append(y_train[i * avg_size: (i + 1) * avg_size])
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
# 包含我们发现k的不同值的精度的字典,运行交叉验证时。 运行交叉验证后，
# k_to_accuracies [k]应该是长度num_folds的列表，以给出不同的使用k的值时发现的精度值。
k_to_accuracies = {}

################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
# 执行k倍交叉验证，以找到k的最佳值。 对于每个可能的k值，运行k-nearest-neighbor算法num_folds次
# 在每种情况下，您会使用除其中一个折叠以外的所有折叠作为训练数据，并且最后折叠作为验证集。 存储所有折叠和所有的精度
# 在k_to_accuracies字典中的个k值。
pass
for k in k_choices:
    accuracies = []
    print(k)
    for i in range(num_folds):
        X_train_cv = np.vstack(X_train_folds[0:i] + X_train_folds[i + 1:])
        y_train_cv = np.hstack(y_train_folds[0:i] + y_train_folds[i + 1:])
        X_valid_cv = X_train_folds[i]
        y_valid_cv = y_train_folds[i]

        classifier.train(X_train_cv, y_train_cv)
        dists = classifier.compute_distances_no_loops(X_valid_cv)
        accuracy = float(np.sum(classifier.predict_labels(dists, k) == y_valid_cv)) / y_valid_cv.shape[0]
        accuracies.append(accuracy)
    k_to_accuracies[k] = accuracies
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))


# plot the raw observations
for k in k_choices:
  accuracies = k_to_accuracies[k]
  plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()

# Based on the cross-validation results above, choose the best value for k,
# retrain the classifier using all the training data, and test it on the test
# data. You should be able to get above 28% accuracy on the test data.
best_k = 10

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
y_test_pred = classifier.predict(X_test, k=best_k)

# Compute and display the accuracy
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))