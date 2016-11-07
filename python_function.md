### numpy.tile(A, reps)[source]

Construct an array by repeating A the number of times given by reps.

c = np.array([1,2,3,4])
>>> np.tile(c,(4,1))
array([[1, 2, 3, 4],
       [1, 2, 3, 4],
       [1, 2, 3, 4],
       [1, 2, 3, 4]])

### numpy.argsort(a, axis=-1, kind='quicksort', order=None)[source]

Returns the indices that would sort an array.

>>> x = np.array([3, 1, 2])
>>> np.argsort(x)
array([1, 2, 0])

### numpy.sum(a, axis=None, dtype=None, out=None, keepdims=False)[source]

Sum of array elements over a given axis.
>>> np.sum([[0, 1], [0, 5]], axis=0)
array([0, 6])
>>> np.sum([[0, 1], [0, 5]], axis=1)
array([1, 5])

### RandomState.rand(d0, d1, ..., dn)
Random values in a given shape.

np.random.rand(3,1)
array([[ 0.3929263 ],
       [ 0.74411272],
       [ 0.02455391]])


### numpy.nonzero(a)[source]

Return the indices of the elements that are non-zero.

Returns a tuple of arrays, one for each dimension of a, containing the indices of the non-zero elements in that dimension. The values in a are always tested and returned in row-major, C-style order. The corresponding non-zero values can be obtained with:

###  Python图表绘制：matplotlib绘图库入门
http://blog.csdn.net/ywjun0919/article/details/8692018


Fig.add_subplot(111).plot(X1, Y1, X2, Y2) # Create a Line2D instance in the axes
