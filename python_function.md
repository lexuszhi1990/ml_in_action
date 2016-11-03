numpy.tile(A, reps)[source]
Construct an array by repeating A the number of times given by reps.

c = np.array([1,2,3,4])
>>> np.tile(c,(4,1))
array([[1, 2, 3, 4],
       [1, 2, 3, 4],
       [1, 2, 3, 4],
       [1, 2, 3, 4]])

numpy.argsort(a, axis=-1, kind='quicksort', order=None)[source]
Returns the indices that would sort an array.

>>> x = np.array([3, 1, 2])
>>> np.argsort(x)
array([1, 2, 0])

numpy.sum(a, axis=None, dtype=None, out=None, keepdims=False)[source]
Sum of array elements over a given axis.
>>> np.sum([[0, 1], [0, 5]], axis=0)
array([0, 6])
>>> np.sum([[0, 1], [0, 5]], axis=1)
array([1, 5])
