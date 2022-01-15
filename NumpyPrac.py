import numpy as np

# call np.array on a valid Python list
array = np.array([[1, 2, 3], [4, 5, 6]])
print("Array 1:\n", array)

# initialize an array of all zeroes or ones
array = np.zeros((4, 4)) # where (3,4) is the shape of the array
print("Array 2:\n", array)

array = np.ones((3, 4, 5)) # where (3, 4, 5) is the shape of the array
print("Array 3:\n", array)

# initialize an array from a particular distribution
array = np.random.normal(loc=0.0, scale=1.0, size=(3,4))  # uses a normal distribution with a mean at 0 and standard deviation of 1.0
print(array)
#print("\nMean of array:", array_normal.mean())
#print("\nStandard deviation of array:", array_normal.std())
#print("\nMean of array along axis-0 (columns):", array_normal.mean(axis=0))

array = np.array([[[1], [2]], [[3], [4]], [[5], [6]]])
print("Data type:", array.dtype)
print("# dimensions:", array.ndim)
print("Shape/size of array:", array.shape)
print("# elements:", array.size)

array = np.arange(24).reshape(4, 6)

print("Array:\n", array)
print("Element 1, 2: ", array[1, 2]) # access one element

print("Column 2:", array[:, 1]) # access a column, using slice notation (:)
print("Row 1:", array[0]) # access a row

print("2nd-4th row, 1st-3rd col:\n", array[1:3, 0:2]) # partial rows and cols

# accessing elements using an array
print("Selecting one element from each row of array using indices in ind:")
inds = np.array([0, 2, 0, 1])
print(array[np.arange(4), inds])  # Prints "[ 0  8  12 19]"
print("-----")

# mathematical methods
x = np.array([[1,2],[3,4]])
print("x: \n", x)
print("-----")
print("Compute sum of all elements in x: ", np.sum(x))  # Compute sum of all elements; prints "10"
print("-----")
print("Compute sum of each column in x:", np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"
print("-----")
print("Compute sum of each row in x: ", np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]"
print("-----")
print("Compute subtraction: ", np.subtract(x, x))

# mathematical transformations
array = np.arange(9).reshape((3,3))
print("Array 1:\n", array)

print(array + 3) # add a constant to every value, broadcasting
array[0] *= 3 # add a constant to only one row
print(array)
print(np.power(array, 2)) # square every element

# reshaping
array = np.arange(9)
print("Array 0:\n", array)
print("Array 0 shape:", array.shape)
array = array.reshape(3, 3)
print("Array 0:\n", array)
print("Array 0 shape:", array.shape)
print()

array1 = np.arange(9).reshape((3,3))
array2 = np.arange(9, 18).reshape((3, 3))
print("Array 1:\n", array1)
print("Array 2:\n", array2)
stack_h = np.hstack((array1, array2))
stack_v = np.vstack((array1, array2))
print("Horizontally stacked array 1 and array 2:\n", stack_h)
print("Vertically stacked array 1 and array 2:\n", stack_v)

A = np.random.randint(10, size=(3, 4))
B = np.random.randint(10, size=(3, 4))
print("Matrix A:\n", A)
print("\nMatrix B:\n", B)
print("\nA+B:\n", A+B)
print("\nTranspose B:\n", B.transpose())
print("\nShape of transpose(B):", B.transpose().shape)
print("\nMatrix multiplication of A and transpose(B):\n", np.matmul(A,B.transpose()))

C = np.matmul(B,A.transpose())
print("\n Invert BA:", np.linalg.inv(C))
