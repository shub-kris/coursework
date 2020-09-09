# Don't use any other packages

"""'
TODO fill in your names:
group member 1: Bozidar Antic
group member 2: Shubham Krishna

"""
#####
import tensorflow as tf

# don't change this
tf.random.set_seed(1234)

# --------------------------------- Instructions-------------------------------#
# - Use only tensorflow functions for calculations.
# - Please clearly indicate the part number while printing the results.
# - When using random tensors as inputs, please also print the inputs as well.
# - Part (a) is already done for your reference.
# -----------------------------------------------------------------------------#

###############################################################################
# 1a (0 point): Create two random 0-d tensors x and y of any distribution.
# Create a TensorFlow object that returns x + y if x > y, and x - y otherwise.
# Hint: look up tf.cond()
# I do the first problem for you
###############################################################################
x = tf.random.uniform([])
y = tf.random.uniform([])
result = tf.cond(x > y, lambda: tf.add(x, y), lambda: tf.subtract(x, y))
tf.print("\npart (a): ", [x, y, result])
###############################################################################
# 1b (1 point): Create two random 0-d tensors x and y from a normal distribution.
# Return x / y if x < y, x * y if x > y, x^2+y^2 otherwise.
# Hint: Look up tf.case().
###############################################################################
x = tf.random.normal([])
y = tf.random.normal([])
r = tf.case(
    [(x < y, lambda: tf.divide(x, y)), (x > y, lambda: tf.multiply(x, y))],
    default=lambda: tf.add(tf.pow(x, 2), tf.pow(y, 2)),
)
tf.print("\npart (b): ", [x, y, r])
###############################################################################
# 1c (1 point): Create the tensor x of the value [[1, -4, -1], [0, 3, 2]]
# and y as a tensor of ones with the same shape as x.
# Return a boolean tensor that yields Trues if absolute value of x equals
# y element-wise.
# Hint: Look up tf.equal().
###############################################################################
x = tf.constant([[1, -4, -1], [0, 3, 2]])
y = tf.ones_like(x)
result = tf.equal(tf.abs(x), y)
tf.print("\npart (c): ", result)
###############################################################################
# 1d (1 point): Create a tensor x having 11 elements with random uniform numbers
# between -1 and 1
# Get the indices of elements in x which are postive.
# Hint: Use tf.where().
# Then extract elements whose values are positive.
# Hint: Use tf.gather().
###############################################################################
x = tf.random.uniform([11], minval=-1, maxval=1)
a = tf.where(x > 0)
b = tf.gather(x, a)
tf.print("\npart (d): ", [x, a, b])
###############################################################################
# 1e (2 points): Create two tensors x and y of shape 10 from any distribution,
# as long as they are from the same distribution.
# Use tf.less() and tf.cond() to return:
# - The mean squared error of (x - y) if the average of all elements in (x - y)
#   is negative, or
# - The sum of absolute value of all elements in the tensor (x - y) otherwise.
# Hint: Look up in TF documentation for methods to compute mean and sum
###############################################################################
x = tf.random.normal([10])
y = tf.random.normal([10])
avg = tf.reduce_mean(x - y)
result = tf.cond(
    avg < 0,
    lambda: tf.reduce_mean(tf.math.squared_difference(x, y)),
    lambda: tf.reduce_sum(tf.abs(x - y)),
)
tf.print("\npart (e): ", [avg, result])
###############################################################################
# 1f (2 points): Create two random 2-d tensors x and y both of size 3 x 4.
# - Concatenate x and y in axis 0  if the sum of all elements of x is greater
#   than the sum of all elements of y
# - Otherwise, Concatenate x and y in axis 1
# Hint: Use tf.concat()
###############################################################################
x = tf.random.normal([3, 4])
y = tf.random.normal([3, 4])
result = tf.cond(
    tf.reduce_sum(x) > tf.reduce_sum(y),
    lambda: tf.concat([x, y], 0),
    lambda: tf.concat([x, y], 1),
)
tf.print("\npart (f): ", [result])
###############################################################################
# 1g (3 points): We want to find the pseudo inverse of a matrix A
# Create a 3x3 tensor A = [[1,2,3],[2,3,7],[7,8,9]]
# Find the transpose of A (Atrans)
# Calculate the matrix B = (Atrans x A)
# Take the inverse of B (Binv)
# Compute the pseudo inverse matrix A_pinv = Binv x Atrans
# Find the inverse of A (A_inv) and print both A_inv and A_pinv
###############################################################################
A = tf.constant([[1, 2, 3], [2, 3, 7], [7, 8, 9]], dtype=tf.float32)
Atrans = tf.transpose(A)
B = tf.matmul(Atrans, A)
Binv = tf.linalg.inv(B)
A_pinv = tf.matmul(Binv, Atrans)
A_inv = tf.linalg.inv(A)
tf.print("\npart (g): ", [A_inv, A_pinv])
