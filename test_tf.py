import tensorflow as tf
import numpy as np

# 19
# 720
# (15, 720, 30)
# (15, 720, 19, 2)


# (15, 720, 19, 30)

data = np.zeros((15,720,30))
x = tf.constant(data)

indices = np.zeros((15,720,19,2))
y = tf.constant(indices)

result = tf.gather_nd(x, indices)
# result = tf.gather_nd(x, y)
with tf.Session() as sess:
    sess.run(result)
    print(result.eval())