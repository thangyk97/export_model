import tensorflow as tf


a = tf.convert_to_tensor([[1, 0, 0, 0, 0], [0, 0, 0, 1, 0]])
b = tf.nn.top_k(a, 5)
sess = tf.Session()
print(sess.run(b).indices)
