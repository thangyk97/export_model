# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7
"""Train and export a simple Softmax Regression TensorFlow model.
The model is from the TensorFlow "MNIST For ML Beginner" tutorial. This program
simply follows all its training instructions, and uses TensorFlow SavedModel to
export the trained model with proper signatures that can be loaded by standard
tensorflow_model_server.
Usage: mnist_export.py [--training_iteration=x] [--model_version=y] export_dir
"""

import os
import sys
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat
from tensorflow_serving.example import mnist_input_data


#training flags
tf.app.flags.DEFINE_integer('training_iteration', 1000,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
FLAGS = tf.app.flags.FLAGS


def main(_):
  if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
    print('Usage: mnist_export.py [--training_iteration=x] '
          '[--model_version=y] export_dir')
    sys.exit(-1)
  if FLAGS.training_iteration <= 0:
    print ('Please specify a positive value for training iteration.')
    sys.exit(-1)
  if FLAGS.model_version <= 0:
    print ('Please specify a positive value for version number.')
    sys.exit(-1)

  # Train model
  print ('Training model...')
  
  #Read the data and format it
  mnist = mnist_input_data.read_data_sets(FLAGS.work_dir, one_hot=True)
  sess = tf.InteractiveSession()
  serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
  feature_configs = {'x': tf.FixedLenFeature(shape=[784], dtype=tf.float32),}
  tf_example = tf.parse_example(serialized_tf_example, feature_configs)
  
  
  #Build model
  x = tf.identity(tf_example['x'], name='x')  # use tf.identity() to assign name
  y_ = tf.placeholder('float', shape=[None, 10])
  w = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  sess.run(tf.global_variables_initializer())
  y = tf.nn.softmax(tf.matmul(x, w) + b, name='y')

  cross_entropy = -tf.reduce_sum(y_ * tf.log(y)) # Loss function

  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) # Train step
  values, indices = tf.nn.top_k(y, 10)
  table = tf.contrib.lookup.index_to_string_table_from_tensor(
      tf.constant([str(i) for i in range(10)]))
  

  #train the model
  prediction_classes = table.lookup(tf.to_int64(indices))

  for _ in range(FLAGS.training_iteration):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

  print ('training accuracy %g' % sess.run(
      accuracy, feed_dict={x: mnist.test.images,
                           y_: mnist.test.labels}))
  print ('Done training!')

  # Save the model
  
  #where to save to?
  export_path_base = sys.argv[-1]
  export_path = os.path.join(
      compat.as_bytes(export_path_base),
      compat.as_bytes(str(FLAGS.model_version)))
  print ('Exporting trained model to', export_path)
  
  #This creates a SERVABLE from our model
  #saves a "snapshot" of the trained model to reliable storage 
  #so that it can be loaded later for inference.
  #can save as many version as necessary
  
  #the tensoroflow serving main file tensorflow_model_server
  #will create a SOURCE out of it, the source
  #can house state that is shared across multiple servables 
  #or versions
  
  #we can later create a LOADER from it using tf.saved_model.loader.load
  
  #then the MANAGER decides how to handle its lifecycle
  
  builder = saved_model_builder.SavedModelBuilder(export_path)

  # Build the signature_def_map.
  #Signature specifies what type of model is being exported, 
  #and the input/output tensors to bind to when running inference.
  #think of them as annotiations on the graph for serving
  #we can use them a number of ways
  #grabbing whatever inputs/outputs/models we want either on server
  #or via client
  classification_inputs = utils.build_tensor_info(serialized_tf_example)
  classification_outputs_classes = utils.build_tensor_info(prediction_classes)
  classification_outputs_scores = utils.build_tensor_info(values)

   
  classification_signature = signature_def_utils.build_signature_def(

      inputs={signature_constants.CLASSIFY_INPUTS: classification_inputs},

      outputs={
          signature_constants.CLASSIFY_OUTPUT_CLASSES: classification_outputs_classes,
          signature_constants.CLASSIFY_OUTPUT_SCORES: classification_outputs_scores
      },
      method_name=signature_constants.CLASSIFY_METHOD_NAME)

  tensor_info_x = utils.build_tensor_info(x)
  tensor_info_y = utils.build_tensor_info(y)

  prediction_signature = signature_def_utils.build_signature_def(
      inputs={'images': tensor_info_x},
      outputs={'scores': tensor_info_y},
      method_name=signature_constants.PREDICT_METHOD_NAME)

  legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
  
  #add the sigs to the servable
  builder.add_meta_graph_and_variables(
      sess,
      [tag_constants.SERVING],
      signature_def_map={
          'predict_images': prediction_signature,
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: classification_signature,
      },
      legacy_init_op=legacy_init_op)

  #save it!
  builder.save()

  print ('Done exporting!')


if __name__ == '__main__':
  tf.app.run()