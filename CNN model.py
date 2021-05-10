#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
train_data_path = 'train.tfrecords' # address to save the hdf5 file
test_data_path = 'test.tfrecords'


# In[2]:


def weight_variable(shape):
    with tf.variable_scope("weight",reuse=tf.AUTO_REUSE):
        return    tf.get_variable("weight",shape=shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
def weight_variable1(shape):
    with tf.variable_scope("weight1",reuse=tf.AUTO_REUSE):
        return    tf.get_variable("weight1",shape=shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
def weight_variable2(shape):
    with tf.variable_scope("weight2",reuse=tf.AUTO_REUSE):
        return    tf.get_variable("weight2",shape=shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
def weight_variable3(shape):
    with tf.variable_scope("weight3",reuse=tf.AUTO_REUSE):
        return    tf.get_variable("weight3",shape=shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
def weight_variable4(shape):
    with tf.variable_scope("weight4",reuse=tf.AUTO_REUSE):
        return    tf.get_variable("weight4",shape=shape,initializer=tf.truncated_normal_initializer(stddev=0.1))

def bias_variable(shape):
    with tf.variable_scope("bias",reuse=tf.AUTO_REUSE):
        return    tf.get_variable("bias",shape=shape,initializer=tf.constant_initializer(0.1))
def bias_variable1(shape):
    with tf.variable_scope("bias1",reuse=tf.AUTO_REUSE):
        return    tf.get_variable("bias1",shape=shape,initializer=tf.constant_initializer(0.1))

def bias_variable2(shape):
    with tf.variable_scope("bias2",reuse=tf.AUTO_REUSE):
        return    tf.get_variable("bias2",shape=shape,initializer=tf.constant_initializer(0.1))

def bias_variable3(shape):
    with tf.variable_scope("bias3",reuse=tf.AUTO_REUSE):
        return   tf.get_variable("bias3",shape=shape,initializer=tf.constant_initializer(0.1))

def bias_variable4(shape):
    with tf.variable_scope("bias4",reuse=tf.AUTO_REUSE):
        return tf.get_variable("bias4",shape=shape,initializer=tf.constant_initializer(0.1))


# In[3]:


def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 4, 4, 1], padding='SAME')
def max_pool_2x2(x):
        x = tf.nn.dropout(x, 0.85)
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


# In[4]:


def my_cnn(x,num_classes, is_training):
        # Conv Layer 1: with 16 filters of size 20 x 20
        W_conv1 = weight_variable([20, 20, 3, 16])
        b_conv1 = bias_variable([16])
        x_image = tf.reshape(x, [-1, 750, 750, 3])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        # Pool
        h_pool1 = max_pool_2x2(h_conv1)
        # Conv Layer 2: with 32 filters of size 15 x 15
        W_conv2 = weight_variable1([15, 15, 16, 32])
        b_conv2 = bias_variable1([32])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        # Pool
        h_pool2 = max_pool_2x2(h_conv2)
        # Conv Layer 3: with 64 filter of size 6 x 6
        W_conv3 = weight_variable2([6, 6, 32, 64])
        b_conv3 = bias_variable2([64])

        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)
        
        W_fc1 = weight_variable3([2*2*64, 64])
        b_fc1 = bias_variable3([64])
        # flatening output of pool layer to feed in FC layer
        h_pool3_flat = tf.reshape(h_pool3, [-1, 2*2*64])
        # FC layer
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
        # Dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)
        W_fc2 = weight_variable4([64, 2])
        b_fc2 = bias_variable4([2])
        # Output
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        return y_conv


# In[7]:


feature = {'train/image': tf.FixedLenFeature([], tf.string),'train/label': tf.FixedLenFeature([], tf.int64)}
feature_test = {'test/image': tf.FixedLenFeature([], tf.string),'test/label': tf.FixedLenFeature([], tf.int64)}
# Create a list of filenames and pass it to a queue
filename_queue = tf.train.string_input_producer([train_data_path])
filename_queue_test = tf.train.string_input_producer([test_data_path])
# Define a reader and read the next record
reader = tf.TFRecordReader()
reader_test = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
_, serialized_example_test = reader_test.read(filename_queue_test)
# Decode the record read by the reader
features = tf.parse_single_example(serialized_example, features=feature)
features_test = tf.parse_single_example(serialized_example_test, features=feature_test)
 # Convert the image data from string back to the numbers
image = tf.decode_raw(features['train/image'], tf.float32)
image_test = tf.decode_raw(features_test['test/image'], tf.float32)
 # Cast label data into int32
label = tf.cast(features['train/label'], tf.int32)
label_test = tf.cast(features_test['test/label'], tf.int32)
 # Reshape image data into the original shape
image = tf.reshape(image, [750, 750, 3])
image_test = tf.reshape(image_test, [750,750,3])
label = tf.reshape(label,[1])
label_test = tf.reshape(label_test, [1])


# In[13]:


num_of_training_records = 0
print(train_data_path)
for record in tf.python_io.tf_record_iterator(train_data_path):
    num_of_training_records +=1
print("Num of Training records")
print(num_of_training_records)
num_of_test_records = 0
for record in tf.python_io.tf_record_iterator(test_data_path):
    num_of_test_records +=1
print("num_of_test_records")    
print(num_of_test_records)
batch_size = 1
# Any preprocessing here ...

# Creates batches by randomly shuffling tensors
images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size,capacity=100, num_threads=1, min_after_dequeue=80)
images_test, labels_test = tf.train.shuffle_batch([image_test, label_test],batch_size=batch_size, capacity=100, num_threads=1, min_after_dequeue=80)

num_classes = 2
lbl = tf.one_hot(labels,num_classes)
lbl_test = tf.one_hot(labels_test,num_classes)
lbl = tf.squeeze(lbl)
lbl_test = tf.squeeze(lbl_test)

global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(1e-4, global_step=global_step,decay_steps=10000, decay_rate=0.97)
function_to_map = lambda x:my_cnn(x,num_classes,is_training=True)
logits = tf.map_fn(function_to_map,images)
logits = tf.squeeze(logits)
function_to_map_test = lambda x:my_cnn(x,num_classes,is_training=True)
logits_test = tf.map_fn(function_to_map_test,images_test)
logits_test = tf.squeeze(logits_test)
y_pred_test = tf.argmax(logits_test,1)
y_true_test = tf.argmax(lbl_test,1)
print(labels,lbl,logits)
aa = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=lbl)
print ('~~~ ' , aa)
#cross_entropy =tf.reduce_mean(aa)
cross_entropy =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=lbl))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(lbl, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
correct_prediction_test = tf.equal(tf.argmax(logits_test, 1), tf.argmax(lbl_test, 1))
accuracy_test = tf.reduce_mean(tf.cast(correct_prediction_test, tf.float32))
saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord,sess=sess)
    num_epoch = 1
    f1score = []
    y_true1 = []
    y_pred1 = []
    training_error = []
    test_accuracy = []
    for i in range(num_epoch):
            for j in range(0,num_of_training_records,batch_size):
                 if j%384 == 0:
                    train_accuracy = sess.run(accuracy)
                    training_error.append(1 - train_accuracy)
                    print("step: %d, training accuracy: %g" % (j,train_accuracy))
                    ts = sess.run(train_step)


# In[ ]:




