import pandas as pd
import os
os.environ['TF_KERAS'] = '1'
# import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
#import tensorflow.compat.v1 as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# from tensorflow.python.ops import variable_scope as vs
import time
import argparse
import csv
import math
# from lookahead import Lookahead
# import keras
# from keras_radam.training import RAdamOptimizer
# from keras_radam import RAdam
# import tensorflow_addons as tfa
tf.compat.v1.disable_eager_execution()


# -------------------------------read data--------------------------------------------
def load_data(datax_path="", datay_path=""):
    #data_s = pd.read_csv(datax_path, names=["T","G","D","H"], header=0)
    #data_c = pd.read_csv(datay_path, names=["x","y","Y"], header=0)
    data_s = pd.read_csv(datax_path,  header=None)
    data_c = pd.read_csv(datay_path,  header=None)
    return (data_s, data_c)

# -------------------------------norm--------------------------------------------
def norm(y_true, y_pred):
    return tf.norm(y_true - y_pred) / tf.norm(y_true)


# -------------------------------LeakyReLU--------------------------------------------
def LeakyReLU(x,leak=0.08,name="LeakyReLU"):
    with tf.variable_scope(name):
        f1 = 0.5*(1 + leak)
        f2 = 0.5*(1 - leak)
        return f1*x+f2*tf.abs(x)


# -------------------------------sigmoid--------------------------------------------
def sigmoid_function(z):
    fz = []
    for num in z:
        fz.append(1/(1 + math.exp(-num)))
    return fz
# ------------------------------normalization-------------------------------------------
def normalization(data_s, data_c):
    #data_x = preprocessing.scale(data_s)
    #data_x=(data_s * [1., 1., 1., 1.] - ([3., 1., 0.1, 1.])) / ([21., 9., 0.5, 9.])
    data_x=preprocessing.minmax_scale(data_s,feature_range=(0, 1), axis=0, copy=True)
    # data_x = np.array(data_s, dtype=np.float64)
    data_y = np.array(data_c, dtype = np.float64)
    return (data_x, data_y)



# -------------------------------split data-------------------------------------------
def split_data(data_x, data_y, num_test, num_validation, random_state=42):
    x_left, x_test, y_left, y_test = train_test_split(data_x, data_y, test_size=int(num_test), random_state=random_state)
    x_train, x_vali, y_train, y_vali = train_test_split(x_left, y_left, test_size=int(num_validation), random_state=random_state)
    return (x_train, y_train, x_vali, y_vali, x_test, y_test)


# -------------------------------initialize w,b-------------------------------------------
def init_wb(i, shape0, shape1):
    init_w = tf.get_variable(name = 'w'+str(i), shape = [shape0,shape1], initializer=None)
    init_b = tf.get_variable(name = 'b'+str(i), shape = [1,shape1], initializer=tf.zeros_initializer())
    return init_w, init_b


# -------------------------------dense layer-------------------------------------------
def dense(x, w, b, activation):
    if (str(activation) == 'None'):
        l = tf.add(tf.matmul(x, w), b)#x*w+b
    else:
        l = activation(tf.add(tf.matmul(x, w), b))
    return l

# --------------------------------create csv---------------------------------------------
def create_csv(path,data):
    with open(path,'w') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data)


#--------------------------------inverse design--------------------------------------------
# This realizes the inverse design. (single input)

def inverse(datac_path, node_input, n_hidden, lr_rate, lr_decay, output_folder_pre, output_folder_ma, Epoch_total, datax_path, datay_path, num_test, num_validation, n_batch, patienceLimit, datas_path, node_output, forward, inverse, pretrain, tandem):
    #load y data (data of sigma)
    # save_result_path = 'new_test_result'
    # if not os.path.exists(save_result_path):
    #     os.makedirs(save_result_path)
    # save_result_path = os.path.join('./{}/{}'.format(save_result_path, 'test_result.csv'))

    data_c = pd.read_csv(datac_path,  header=None)
    data_y = np.array(data_c, dtype = np.float64)
    # data_s, data_c = load_data(datax_path, datay_path)
    # data_x, data_y = normalization(data_s, data_c)
    # x_train, y_train, x_vali, y_vali, x_test, y_test = split_data(data_x, data_y, num_test, num_validation,random_state=42)

    # Set x,y
    x_size_column = node_input
    y_size_column = data_y.shape[1]

    # init_x = tf.constant(np.random.rand(1,x_size_column),dtype=tf.float32)
    # init_x = tf.truncated_normal_initializer(mean = 0.0, stddev = 1.0, dtype = tf.float32)
    init_x = tf.random_uniform_initializer(minval = -5, maxval = 5, dtype = tf.float32)
    x0 = tf.get_variable(name="s", shape = [1,x_size_column], initializer = init_x)
    # x0 = tf.get_variable(name="s", initializer = init_x)

    y = tf.placeholder("float", shape=[None, y_size_column])

    # Run Session
    with tf.Session() as sess:

        # load w,b
        saver = tf.train.import_meta_graph(output_folder_pre + 'wb.meta')
        saver.restore(sess, output_folder_pre + "wb")
        graph = tf.get_default_graph()
        w1 = graph.get_tensor_by_name("w1:0")
        b1 = graph.get_tensor_by_name("b1:0")
        w2 = graph.get_tensor_by_name("w2:0")
        b2 = graph.get_tensor_by_name("b2:0")
        w3 = graph.get_tensor_by_name("w3:0")
        b3 = graph.get_tensor_by_name("b3:0")
        w4 = graph.get_tensor_by_name("w4:0")
        b4 = graph.get_tensor_by_name("b4:0")
        w5 = graph.get_tensor_by_name("w5:0")
        b5 = graph.get_tensor_by_name("b5:0")

        # convert w,b to constant tensor
        # w0 = tf.constant(np.ones([1,4]),tf.float32)
        # b0 = tf.constant(np.zeros([1,4]),tf.float32)
        # w0 = tf.constant(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),tf.float32)
        # b0 = tf.constant(np.array([0,0,0,0]),tf.float32)
        w1 = tf.constant(sess.run(w1))
        b1 = tf.constant(sess.run(b1))
        w2 = tf.constant(sess.run(w2))
        b2 = tf.constant(sess.run(b2))
        w3 = tf.constant(sess.run(w3))
        b3 = tf.constant(sess.run(b3))
        w4 = tf.constant(sess.run(w4))
        b4 = tf.constant(sess.run(b4))
        w5 = tf.constant(sess.run(w5))
        b5 = tf.constant(sess.run(b5))

        # Forward propagation
        # l0 = dense(x, w0, b0, tf.sigmoid)
        x=tf.sigmoid(x0)
        l1 = dense(x, w1, b1, tf.nn.relu)  # hidden layer1
        l2 = dense(l1, w2, b2, tf.nn.relu)  # hidden layer2
        l3 = dense(l2, w3, b3, tf.nn.relu)  # hidden layer3
        l4 = dense(l3, w4, b4, tf.nn.relu)  # hidden layer4
        output = dense(l4, w5, b5, None)  # output layer
        # l1 = dense(x, w1, b1,  LeakyReLU)  # hidden layer1
        # l2 = dense(l1, w2, b2, LeakyReLU)  # hidden layer2
        # l3 = dense(l2, w3, b3, LeakyReLU)  # hidden layer3
        # l4 = dense(l3, w4, b4, LeakyReLU)  # hidden layer4
        # output = dense(l4, w5, b5, None)  # output layer

        # Backward propagation
        global_step = tf.Variable(0, trainable=False)
        loss = tf.losses.mean_squared_error(y, output)
        # loss=tf.losses.absolute_difference(y, output)
        learning_rate = tf.train.exponential_decay(lr_rate, global_step, 1000, lr_decay, staircase=False)
        # optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        train_optimize = optimizer.minimize(loss,var_list=[x0])


        # global_parans=tf.trainable_variables()
        # print(global_parans)
        # initialize variables
        init = tf.global_variables_initializer()
        sess.run(init)

        # inverse start
        step =0; Epoch_num =0; loss_total =0;
        inverse_loss_file = open(output_folder_ma + "/inverse_loss.txt", 'w')
        start_time=time.time()
        print("========                         Iterations started                  ========")
        while step < Epoch_total*2:
            loss_value, yhat = sess.run([loss, output], feed_dict = {y: data_y})
            sess.run(train_optimize, feed_dict={y: data_y})
            loss_total += loss_value
            step += 1
            if (step % 20 == 0 or step == 1):
                print("Step: " + str(step) + " ; Loss: " + str(loss_total))
                # inverse_loss_file.write("step:" + str(step) + ";loss:" + str(float(loss_value)) + "; x: " + str(x.eval()) + str("\n"))
                # inverse_loss_file.flush()
            loss_total = 0
            if (loss_value == 0):
                print("Minimun loss meant.")
                break
        # output structural parameters
        s = sess.run(x)
        # R = s[0,0]*22.8+2
        # I = s[0,1]*5.4+0.1
        # S= s[0,2]*0.9+0.1
        # L = s[0,3]*9.8+1
        R = s[0, 0] * 24 + 2
        I = s[0, 1] * 10 + 0.1
        S = s[0, 2] * 0.9 + 0.1
        L = s[0, 3] * 9.8 + 1
        inverse_loss_file.write("R: " + str(R) + "; I: " + str(I) + "; S: " + str(S) + "; L: " + str(L))
        inverse_loss_file.flush()
        inverse_loss_file.close()
    print('实部:%.2f，虚部：%.2f，均方根高度 :%.2f，相关长度:%.2f'%(R,I,S,L))
    print( "========Iterations completed in : " + str(time.time()-start_time) + " ========")






if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="s_c Net Training")#命令项选项与参数解析的模块
    parser.add_argument("--datax_path",type=str,default='./home/input/canshu.csv') # Path of file x data
    parser.add_argument("--datay_path",type=str,default='./home/input/sigma.csv') # Path of file y data
    parser.add_argument("--datas_path",type=str,default='./home/input/canshu_ceshi.csv') # Path of forward design file
    parser.add_argument("--datac_path",type=str,default='./home/input/sigma_ceshi.csv') # Path of inverse design file
    parser.add_argument("--node_output",type=int,default=8) # Number of output node
    parser.add_argument("--node_input",type=int,default=4 )# Number of input node
    parser.add_argument("--num_test",type=int,default=1300) # Number of test data
    parser.add_argument("--num_validation",type=int,default=3000) # Number of validation data
    parser.add_argument("--n_hidden",type=int,default=300) # Number of neurons per layer. Fully connected layers.
    parser.add_argument("--lr_rate",type=float,default=.001) # Initial learning Rate.
    parser.add_argument("--lr_decay",type=float,default=.99) # Learning rate decay. It decays by this factor every epoch.
    parser.add_argument("--output_folder_pre",type=str,default='./home/pre/')
    parser.add_argument("--output_folder_ma",type=str,default='./home/ma/')
    parser.add_argument("--Epoch_total",type=int,default=10000) # Max number of epochs to iterite, if patience condition is not met.
    parser.add_argument("--n_batch",type=int,default=20) # Batch Size
    parser.add_argument("--patienceLimit",type=int,default=100) # Patience for stopping. If validation loss has not decreased in this many steps, it will stop the training.
    parser.add_argument("--pretrain",default='False')
    parser.add_argument("--forward",default='False')
    parser.add_argument("--inverse",default='True')
    parser.add_argument("--tandem",default='False')


    args = parser.parse_args()
    dict = vars(args)
    print(dict)

    for key,value in dict.items():
        if (dict[key]=="False"):
            dict[key] = False
        elif dict[key]=="True":
            dict[key] = True
        try:
            if dict[key].is_integer():
                dict[key] = int(dict[key])
            else:
                dict[key] = float(dict[key])
        except:
            pass
    print (dict)

        
    kwargs = {  
            'datax_path':dict['datax_path'],
            'datay_path':dict['datay_path'],
            'datas_path':dict['datas_path'],
            'datac_path':dict['datac_path'],
            'node_output':dict['node_output'],
            'node_input':dict['node_input'],
            'num_test':dict['num_test'],
            'num_validation':dict['num_validation'],
            'n_hidden':int(dict['n_hidden']),
            'lr_rate':dict['lr_rate'],
            'lr_decay':dict['lr_decay'],
            'output_folder_pre':dict['output_folder_pre'],
            'output_folder_ma':dict['output_folder_ma'],
            'Epoch_total':dict['Epoch_total'],
            'n_batch':dict['n_batch'],
            'patienceLimit':dict['patienceLimit'],
            'pretrain':dict['pretrain'],
            'forward':dict['forward'],
            'inverse':dict['inverse'],
            'tandem':dict['tandem'],
            }


    if kwargs['inverse'] == True:
        inverse(**kwargs)





