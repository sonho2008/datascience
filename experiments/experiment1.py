# import packages
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

#%matplotlib inline
print('OK')


sess = tf.InteractiveSession()
df = pd.read_csv('./data/house-prices-adv-regressions/train.csv', index_col=0)
df_train = df

df['logSalePrice'] = np.log(df['SalePrice'])
xvar = 'TotalBsmtSF'
yvar = 'logSalePrice'

m = len(df)

df[xvar] = df[xvar].fillna(value = df[xvar].median())
df[yvar] = df[yvar].fillna(value = df[yvar].median())

### Rescale ###
mu = df[xvar].mean()
stdev = df[xvar].std()
df[xvar] = (df[xvar] - mu + 12)/ stdev

x_values = df[xvar].values.reshape((m,1)).astype(np.float32)
y_values = df[yvar].values.reshape((m,1)).astype(np.float32)


def weight_variable(shape, name=None):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            activations = act(preactivate, name='activation')
    return activations


x = tf.placeholder(tf.float32, [None, 1])

# Labels in y_
y_ = tf.placeholder(tf.float32, [None, 1])


W1 = weight_variable([1,1])
W2 = weight_variable([1,1])
b1 = bias_variable([1])
b2 = bias_variable([1])

y1 = tf.matmul(x, W1) + b1
y2 = tf.matmul(x, W2) + b2

loss2 = tf.reduce_mean(tf.square(y1 - y_))
loss1 = tf.reduce_mean(tf.abs(y2 - y_))

###

learning_rate = 0.4
max_steps = 20000

train_step1 =  tf.train.GradientDescentOptimizer(learning_rate).minimize(loss1)
train_step2 =  tf.train.GradientDescentOptimizer(learning_rate).minimize(loss2)

init = tf.global_variables_initializer()
sess.run(init)
##

for i in range(3000):
    l1, l2 = sess.run([loss1,loss2],
             feed_dict = {x: x_values, y_: y_values})
    if i%500==1:
        print('L1 loss:',l1, 'L2 loss:', l2)
    ts1, ts2 = sess.run([train_step1, train_step2],
             feed_dict = {x: x_values, y_: y_values})
    
print(W1.eval(), W2.eval(), b1.eval(), b2.eval())

##
def linear(l):
    return (lambda x: (l[0] + l[1]*x))

def scatter_with_lines(x, y, lines = []):
    
    plt.scatter(x,y)
    xpts = [min(x), max(x)]
    for i,l in enumerate(lines):
        plt.plot(xpts,list(map(linear(l), xpts)), label=str(i))
    plt.legend()        
    plt.show()
    plt.close()
    return
        
##    
l1 = (b1.eval()[0], W1.eval()[0])
l2 = (b2.eval()[0], W2.eval()[0])
scatter_with_lines(list(x_values),list(y_values),[l1, l2])
