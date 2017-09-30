# import packages
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import sondatatools as dt

from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

#%matplotlib inline
print('OK')


sess = tf.InteractiveSession()
df = pd.read_csv('Kaggle/data/house-prices-adv-regressions/train.csv', index_col=0)
df = df.select_dtypes([np.number])

df['logSalePrice'] = np.log(df['SalePrice'])
df['log1pTotalBsmtSF'] = np.log1p(df['TotalBsmtSF'])

#xcol = 'log1pTotalBsmtSF'
xcol = 'log1pTotalBsmtSF'
ycol = 'logSalePrice'

m = len(df)

df[xcol] = df[xcol].fillna(value = df[xcol].median())
df[ycol] = df[ycol].fillna(value = df[ycol].median())

### Rescale ###
newDf, W = dt.standardize(df)
df=newDf

x_values = df[xcol].values.reshape((m,1)).astype(np.float32)
y_values = df[ycol].values.reshape((m,1)).astype(np.float32)


def weight_variable(shape, name=None):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)

x = tf.placeholder(tf.float32, [None, 1])

# Labels in y_
y_ = tf.placeholder(tf.float32, [None, 1])


### make a 2 network with 1 hidden layer
W1 = weight_variable([1,4])
b1 = bias_variable([4])
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)

W2 = weight_variable([4,1])
b2 = bias_variable([1])
y = tf.matmul(hidden1, W2) + b2

loss = tf.reduce_mean(tf.square(y - y_))

###

learning_rate = 0.2
max_steps = 5000

train_step2 =  tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)
##

for i in range(5000):
    l2 = sess.run([loss],
             feed_dict = {x: x_values, y_: y_values})
    if i%500==1:
        print('L2 loss:', l2)
    ts2 = sess.run([train_step2],
             feed_dict = {x: x_values, y_: y_values})
    

##
def linear(l):
    """ l = (l[0],l[1]). returns a function mapping x to l[0]+l[1]*x
    """
    return (lambda x: (l[0] + l[1]*x))
        
    
def scatterplot_with_pred(xs, ys, xpts, preds, color='g'):
    plt.scatter(xs,ys)
    plt.plot(xpts, preds,'k', label='Predictions', color=color)
    plt.legend()
    plt.show()
    plt.close()
    return

num = 100    
xmin = np.min(x_values)
xmax = np.max(x_values)
step = (xmax - xmin)/num
xpts = np.arange(xmin, xmax, step)
xpts = np.reshape(np.array(xpts), [num,1])
preds = sess.run(y, feed_dict={x: xpts})
scatterplot_with_pred(x_values.flatten(), y_values.flatten(), xpts.flatten(), preds.flatten())




