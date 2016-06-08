import tensorflow as tf
import numpy as np
import pandas as pd


tf.set_random_seed(0)

print "-------- PROCESSING DATA  ---------"

df = pd.read_csv('train.csv')

def transform(dd):
    dd['1_xy_buckets16'] = ((dd.x * 16).astype(np.int32) + 1000*((dd.y*16).astype(np.int32))).astype(np.int32)
    dd['2_xy_buckets25'] = ((dd.x * 28).astype(np.int32) + 1000*((dd.y*28).astype(np.int32))).astype(np.int32)    
    dd['3_xy_buckets40'] = ((dd.x * 40).astype(np.int32) + 10000*((dd.y*40).astype(np.int32))).astype(np.int32)
    
    dd['1_time_mod_1h'] = dd.time.mod(60).astype(np.int)
    dd['1_time_mod_4h'] = dd.time.mod(240).divide(5).astype(np.int)
    dd['1_time_mod_24h'] = dd.time.mod(1440).divide(20).astype(np.int)
    dd['1_time_mod_7d'] = dd.time.mod(10080).divide(120).astype(np.int)
    dd['1_time_mod_4w'] = dd.time.mod(10080*4).divide(480).astype(np.int)    
    
    dd['2_time_mod_1h'] = dd.time.mod(60).astype(np.int)
    dd['2_time_mod_4h'] = dd.time.mod(240).astype(np.int)
    dd['2_time_mod_24h'] = dd.time.mod(1440).divide(2).astype(np.int)
    dd['2_time_mod_7d'] = dd.time.mod(10080).divide(12).astype(np.int)
    dd['2_time_mod_4w'] = dd.time.mod(10080*4).divide(48).astype(np.int)
    
    dd['3_time_mod_1h'] = dd.time.mod(60).astype(np.int)
    dd['3_time_mod_4h'] = dd.time.mod(240).astype(np.int)
    dd['3_time_mod_24h'] = dd.time.mod(1440).astype(np.int)
    dd['3_time_mod_7d'] = dd.time.mod(10080).astype(np.int)
    dd['3_time_mod_4w'] = dd.time.mod(10080*4).divide(4).astype(np.int)
    
    dd.accuracy = dd.accuracy - 1
    
    dd['1_accuracy'] = dd.accuracy
    dd['2_accuracy'] = dd.accuracy
    dd['3_accuracy'] = dd.accuracy
    del dd['time']
    del dd['row_id']

transform(df)

place_ids = sorted(df.place_id.unique())
place_id_2_index = dict(zip(place_ids, range(len(place_ids))))
index2place_id = dict(zip(range(len(place_ids)), place_ids))
df.place_id = df.place_id.map(place_id_2_index.get)

xy_ids = sorted(df['1_xy_buckets16'].unique())
xy_id_2_index16 = dict(zip(xy_ids, range(len(xy_ids))))
df['1_xy_buckets16'] = df['1_xy_buckets16'].map(xy_id_2_index16.get)

xy_ids = sorted(df['2_xy_buckets25'].unique())
xy_id_2_index25 = dict(zip(xy_ids, range(len(xy_ids))))
df['2_xy_buckets25'] = df['2_xy_buckets25'].map(xy_id_2_index25.get)

xy_ids = sorted(df['3_xy_buckets40'].unique())
xy_id_2_index40 = dict(zip(xy_ids, range(len(xy_ids))))
df['3_xy_buckets40'] = df['3_xy_buckets40'].map(xy_id_2_index40.get)

categoricalColumnsLeft_1 = [
 '1_xy_buckets16',
 '1_time_mod_1h',
 '1_time_mod_4h',
 '1_time_mod_24h',
 '1_time_mod_7d',
 '1_time_mod_4w',
 '1_accuracy'
]

categoricalColumnsLeft_2 = [
 '2_xy_buckets25',
 '2_time_mod_1h',
 '2_time_mod_4h',
 '2_time_mod_24h',
 '2_time_mod_7d',
 '2_time_mod_4w',
 '2_accuracy'
]

categoricalColumnsLeft_3 = [
 '3_xy_buckets40',
 '3_time_mod_1h',
 '3_time_mod_4h',
 '3_time_mod_24h',
 '3_time_mod_7d',
 '3_time_mod_4w',
 '3_accuracy'
]

ccLeft = categoricalColumnsLeft_1 + categoricalColumnsLeft_2 + categoricalColumnsLeft_3

categoricalColumnsRight = ['place_id']

numericalColumns = [
]

categoricalColumnVocabSizes = {}

print "-------- CREATING TENSOR GRAPH  ---------"

for ccol in categoricalColumnsLeft_1 + categoricalColumnsLeft_2 + categoricalColumnsLeft_3 + categoricalColumnsRight:
    categoricalColumnVocabSizes[ccol] = df[ccol].max() + 1

batch_size = 2048

embSize = {}

print "Categorical Vocab sizes:", categoricalColumnVocabSizes

def calcEmbSize(vocabSize, left):
    if left:
        if vocabSize> 10000:
            return int(2*np.sqrt(vocabSize))
        else:
            return int((3.) * np.sqrt(vocabSize))
    else:
        return int((5)*np.sqrt(vocabSize))


ccW = {}

for ccol in ccLeft + categoricalColumnsRight:
    isLeft = ccol in ccLeft
    vocabSize = categoricalColumnVocabSizes[ccol]
    embSize[ccol] = calcEmbSize(vocabSize, isLeft)
    ccW[ccol] = tf.Variable(
        tf.random_uniform([vocabSize, embSize[ccol]], -1.0, 1.0))

_y = tf.placeholder("float32", [batch_size, 1])

inputsLeft = {}
inputsRight = {}

for ccol in ccLeft:
    inputsLeft[ccol] = tf.placeholder("int32", [batch_size, 1])
    
for ccol in categoricalColumnsRight:
    inputsRight[ccol] = tf.placeholder("int32", [batch_size, 1])

for ncol in numericalColumns:
    inputsLeft[ncol] = tf.placeholder("float32", [batch_size, 1])

embeddings = {}

for ccol in ccLeft:
    embeddings[ccol] = tf.reshape(tf.nn.embedding_lookup(tf.nn.l2_normalize(ccW[ccol], 1), inputsLeft[ccol]), [batch_size, embSize[ccol]])

for ccol in categoricalColumnsRight:
    embeddings[ccol] = tf.reshape(tf.nn.embedding_lookup(tf.nn.l2_normalize(ccW[ccol], 1), inputsRight[ccol]), [batch_size, embSize[ccol]])

# just concatenate all left, then slap another layer

leftSide_1 = tf.concat(1, [inputsLeft[ncol] for ncol in numericalColumns] + [embeddings[ccol] for ccol in categoricalColumnsLeft_1])

leftSideShape_1 = (batch_size, len(numericalColumns) + sum([embSize[ccol] for ccol in categoricalColumnsLeft_1]))

leftSide_2 = tf.concat(1, [inputsLeft[ncol] for ncol in numericalColumns] + [embeddings[ccol] for ccol in categoricalColumnsLeft_2])

leftSideShape_2 = (batch_size, len(numericalColumns) + sum([embSize[ccol] for ccol in categoricalColumnsLeft_2]))

leftSide_3 = tf.concat(1, [inputsLeft[ncol] for ncol in numericalColumns] + [embeddings[ccol] for ccol in categoricalColumnsLeft_3])

leftSideShape_3 = (batch_size, len(numericalColumns) + sum([embSize[ccol] for ccol in categoricalColumnsLeft_3]))


rightSide = tf.concat(1, [embeddings[ccol] for ccol in categoricalColumnsRight])

rightSideShape = (batch_size, sum([embSize[ccol] for ccol in categoricalColumnsRight]))


print "Emb sizes:", embSize

print "Left side shapes:", leftSideShape_1, leftSideShape_2, leftSideShape_3
print "Right side shape:", rightSideShape

_remove_prob = tf.placeholder(tf.float32, shape=[])

leftSide_1 = tf.nn.dropout(leftSide_1, keep_prob=1. - _remove_prob/10.)
leftSide_2 = tf.nn.dropout(leftSide_2, keep_prob=1. - _remove_prob/10.)
leftSide_3 = tf.nn.dropout(leftSide_3, keep_prob=1. - _remove_prob/10.)
rightSide = tf.nn.dropout(rightSide, keep_prob=1. - _remove_prob/10.)

epsilon = 1e-3

PROJ_W_1 = tf.get_variable("PW1", shape=[leftSideShape_1[1], rightSideShape[1]], initializer=tf.contrib.layers.xavier_initializer())
PROJ_Z_1 = tf.matmul(leftSide_1, PROJ_W_1)
batch_mean1, batch_var1 = tf.nn.moments(PROJ_Z_1,[0])
scale1 = tf.Variable(tf.ones([rightSideShape[1]]))
PROJ_B_1 = tf.Variable(tf.zeros([rightSideShape[1]]))
BN1 = tf.nn.batch_normalization(PROJ_Z_1,batch_mean1,batch_var1,PROJ_B_1,scale1,epsilon)
LEFT_1 = tf.nn.relu6(BN1) - 3.

PROJ_W_2 = tf.get_variable("PW2", shape=[leftSideShape_2[1], rightSideShape[1]], initializer=tf.contrib.layers.xavier_initializer())
PROJ_Z_2 = tf.matmul(leftSide_2, PROJ_W_2)
batch_mean2, batch_var2 = tf.nn.moments(PROJ_Z_2,[0])
scale2 = tf.Variable(tf.ones([rightSideShape[1]]))
PROJ_B_2 = tf.Variable(tf.zeros([rightSideShape[1]]))
BN2 = tf.nn.batch_normalization(PROJ_Z_2,batch_mean2,batch_var2,PROJ_B_2,scale2,epsilon)
LEFT_2 = tf.nn.relu6(BN2) - 3.

PROJ_W_3 = tf.get_variable("PW3", shape=[leftSideShape_3[1], rightSideShape[1]], initializer=tf.contrib.layers.xavier_initializer())
PROJ_Z_3 = tf.matmul(leftSide_3, PROJ_W_3)
batch_mean3, batch_var3 = tf.nn.moments(PROJ_Z_3,[0])
scale3 = tf.Variable(tf.ones([rightSideShape[1]]))
PROJ_B_3 = tf.Variable(tf.zeros([rightSideShape[1]]))
BN3 = tf.nn.batch_normalization(PROJ_Z_3,batch_mean3,batch_var3,PROJ_B_3,scale3,epsilon)
LEFT_3 = tf.nn.relu6(BN3) - 3.

theta_2 = tf.Variable(0.)
theta_3 = tf.Variable(0.)

FINAL_LEFT_1 = tf.nn.dropout(LEFT_1, keep_prob=1. - _remove_prob)

FINAL_LEFT_2 = FINAL_LEFT_1 + tf.nn.dropout(tf.scalar_mul(theta_2,LEFT_2), keep_prob=1. - _remove_prob)

FINAL_LEFT_3 = FINAL_LEFT_2 + tf.nn.dropout(tf.scalar_mul(theta_3,LEFT_3), keep_prob=1. - _remove_prob)

RIGHT = rightSide

final_dim = rightSideShape[1]

def batchDotProduct(A, B, dim=final_dim, batch_size=batch_size):
    return tf.reshape(tf.batch_matmul(tf.reshape(A, [batch_size, 1, dim]), tf.reshape(B, [batch_size, dim, 1])), [batch_size, 1])

def logloss(y, A, B):
    return -tf.mul(y, tf.log(1./(1.+tf.exp(-batchDotProduct(A, B))))) - tf.mul((1. - y), tf.log(1. - 1./(1.+tf.exp(-batchDotProduct(A, B)))))

_lr = tf.placeholder(tf.float32, shape=[])

alpha = tf.placeholder(tf.float32, shape=[3])
alpha_sm = tf.reshape(tf.nn.softmax(tf.reshape(alpha, shape=[1,3])), shape=[3])

cost = alpha_sm[0]*tf.reduce_mean(logloss(_y, RIGHT, FINAL_LEFT_1)) + alpha_sm[1]*tf.reduce_mean(logloss(_y, RIGHT, FINAL_LEFT_2)) + alpha_sm[2]*tf.reduce_mean(logloss(_y, RIGHT, FINAL_LEFT_3))
optimizer = tf.train.AdamOptimizer(learning_rate=_lr).minimize(cost)

dot = tf.matmul(FINAL_LEFT_3, tf.transpose(tf.nn.l2_normalize(ccW['place_id'], 1)))
probs = 1./(1.+tf.exp(-dot))
top_probs, top_indices = tf.nn.top_k(probs, k=8, sorted=True)

_, top_indices256 = tf.nn.top_k(probs, k=256, sorted=True)
_, top_indices128 = tf.nn.top_k(probs, k=128, sorted=True)
_, top_indices64 = tf.nn.top_k(probs, k=64, sorted=True)
_, top_indices32 = tf.nn.top_k(probs, k=32, sorted=True)


training_epochs = 3
negativeSamples = 15
num_train_final = int(df.shape[0]*0.95)

df_train = df.iloc[0:num_train_final]
df_train = df_train.iloc[np.random.permutation(len(df_train))]

df_test = df.iloc[num_train_final:]
df_test = df_test.iloc[np.random.permutation(len(df_test))]

test_batch = df_test[0:batch_size]

#df_test = df.iloc[num_train_final:]

def apk(actual, predicted, k=3):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=3):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

print "---------- START TRAINING ----------"

init = tf.initialize_all_variables()
saver = tf.train.Saver()

REMOVE_PROB = 0.2

# lr = 0.005
lr = 0.02

display_step = 50
save_step = 10000
last_losses = [1. for i in range(20)]
last_rate_switch = 0
switch_max_every = 2000
lr_decay = 0.8
phase = 1
decaysInPhase = 0
first_phase_counter = 100
second_phase_counter = -100
third_phase_counter = -100
additional_phase_decay = 0.9995
maxDecaysPerPhase = 6
with tf.Session() as sess:
    sess.run(init)    
    for epoch in range(training_epochs):
        avg_cost = 0.
        positive_in_batch = batch_size/(1+negativeSamples)
        total_batch = int(df_train.shape[0]/(positive_in_batch))
        for k in range(total_batch):                        
            if phase == 1:
                first_phase_counter += 1
            elif phase == 2:
                second_phase_counter += 1
                first_phase_counter *= additional_phase_decay
            elif phase == 3:
                third_phase_counter += 1
                second_phase_counter *= additional_phase_decay
                first_phase_counter *= additional_phase_decay            
            positive = df_train.iloc[k*positive_in_batch:(k+1)*positive_in_batch]
            if positive.shape[0] != positive_in_batch:
                continue
            feed_d = {}
            for ccol in ccLeft:
                feed_d[inputsLeft[ccol]] = np.vstack([
                    np.reshape(positive[ccol].as_matrix(), (positive_in_batch, 1)) for i in range(negativeSamples+1)])
            for ccol in numericalColumns:
                feed_d[inputsLeft[ccol]] = np.vstack([
                    np.reshape(positive[ccol].as_matrix(), (positive_in_batch, 1)) for i in range(negativeSamples+1)])
            feed_d[_y] = np.vstack([np.ones((positive_in_batch, 1))] + [np.zeros((positive_in_batch*negativeSamples, 1))])
            feed_d[_lr] = lr
            feed_d[_remove_prob] = REMOVE_PROB
            
            feed_d[alpha] = np.array([first_phase_counter, second_phase_counter, third_phase_counter])
            
            feed_train = dict(feed_d)
            current_top_choice = top_indices32
            if (decaysInPhase<2):
                current_top_choice = top_indices256
            elif (decaysInPhase<4):
                current_top_choice = top_indices128
            elif (decaysInPhase<6):
                current_top_choice = top_indices64
            elif (decaysInPhase==6):
                current_top_choice = top_indices32
            top_ind = sess.run(current_top_choice, feed_dict=feed_train)
            top_ind = top_ind[:positive_in_batch]
            def mkright():
                for index, row in zip(range(len(top_ind)), top_ind):
#                    print index, len(top_ind), len(row)
#                    print row
#                    print positive['place_id'].iloc[index % positive_in_batch]
                    yield np.random.choice(np.setdiff1d(row, [positive['place_id'].iloc[index]]), negativeSamples/2)
            correct_negative_samples = np.array(list(mkright()))
            for ccol in categoricalColumnsRight:
                feed_d[inputsRight[ccol]] = np.vstack([
                    np.reshape(positive[ccol].as_matrix(), (positive_in_batch, 1))
                ] + [
                    np.reshape(np.random.choice(df[ccol], size=positive_in_batch),
                               (positive_in_batch, 1)) for i in range(negativeSamples/2 + 1)
                ] + np.array(list(mkright())).T.reshape(((negativeSamples/2),positive_in_batch,1)).tolist()
                ) #                           CHANGE TO SAMPLE 7 out of 128 or so.
            feed_train = dict(feed_d)
            feed_test = dict(feed_d)
            feed_test[_remove_prob] = 0.

            sess.run(optimizer, feed_dict=feed_train)
            if (k % display_step == 0):
                
                inp = test_batch
                feed_eval = dict(feed_d)
                for ccol in ccLeft:
                    feed_eval[inputsLeft[ccol]] = np.reshape(inp[ccol].as_matrix(), (batch_size, 1))
                for ccol in numericalColumns:
                    feed_eval[inputsLeft[ccol]] = np.reshape(inp[ccol].as_matrix(), (batch_size, 1))
                feed_eval[_remove_prob] = 0.
                
                batch_logloss = sess.run(cost, feed_dict=feed_test)
                last_losses.append(batch_logloss)
                last10 = np.array(last_losses[-10:]).mean()
                last20 = np.array(last_losses[-20:]).mean()
                if (last20 <= last10) and (last_rate_switch + switch_max_every < k):
                    last_rate_switch = k
                    lr = lr*lr_decay
                    print "[DECAYING LEARNING RATE] new rate:", lr, "last20:", last20, "last10", last10
                    decaysInPhase += 1
                    print "Decays in phase:", decaysInPhase
                    if decaysInPhase > maxDecaysPerPhase:
                        if phase == 3:
                            continue
                        else:
                            decaysInPhase = 0
                            phase += 1
                            print "Entering Phase:", phase
                print "iter:", k, "out of:", total_batch, "phase:", phase, "decays:", decaysInPhase, "lr:", lr, "logloss:", batch_logloss, "last20:", last20, "last10:", last10, "cost phases:", sess.run(alpha_sm, feed_dict=feed_test), "embedding theta valves:", sess.run((theta_2, theta_3), feed_dict=feed_test)
                if (k>0 and (k % save_step == 0)):
                    save_path = saver.save(sess, "./model_" + str(k+1) + "_epoch" + str(epoch) + "_loss_" + str(batch_logloss) + ".ckpt")
                    print "Model saved in file: %s" % save_path
                indices = sess.run(top_indices, feed_dict=feed_eval)
                r = pd.DataFrame(indices)
                r.columns = ['1', '2', '3', '4', '5', '6', '7', '8']
                r['0'] = test_batch.reset_index()['place_id']
                r['correct8'] = (r['1'] == r['0']) | (r['2'] == r['0']) | (r['3'] == r['0']) | (r['4'] == r['0']) | \
                 (r['5'] == r['0']) | (r['6'] == r['0']) | (r['7'] == r['0']) | (r['8'] == r['0'])                 
                r['correct3'] = (r['1'] == r['0']) | (r['2'] == r['0']) | (r['3'] == r['0'])
                r['correct1'] = (r['1'] == r['0'])
                print "c1:", r[r['correct1']==True].shape[0], "c3:", r[r['correct3']==True].shape[0], "c8:", r[r['correct8']==True].shape[0], "\t", "MAP@3", mapk(np.reshape(np.array(test_batch['place_id']), (test_batch.shape[0],1)), indices.tolist())
        print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
        save_path = saver.save(sess, "./model_epoch" + str(epoch) + ".ckpt")
        print "Model saved in file: %s" % save_path
    print "Optimization Finished!"

exit(-1)

tt = pd.read_csv('test.csv')

transform(tt)

tt['xy_buckets30'] = tt['xy_buckets30'].map(xy_id_2_index30.get)
tt['xy_buckets10'] = tt['xy_buckets10'].map(xy_id_2_index10.get)

_, top_indices_test = tf.nn.top_k(probs, k=3, sorted=True)

all_top_indices = []

with tf.Session() as sess:
    saver.restore(sess, "model_60001_epoch0_loss_0.0803847.ckpt")
    print("Model restored.")
    total_batch = int(tt.shape[0]/batch_size)
    for k in range(total_batch+1):
        inp = tt.iloc[k*batch_size:(k+1)*batch_size]
        if inp.shape[0] != batch_size:
            inp = np.vstack([inp for i in range(batch_size/inp.shape[0] + 1)])[:batch_size]
            inp = pd.DataFrame(inp)
            inp.columns = tt.columns
        feed_d = {}
        for ccol in ccLeft:
            feed_d[inputsLeft[ccol]] = np.reshape(inp[ccol].as_matrix(), (batch_size, 1))
        for ccol in numericalColumns:
            feed_d[inputsLeft[ccol]] = np.reshape(inp[ccol].as_matrix(), (batch_size, 1))
        feed_test = dict(feed_d)
        feed_test[_remove_prob] = 0.
        print k, "out of", total_batch
        all_top_indices.append(sess.run(top_indices_test, feed_dict=feed_test))


result = np.vstack(all_top_indices)
result = result[:tt.shape[0]]

r = pd.DataFrame(result)
r.columns = ['1', '2', '3']
r['1'] = r['1'].map(index2place_id)
r['2'] = r['2'].map(index2place_id)
r['3'] = r['3'].map(index2place_id)
r['row_id'] = r.index
r['place_id'] = r['1'].astype(str) + ' ' + r['2'].astype(str) + ' ' + r['3'].astype(str)

r[['row_id', 'place_id']].to_csv('submission_model_60001_epoch0_loss_0.0803847.csv', index=False)

