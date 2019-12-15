import tensorflow as tf
import os
import numpy as np
import random
from collections import deque
#sess = tf.Session(config=config)

import sys
sys.path.append("../")
tf.reset_default_graph()
class Net():
    def __init__(self,state_size):
        self.fc_weight_1 = tf.get_variable('weight1',shape=[state_size,48],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.fc_bias_1 = tf.get_variable('bias1',shape = [48],
                                     initializer=tf.constant_initializer(0.0))
        self.fc_weight_2 = tf.get_variable('weight2',shape=[48,24],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.fc_bias_2 = tf.get_variable('bias2',shape = [24],
                                     initializer=tf.constant_initializer(0.0))
        
        self.fc_weight_3 = tf.get_variable('weight3',shape=[24,state_size],initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.fc_bias_3 = tf.get_variable('bias3',shape = [state_size],
                                     initializer=tf.constant_initializer(0.0))
        
    def forward(self,x):
        fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(x,self.fc_weight_1),
                                          self.fc_bias_1))
        fc2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc1,self.fc_weight_2),
                                          self.fc_bias_2))
        fc_out = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc2,self.fc_weight_3),
                                           self.fc_bias_3))
        
        return fc_out
        
        #tf.Variable(tf.random_normal(shape),dtype=)
        
class RL_QG_agent:
    def __init__(self,board_size):
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TicTacToe_board_size_'+str(board_size))
        
        self.board_size = board_size
        self.state_size = board_size**2
        self.step_counter = 0
        self.memory_counter = 0
        self.save_counter = 64
        #self.MEMORY_CAPACITY = 256
        self.TARGET_REPLACE_ITER = 32
        self.BATCH_SIZE = 32
        self.memory = deque(maxlen=256)    # initialize memory
        self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.01)
        self.EPOSILON = 0.1
        self.GAMMA = 0.9
        

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        
        self.x = tf.placeholder(tf.float32,shape=[None,self.state_size])
        self.x_ = tf.placeholder(tf.float32,shape=[None,self.state_size])
        self.q_target = tf.placeholder(tf.float32,shape = [None,1])# whether set shape = batch * n_actions?
        
        
        
        #use to choose action
        self.max_action = -1
        
        
        #self.enables = enables
        
        self.init_model()
    def init_model(self):
        with tf.variable_scope('eval_net'):
            self.eval_net = Net(self.state_size)
            self.q_eval = self.eval_net.forward(self.x)
            
        with tf.variable_scope('target_net'):
            self.target_net = Net(self.state_size)
            self.q_next = self.target_net.forward(self.x_)
        
        with tf.variable_scope('loss'):
            self.action_index = tf.placeholder(tf.int32,shape=[None,1])
            self.q_eval = tf.gather(self.q_eval,self.action_index,axis=1)
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train_op'):
            self.train_op = self.optimizer.minimize(self.loss)
        
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
    
        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        
        
        
        
        self.sess = tf.InteractiveSession(config=self.config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=2)
        # self.summary_writer = tf.summary.FileWriter('./summary/'+str(self.board_size)+'_summary',self.sess.graph)
        checkpoint = tf.train.get_checkpoint_state(self.model_dir)  # if you have completed training once, the model is saved in this directory
        if checkpoint and checkpoint.model_checkpoint_path:
            self.load_model()
    def store_transition(self, s, a, r, s_, mask):
        self.memory.append((s, a, r, s_,mask))
        # replace the old memory with new memory
        #index = self.memory_counter % MEMORY_CAPACITY
        
        self.memory_counter += 1

    def choose_action(self,x,enables):
        if np.random.random() > self.EPOSILON:
            #greedy
            
            
            x = np.reshape(x,[-1,self.state_size])
            
            
            out = self.eval_net.forward(self.x)
            # out = self.sess.run(out,{self.x:x})
            out = out.eval({self.x:x})
#             #print(out[0][enables])
#             print(enables)
            action = np.argmax(out[0][enables])
            action = enables[action]
            
        else:
            action = np.random.choice(enables)#choose from legal place from state
        return action  
        
    def learn(self):    
        #print('learn')
        if self.step_counter % self.save_counter==0:
            self.save_model()
        
        
        # target parameter update 到达一定步数更新目标网络
        if self.step_counter % self.TARGET_REPLACE_ITER == 0:
            self.sess.run(self.target_replace_op)
        

        
        minibatch = random.sample(self.memory, self.BATCH_SIZE)
        minibatch = np.array(minibatch)

        b_s = []
        b_a = []
        b_r = []
        b_s_ = []
        b_mask = []
        
        for i in range(self.BATCH_SIZE):
            b_s.append(minibatch[i,0])
            b_a.append(minibatch[i,1])
            b_r.append(minibatch[i,2])
            b_s_.append(minibatch[i,3])
            b_mask.append(minibatch[i,4])
        
        # tf.summary.scalar('learn_step_reward',np.array(b_r).mean())
        # summary = tf.summary.merge_all()
        b_s = np.reshape(b_s,[-1,self.state_size])
        b_s_ = np.reshape(b_s_,[-1,self.state_size])
        
        # q_eval w.r.t the action in experience
        # eval_net的输出是（BATCH_SIZE ，2）,用action当做下标来重新组织成（BATCH_SIZE ，1）
        #q_eval,q_next = self.sess.run([self.q_eval,self.q_next],{self.x:b_s,self.x_:b_s_})
        #q_eval = self.q_eval.eval(feed_dict = {self.x:b_s})
        b_a = np.array(b_a).reshape(-1,1)
        q_next = self.q_next.eval(feed_dict = {self.x_:b_s_})

        
        
        q_target_ = b_r + self.GAMMA * np.max(q_next,axis = 1)  # shape (batch, 1)
        q_target_ = np.array(q_target_).reshape(-1,1)
        
        
        #print(q_target_.shape)
        for i in range(self.BATCH_SIZE):
            if b_mask[i]:
                q_target_[i] = b_r[i]
        
        #print(q_target.shape)
        self.sess.run([self.train_op],{self.q_target:q_target_,self.action_index:b_a,self.x:b_s})
        return self.loss.eval({self.q_target:q_target_,self.action_index:b_a,self.x:b_s}).mean()
        # self.summary_writer.add_summary(summary,   self.step_counter-128)
        
    def decay_epsilon(self):
        self.EPSILON = max(self.EPSILON-0.0001,0.005)
    
    def place(self,state,enables):
        # 这个函数 主要用于测试， 返回的 action是 0-63 之间的一个数值，
        # action 表示的是 要下的位置。
        action = self.choose_action(state,enables)
        
        return action

    def save_model(self):  # 保存 模型
        print('model saved successfully!')
        self.saver.save(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))

    def load_model(self):# 重新导入模型
        print('model loaded successfully!')
        self.saver.restore(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))