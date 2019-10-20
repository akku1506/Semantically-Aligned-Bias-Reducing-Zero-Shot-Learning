from __future__ import print_function
import os,os.path
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys
import random
import argparse
import util
import classifier2

#os.environ['CUDA_VISIBLE_DEVICE']='0,1'
#python clswgan_new.py --manualSeed 3483 --val_every 1 --cls_weight 0.1 --preprocessing --image_embedding resnet_finetune_models  --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 70 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB --batch_size 64 --nz 312 --attSize 312 --resSize 2048 --syn_num 2400 --modeldir models_cub  --dataroot '/home/test/notebooks/transductive+learning/xlsa17/data' --classifier_modeldir './models_classifier'  --classifier_checkpoint 49 --unlabelled_num 20
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='./logs_classifier_now/models_27.ckpt', type=str)
parser.add_argument('--dataset', default='FLO', help='FLO')
parser.add_argument('--dataroot', default='/BS/xian/work/cvpr18-code-release/data/', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
#parser.add_argument('--unlabelled_num', type=int, default=0, help='number of unlabelled instances per class')
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discriminator')
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=1, help='weight of the classification loss')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--netG_name', default='')
parser.add_argument('--netD_name', default='')
parser.add_argument('--modeldir', default='./models/', help='folder to output  model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')

opt = parser.parse_args()
print(opt)


if not os.path.exists(opt.modeldir):
    os.makedirs(opt.modeldir)

random.seed(opt.manualSeed)
tf.set_random_seed(opt.manualSeed)

###################model definition of functions######################################################3
def generator(x,opt,name="generator",reuse=False,isTrainable=True):
    #describes the model of generator ie ngh,resSize from command lines
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        
        net = tf.layers.dense(inputs=x, units=opt.ngh, \
                            kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.02),  \
                            activation=tf.nn.leaky_relu, name='gen_fc1',trainable=isTrainable,reuse=reuse)        
        
        net = tf.layers.dense(inputs = net, units = opt.resSize, \
                            kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.02), activation = tf.nn.relu ,name='gen_fc2', trainable = isTrainable, reuse=reuse)
        # the output is relu'd as the encoded representation is also the activations by relu
                      

        return tf.reshape(net, [-1, opt.resSize])

def discriminator(x,opt,name="discriminator",reuse=False,isTrainable=True):
    #describes the model of discriminator ie ndh,resSize from command lines    
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        
        net = tf.layers.dense(inputs=x, units=opt.ndh, \
                            kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.02),  \
                            activation=tf.nn.leaky_relu, name='disc_fc1',trainable=isTrainable,reuse=reuse)
           
        real_fake = tf.layers.dense(inputs=net, units=1, \
                            kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.02),  \
                            activation=None, name='disc_rf',trainable=isTrainable,reuse=reuse)        
        
        return tf.reshape(real_fake, [-1])

def classificationLayer(x,classes,name="classification",reuse=False,isTrainable=True):
       
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        net = tf.layers.dense(inputs=x, units=classes,  \
                            kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.02),  \
                            activation=None, name='fc1',trainable=isTrainable,reuse=reuse)

        net = tf.reshape(net, [-1, classes])    
    return net

def next_feed_dict(data,opt):
    batch_feature, batch_labels, batch_att = data.next_batch(opt.batch_size)
    batch_label = util.map_label(batch_labels, data.seenclasses)
    z_rand = np.random.normal(0, 1, [opt.batch_size, opt.nz]).astype(np.float32)
    
    return batch_feature,batch_att,batch_label,z_rand

def next_unseen_feed_dict(data,opt):
    batch_feature, batch_labels, batch_att = data.next_unseen_batch(opt.batch_size)
    z_rand = np.random.normal(0, 1, [opt.batch_size, opt.nz]).astype(np.float32)
    return batch_feature,batch_att,batch_label,z_rand

##################################################################################

### data reading
data = util.DATA_LOADER(opt)
print("#####################################")
print("# of training samples: ", data.ntrain)
print(data.seenclasses)
print(data.unseenclasses)
print(data.ntrain_class)
print(data.ntest_class)
print(data.train_mapped_label.shape)
print("#####################################")
#sys.exit(0)
####################################################################################
g1 = tf.Graph()
g2 = tf.Graph()
################### graoh1 definition ##########################################################
with g1.as_default():

    ########## placeholderS ############################ 
    input_res = tf.placeholder(tf.float32,[opt.batch_size, opt.resSize],name='input_features')
    input_att = tf.placeholder(tf.float32,[opt.batch_size, opt.attSize],name='input_attributes')
    noise_z = tf.placeholder(tf.float32,[opt.batch_size, opt.nz],name='noise')
    input_label = tf.placeholder(tf.int32,[opt.batch_size],name='input_label')
    
    ########## model definition ###########################
    train = True
    reuse = False

    noise = tf.concat([noise_z, input_att], axis=1)

    gen_res = generator(noise,opt,name='cond_gen',isTrainable=train,reuse=reuse)
    classificationLogits = classificationLayer(gen_res,data.seenclasses.shape[0],name='classifier',isTrainable=False,reuse=reuse)
    clusterLogits = classificationLayer(gen_res,opt.attSize,name='cluster',isTrainable=False,reuse=reuse)
    
    targetEmbd = tf.concat([input_res,input_att], axis=1)
    targetDisc = discriminator(targetEmbd,opt,name='cond_disc',isTrainable=train,reuse=reuse)
    genTargetEmbd = tf.concat([gen_res,input_att], axis=1)
    genTargetDisc = discriminator(genTargetEmbd,opt,name='cond_disc',isTrainable=train, reuse=True)
    ############################################################    

    ############ classification loss #########################
    input_labels = tf.one_hot(input_label,data.seenclasses.shape[0])
    classificationLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=classificationLogits,labels=input_labels))
    within_cluster = tf.matmul(clusterLogits,data.mapped_attributes,transpose_b= True)
    clusterLoss = -tf.reduce_sum(tf.multiply(tf.nn.log_softmax(within_cluster),input_labels))
        
    ############ conditional discriminator loss ##########################
    
    genDiscMean = tf.reduce_mean(genTargetDisc)
    targetDiscMean = tf.reduce_mean(targetDisc)
    discriminatorLoss = tf.reduce_mean(genTargetDisc - targetDisc)
    alpha = tf.random_uniform(shape=[opt.batch_size,1], minval=0.,maxval=1.)


    interpolates = alpha * input_res + ((1 - alpha) * gen_res)
    interpolate = tf.concat([interpolates, input_att], axis=1)
    gradients = tf.gradients(discriminator(interpolate,opt,name='cond_disc',reuse=True,isTrainable=train), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradientPenalty = tf.reduce_mean((slopes-1.)**2)   
        
    gradientPenalty = opt.lambda1*gradientPenalty
    discriminatorLoss = discriminatorLoss + gradientPenalty


    #Wasserstein generator loss
    genLoss = -genDiscMean
    generatorLoss = genLoss + opt.cls_weight*(classificationLoss+0.01*clusterLoss)


    #################### getting parameters to optimize ####################
    discParams = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='cond_disc')
    generatorParams = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='cond_gen')

    ##############################################################################
    for params in discParams:
        print (params.name)
        print ('...................')

    
    for params in generatorParams:
        print (params.name)
        print ('...................')

    ##################################################################################

    discOptimizer = tf.train.AdamOptimizer(learning_rate=opt.lr,beta1=opt.beta1,beta2=0.999)
    genOptimizer = tf.train.AdamOptimizer(learning_rate=opt.lr, beta1=opt.beta1,beta2=0.999)

    discGradsVars = discOptimizer.compute_gradients(discriminatorLoss,var_list=discParams)    
    genGradsVars = genOptimizer.compute_gradients(generatorLoss,var_list=generatorParams)



    discTrain = discOptimizer.apply_gradients(discGradsVars)
    generatorTrain = genOptimizer.apply_gradients(genGradsVars)

    #################### what all to visualize  ############################
    tf.summary.scalar("DiscriminatorLoss",discriminatorLoss)
    tf.summary.scalar("ClassificationLoss",classificationLoss)
    tf.summary.scalar("ClusterLoss",clusterLoss)
    tf.summary.scalar("GeneratorLoss",generatorLoss)

    for g,v in discGradsVars:    
        tf.summary.histogram(v.name,v)
        tf.summary.histogram(v.name+str('grad'),g)


    for g,v in genGradsVars:    
        tf.summary.histogram(v.name,v)
        tf.summary.histogram(v.name+str('grad'),g)
    
    merged_all = tf.summary.merge_all()

############### training g1 graph ################################################
k=1 

with tf.Session(graph = g1) as sess:
    sess.run(tf.global_variables_initializer())
    

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(opt.modeldir, sess.graph)


    params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='classifier')
    params1 = ['resnet_v1_101/classifier/fc3/weights','resnet_v1_101/classifier/fc3/biases']
    for name in params1:
        v = tf.contrib.framework.load_variable(opt.model_path, name)
        
        v = np.squeeze(v)
        if 'weights' in name:
            new_name = name.replace('resnet_v1_101/classifier/fc3/weights','classifier/fc1/kernel:0')
        if 'biases' in name:
            new_name = name.replace('resnet_v1_101/classifier/fc3/biases','classifier/fc1/bias:0')

        print (new_name)
        for i, s in enumerate(params):
                if new_name in s.name:
                    a=tf.assign(s,v)
                    sess.run(a)
                    print ('Model_loaded')
    
    params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cluster')
    params1 = ['resnet_v1_101/cluster/fc3/weights','resnet_v1_101/cluster/fc3/biases']
    for name in params1:
        v = tf.contrib.framework.load_variable(opt.model_path, name)
        
        v = np.squeeze(v)
        if 'weights' in name:
            new_name = name.replace('resnet_v1_101/cluster/fc3/weights','cluster/fc1/kernel:0')
        if 'biases' in name:
            new_name = name.replace('resnet_v1_101/cluster/fc3/biases','cluster/fc1/bias:0')

        print (new_name)
        for i, s in enumerate(params):
                if new_name in s.name:
                    a=tf.assign(s,v)
                    sess.run(a)
                    print ('Model_loaded')

    print ("Model loaded")
    saver = tf.train.Saver()
    for epoch in range(opt.nepoch):
        print ("Starting at epoch"+str(epoch))
        for i in range(0, data.ntrain, opt.batch_size):
            for j in range(opt.critic_iter):
                batch_feature,batch_att,batch_label,z_rand = next_feed_dict(data,opt)
                batch_unseen_feature,batch_unseen_att,_,z_unseen_rand = next_unseen_feed_dict(data,opt)
                fd = {  input_res:batch_feature,\
                        input_att:batch_att,\
                        input_label:batch_label,\
                        noise_z:z_rand \
                    }
                _,discLoss,merged = sess.run([discTrain,discriminatorLoss,merged_all],feed_dict=fd)

            
            batch_feature,batch_att,batch_label,z_rand = next_feed_dict(data,opt)
            batch_unseen_feature,batch_unseen_att,_,z_unseen_rand = next_unseen_feed_dict(data,opt)
            fd = {  input_res:batch_feature,\
                    input_att:batch_att,\
                    input_label:batch_label,\
                    noise_z:z_rand \
                    
                }
            _,genLoss,merged = sess.run([generatorTrain,generatorLoss,merged_all],feed_dict=fd)                

            summary_writer.add_summary(merged, k)
            
            
            k=k+1

        if (epoch%5==0) or (epoch==opt.nepoch-1):
            saver.save(sess, os.path.join(opt.modeldir, 'models_'+str(epoch)+'.ckpt')) 
            print ("Model saved")

##################### graph 2 definition ########################################################
with g2.as_default():
    
    ########## placeholderS ############################ data.unseenclasses, data.attribute,
    syn_att = tf.placeholder(tf.float32,[None, opt.attSize],name='input_attributes')
    noise_z1 = tf.placeholder(tf.float32,[None, opt.nz],name='noise')
    ########## model definition ##################################################

    noise1 = tf.concat([noise_z1, syn_att], axis=1)
    gen_res = generator(noise1,opt,name='cond_gen',isTrainable=False,reuse=False)



############ getting features from g2 graph ############################3
syn_res = np.empty((0,opt.resSize),np.float32)
syn_label = np.empty((0),np.float32)

with tf.Session(graph = g2) as sess:
    
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cond_gen')
    
    saver = tf.train.Saver(var_list=params)
            
    for var in params:
        print (var.name+"\t")

    string = opt.modeldir+'/models_'+str(opt.nepoch-1)+'.ckpt'
    #string = opt.modeldir+'/models_'+str(65)+'.ckpt'
    #string = './models_'+str(9)+'.ckpt'

    print (string) 
    
    try:
        saver.restore(sess, string)
    except:
        print("Previous weights not found of generator") 
        sys.exit(0)

    print ("Model loaded")

    saver = tf.train.Saver()

    for i in range(0,data.unseenclasses.shape[0]):
        iclass = data.unseenclasses[i]
        iclass_att = np.reshape(data.attribute[iclass],(1,opt.attSize))
        #print (iclass_att.shape)
        #print (iclass_att)
        batch_att = np.repeat(iclass_att,[opt.syn_num],axis=0)
        #print (batch_att.shape)
        z_rand = np.random.normal(0, 1, [opt.syn_num, opt.nz]).astype(np.float32)

        syn_features = sess.run(gen_res,feed_dict={syn_att:batch_att, noise_z1:z_rand})
        syn_res = np.vstack((syn_res,syn_features))
        temp=np.repeat(iclass,[opt.syn_num],axis=0)
        #print (temp.shape)
        syn_label = np.concatenate((syn_label,temp))
    '''
    for i in range(0,data.seenclasses.shape[0]):
        iclass = data.seenclasses[i]
        iclass_att = np.reshape(data.attribute[iclass],(1,opt.attSize))
        #print (iclass_att.shape)
        #print (iclass_att)
        batch_att = np.repeat(iclass_att,[5],axis=0)
        #print (batch_att.shape)
        z_rand = np.random.normal(0, 1, [5, opt.nz]).astype(np.float32)

        syn_features = sess.run(gen_res,feed_dict={syn_att:batch_att, noise_z1:z_rand})
        syn_res = np.vstack((syn_res,syn_features))
        temp=np.repeat(iclass,[5],axis=0)
        #print (temp.shape)
        syn_label = np.concatenate((syn_label,temp))
    '''
    
############## evaluation ################################################
if opt.gzsl:
    train_X = np.concatenate((data.train_feature, syn_res), axis=0)
    train_Y = np.concatenate((data.train_label, syn_label), axis=0)
    nclass = opt.nclass_all
    train_cls = classifier2.CLASSIFICATION2(train_X, train_Y, data, nclass, 'logs_gzsl_classifier','models_gzsl_classifier', opt.classifier_lr, 0.5, 50, 100, True)
    print('unseen=%.4f, seen=%.4f, h=%.4f' % (train_cls.acc_unseen, train_cls.acc_seen, train_cls.H))
    
else:
    train_cls = classifier2.CLASSIFICATION2(syn_res, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.shape[0], 'logs_zsl_classifier','models_zsl_classifier', opt.classifier_lr , 0.5, 50, 100, False)
    acc = train_cls.acc
    print('unseen class accuracy= ', acc)


