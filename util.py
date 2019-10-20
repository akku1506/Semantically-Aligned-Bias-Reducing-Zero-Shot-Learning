from __future__ import print_function
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import sys
import h5py

def map_label(label, classes):
    mapped_label =  np.empty_like(label)
    for i in range(classes.shape[0]):
        mapped_label[label==classes[i]] = i    

    return mapped_label


class DATA_LOADER(object):
    def __init__(self, opt):
        print (opt.matdataset)
        if opt.matdataset=='mat':
            print ("outside")
            self.read_matdataset(opt)
        else:
            print ("inside")
            self.read_mat_h5dataset(opt)   
            #if opt.dataset == 'imageNet1K':
            #    self.read_matimagenet(opt)
            #else:
            #    self.read_matdataset(opt)

        self.index_in_epoch = 0
        self.epochs_completed = 0
                
    def read_mat_h5dataset(self, opt):
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".h5", 'r')
        feature = np.array(fid['feature']).squeeze()
        label = np.array(fid['label']).astype(int).squeeze() - 1
        print (feature)
        print (label)
        print ("............................................")
        #matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        #feature1 = matcontent['features'].T
        #label1 = matcontent['labels'].astype(int).squeeze() - 1
        #print (feature1)
        #print (label1)
        #sys.exit(0)
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
        #print ('TEST UNSEEEN')
       # print (test_unseen_loc)
        self.attribute = matcontent['att'].T.astype(np.float32) 
        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()
                
                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
              
                test_unseen_feature = _test_unseen_feature.astype(np.float32)
                test_unseen_label = np.array(label[test_unseen_loc]).astype(np.int32)
                unseenclasses = np.unique(test_unseen_label)
                """
                dictionary={}
                train_label = label[trainval_loc].astype(np.int)
                
                for clas in train_label:
                    if clas in dictionary:
                        dictionary[clas]=dictionary[clas]+1
                    else:
                        dictionary[clas]=1
                print (dictionary)
                temp=200
                for k in dictionary:
                    if dictionary[k]<=temp:
                        temp = dictionary[k]
                print (temp)
                sys.exit(0)
                          
                temp_train_unseen_feature=np.empty((0,opt.resSize),np.float32)
                temp_train_unseen_label=np.empty((0),np.int)
                temp_test_unseen_feature=np.empty((0,opt.resSize),np.float32)
                temp_test_unseen_label=np.empty((0),np.int)

                for clas in unseenclasses:
                    #print (clas)
                    indices = np.where(test_unseen_label == clas)[0]
                    #print (indices)
                    #print ('..................................')
                    #sys.exit(0)
                    temp_train_unseen_label=np.concatenate((temp_train_unseen_label,test_unseen_label[indices[0:opt.unlabelled_num]]))
                    temp_train_unseen_feature=np.vstack((temp_train_unseen_feature,test_unseen_feature[indices[0:opt.unlabelled_num]]))
                    temp_test_unseen_label=np.concatenate((temp_test_unseen_label,test_unseen_label[indices[opt.unlabelled_num:]]))
                    temp_test_unseen_feature=np.vstack((temp_test_unseen_feature,test_unseen_feature[indices[opt.unlabelled_num:]]))
                """    
                #print (temp_test_unseen_label.shape)
                #sys.exit(0)
                self.train_feature = _train_feature.astype(np.float32)
                mx = self.train_feature.max()
                self.train_feature*(1/mx)
                self.train_label = label[trainval_loc].astype(np.int32) 
                test_unseen_feature = test_unseen_feature*(1/mx)

                self.train_unseen_feature = test_unseen_feature
                self.train_unseen_label = test_unseen_label

                self.test_unseen_feature = test_unseen_feature
                self.test_unseen_label = test_unseen_label

                """
                self.train_unseen_feature = temp_train_unseen_feature*(1/mx)                
                self.train_unseen_label = temp_train_unseen_label

                self.test_unseen_feature = temp_test_unseen_feature*(1/mx)
                self.test_unseen_label = temp_test_unseen_label
                """
                self.test_seen_feature = _test_seen_feature.astype(np.float32)             
                self.test_seen_feature*(1/mx)
                self.test_seen_label = label[test_seen_loc].astype(np.int32)
                
                print(self.train_feature.shape)
                print(self.train_label.shape)

                print(self.test_seen_feature.shape)
                print(self.test_seen_label.shape)

                print(self.test_unseen_feature.shape)
                print(self.test_unseen_label.shape)

                print(self.train_unseen_feature.shape)
                print(self.train_unseen_label.shape)

                print(".....................")
                            
            else:
                self.train_feature = feature[trainval_loc].astype(np.float32)
                self.train_label = label[trainval_loc].astype(np.int) 
                self.test_unseen_feature = feature[test_unseen_loc].astype(np.float32)
                self.test_unseen_label = label[test_unseen_loc].astype(np.int) 
                self.test_seen_feature = feature[test_seen_loc].astype(np.float32) 
                self.test_seen_label = label[test_seen_loc].astype(np.int)

        else:
            self.train_feature = feature[train_loc].astype(np.float32)
            self.train_label = label[train_loc].astype(np.int)
            self.test_unseen_feature = feature[val_unseen_loc].astype(np.float32)
            self.test_unseen_label = label[val_unseen_loc].astype(np.int) 
        #for i in range(0,len(self.train_label)):
        #    print (self.train_label[i])
        #sys.exit(0)
        self.seenclasses = np.unique(self.train_label)
        self.unseenclasses = np.unique(self.test_unseen_label)
        self.ntrain = (self.train_feature).shape[0]
        self.ntrain_class = (self.seenclasses).shape[0]
        self.ntest_class = (self.unseenclasses).shape[0]
        self.train_class = np.copy(self.seenclasses)
        self.allclasses = np.arange(0, self.ntrain_class+self.ntest_class).astype(np.int32)
        self.mapped_attributes = self.attribute[self.seenclasses]
        self.train_mapped_label = map_label(self.train_label, self.seenclasses) 
        #for i in range(0,len(self.test_seen_label)):
        #    print (self.test_seen_label[i])

    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
        #print ('TEST UNSEEEN')
       # print (test_unseen_loc)
        self.attribute = matcontent['att'].T.astype(np.float32) 
        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()
                
                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
              
                test_unseen_feature = _test_unseen_feature.astype(np.float32)
                test_unseen_label = np.array(label[test_unseen_loc]).astype(np.int)
                unseenclasses = np.unique(test_unseen_label)
                """
                dictionary={}
                train_label = label[trainval_loc].astype(np.int)
                
                for clas in train_label:
                    if clas in dictionary:
                        dictionary[clas]=dictionary[clas]+1
                    else:
                        dictionary[clas]=1
                print (dictionary)
                temp=200
                for k in dictionary:
                    if dictionary[k]<=temp:
                        temp = dictionary[k]
                print (temp)
                sys.exit(0)
                          
                temp_train_unseen_feature=np.empty((0,opt.resSize),np.float32)
                temp_train_unseen_label=np.empty((0),np.int)
                temp_test_unseen_feature=np.empty((0,opt.resSize),np.float32)
                temp_test_unseen_label=np.empty((0),np.int)

                for clas in unseenclasses:
                    #print (clas)
                    indices = np.where(test_unseen_label == clas)[0]
                    #print (indices)
                    #print ('..................................')
                    #sys.exit(0)
                    temp_train_unseen_label=np.concatenate((temp_train_unseen_label,test_unseen_label[indices[0:opt.unlabelled_num]]))
                    temp_train_unseen_feature=np.vstack((temp_train_unseen_feature,test_unseen_feature[indices[0:opt.unlabelled_num]]))
                    temp_test_unseen_label=np.concatenate((temp_test_unseen_label,test_unseen_label[indices[opt.unlabelled_num:]]))
                    temp_test_unseen_feature=np.vstack((temp_test_unseen_feature,test_unseen_feature[indices[opt.unlabelled_num:]]))
                """    
                #print (temp_test_unseen_label.shape)
                #sys.exit(0)
                self.train_feature = _train_feature.astype(np.float32)
                mx = self.train_feature.max()
                self.train_feature*(1/mx)
                self.train_label = label[trainval_loc].astype(np.int32) 
                test_unseen_feature = test_unseen_feature*(1/mx)

                self.train_unseen_feature = test_unseen_feature
                self.train_unseen_label = test_unseen_label

                self.test_unseen_feature = test_unseen_feature
                self.test_unseen_label = test_unseen_label

                """
                self.train_unseen_feature = temp_train_unseen_feature*(1/mx)                
                self.train_unseen_label = temp_train_unseen_label

                self.test_unseen_feature = temp_test_unseen_feature*(1/mx)
                self.test_unseen_label = temp_test_unseen_label
                """
                self.test_seen_feature = _test_seen_feature.astype(np.float32)             
                self.test_seen_feature*(1/mx)
                self.test_seen_label = label[test_seen_loc].astype(np.int32)
                
                print(self.train_feature.shape)
                print(self.train_label.shape)

                print(self.test_seen_feature.shape)
                print(self.test_seen_label.shape)

                print(self.test_unseen_feature.shape)
                print(self.test_unseen_label.shape)

                print(self.train_unseen_feature.shape)
                print(self.train_unseen_label.shape)

                print(".....................")
                            
            else:
                self.train_feature = feature[trainval_loc].astype(np.float32)
                self.train_label = label[trainval_loc].astype(np.int32) 
                self.test_unseen_feature = feature[test_unseen_loc].astype(np.float32)
                self.test_unseen_label = label[test_unseen_loc].astype(np.int32) 
                self.test_seen_feature = feature[test_seen_loc].astype(np.float32) 
                self.test_seen_label = label[test_seen_loc].astype(np.int32)

        else:
            self.train_feature = feature[train_loc].astype(np.float32)
            self.train_label = label[train_loc].astype(np.int32)
            self.test_unseen_feature = feature[val_unseen_loc].astype(np.float32)
            self.test_unseen_label = label[val_unseen_loc].astype(np.int32) 
    
        self.seenclasses = np.unique(self.train_label)
        self.unseenclasses = np.unique(self.test_unseen_label)
        self.ntrain = (self.train_feature).shape[0]
        self.ntrain_class = (self.seenclasses).shape[0]
        self.ntest_class = (self.unseenclasses).shape[0]
        self.train_class = np.copy(self.seenclasses)
        self.allclasses = np.arange(0, self.ntrain_class+self.ntest_class).astype(np.int32)
        self.mapped_attributes = self.attribute[self.seenclasses]
        
        self.train_mapped_label = map_label(self.train_label, self.seenclasses) 

    
    def next_batch(self, batch_size):
        idx = np.random.permutation(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att


    def next_unseen_batch(self, batch_size):
        idx = np.random.permutation((self.train_unseen_feature).shape[0])[0:batch_size]
        batch_feature = self.train_unseen_feature[idx]
        batch_label = self.train_unseen_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att
 