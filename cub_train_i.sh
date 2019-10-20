python cls_condwgan.py --manualSeed 3483 --cls_weight 0.1 --preprocessing --image_embedding "./SABR/features/CUB/features_resnet_ae_nodec" --class_embedding "./SABR/features/CUB/att" --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 45 --ngh 2048 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB --batch_size 128 --nz 312 --attSize 312 --resSize 1024 --syn_num 2400 --modeldir "./SABR/models/CUB/SABR-I/models_cub/" --model_path"./SABR/models/CUB/logs_classifier_now/models_132.ckpt" --nclass_all 200 --gzsl
# 63.87
#unseen=55.0352, seen=58.6716, h=56.7953

##this dataset has 30 to 54 instances per class



