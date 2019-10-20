python cls_condwgan.py --manualSeed 3483 --cls_weight 0.1 --preprocessing --image_embedding "./SABR/features/CUB/features_resnet_ae_nodec" --class_embedding "./SABR/features/CUB/att" --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 150 --ngh 2048 --ndh 4096 --lr 0.0002 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB --batch_size 128 --nz 312 --attSize 312 --resSize 1024 --syn_num 2400 --modeldir "./SABR/models/CUB/SABR-T/models_cub/" --model_path"./SABR/models/CUB/logs_classifier_now/models_132.ckpt" --nclass_all 200

python sep_clswgan.py --manualSeed 1 --cls_weight 0.1 --preprocessing --image_embedding "./SABR/features/CUB/features_resnet_ae_nodec" --class_embedding "./SABR/features/CUB/att" --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 120 --ngh 2048 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB --batch_size 128 --nz 312 --attSize 312 --resSize 1024 --syn_num 300 --modeldir "./SABR/models/CUB/SABR-T/marg_cub" --generator_checkpoint 100 --conditional_modeldir "./SABR/models/CUB/SABR-T/models_cub" --model_path "./SABR/models/CUB/logs_classifier_now/models_132.ckpt" --nclass_all 200 --regulariser 0.003 --gzsl
#74.0383
#unseen=67.2081, seen=73.6774, h=70.2942