python cls_condwgan.py --manualSeed 3483 --cls_weight 0.1 --preprocessing --image_embedding "./SABR/features/SUN/features_resnet_ae_nodec"  --class_embedding "./SABR/features/SUN/att" --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 66 --ngh 2048 --ndh 4096 --lr 0.0002 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 128 --nz 102 --attSize 102 --resSize 1024 --syn_num 2400 --modeldir "./SABR/models/SUN/SABR-T/models_sun/" --model_path "./SABR/models/SUN/logs_classifier_now/models_134.ckpt" --nclass_all 717
#unseen class accuracy=  62.361111111111114
#unseen=49.3750, seen=32.0155, h=38.8440


python sep_clswgan.py --manualSeed 3483 --cls_weight 0.1 --preprocessing --image_embedding "./SABR/features/SUN/features_resnet_ae_nodec"  --class_embedding "./SABR/features/SUN/att" --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 110 --ngh 2048 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 128 --nz 102 --attSize 102 --resSize 1024 --syn_num 300 --modeldir "./SABR/models/SUN/SABR-T/marg_sun/" --generator_checkpoint 65 --conditional_modeldir "./SABR/models/SUN/SABR-T/models_sun/" --model_path "./SABR/models/SUN/logs_classifier_now/models_134.ckpt"  --regulariser 0.002 --nclass_all 717 --gzsl
#unseen=58.8194, seen=41.4729, h=48.6460

