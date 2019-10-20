

python cls_condwgan.py --manualSeed 3483 --cls_weight 0.1 --preprocessing --image_embedding "./SABR/features/AWA2/features_resnet_ae_nodec"  --class_embedding "./SABR/features/AWA2/att"  --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 61 --ngh 2048 --ndh 4096 --lr 0.00001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset AWA2 --batch_size 128 --nz 85 --attSize 85 --resSize 1024 --syn_num 2400 --modeldir "./SABR/models/AWA2/SABR-T/models_awa2/" --model_path "./SABR/models/AWA2/logs_classifier_now/models_27.ckpt"  --nclass_all 50

# unseen class accuracy=  65.20058135379655
# unseen=27.9557, seen=90.5525, h=42.7221



python sep_clswgan.py --manualSeed 3483 --cls_weight 0.1 --preprocessing --image_embedding "./SABR/features/AWA2/features_resnet_ae_nodec"  --class_embedding "./SABR/features/AWA2/att"  --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 80 --ngh 2048 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset AWA2 --batch_size 128 --nz 85 --attSize 85 --resSize 1024 --syn_num 2400 --modeldir --modeldir "./SABR/models/AWA2/models_awa2/SABR-T/marg_awa20.008/"  --generator_checkpoint 60 --conditional_modeldir "./SABR/models/AWA2/SABR-T/models_awa2/" --model_path "./SABR/models/AWA2/logs_classifier_now/models_27.ckpt"  --regulariser 0.008 --nclass_all 50

# unseen class accuracy=  88.9910505145872
# unseen=79.7245, seen=91.0645, h=85.0180

