#!/usr/bin/env bash

{
  dataset_name="assist2017"
  data_type="single_concept"
  fold=0


#  weights_adv_pred_loss='1'
#  etas='5 10 20'
#  gammas='5 10 20'
#  dropouts='0.2'
#  adv_learning_rates='1 5 10 30'
#  weights_decay='0.00001 0.000001 0.0000001 0'
#  for weight_adv_pred_loss in ${weights_adv_pred_loss}
#  do
#    for eta in ${etas}
#    do
#      for gamma in ${gammas}
#      do
#        for dropout in ${dropouts}
#        do
#          for adv_learning_rate in ${adv_learning_rates}
#          do
#            for weight_decay in ${weights_decay}
#            do
#              echo -e "lr: 0.001, no lr decay, weight decay: ${weight_decay}, adv_learning_rate: ${adv_learning_rate}, dropout: ${dropout}, weight_adv_pred_loss: ${weight_adv_pred_loss}, eta: ${eta}, gamma: ${gamma}"
#              CUDA_VISIBLE_DEVICES=0 python /home/xiongzj/myProjects/KT/dlkt/example/train/lpkt_max_entropy_aug.py \
#                --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
#                --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
#                --optimizer_type "adam" --weight_decay "${weight_decay}" --momentum 0.9 \
#                --train_strategy "valid_test" --num_epoch 200 \
#                --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
#                --main_metric "AUC" --use_multi_metrics False \
#                --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "StepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
#                --train_batch_size 64 --evaluate_batch_size 256 \
#                --enable_clip_grad False --grad_clipped 10.0 \
#                --num_concept 101 --num_question 2803 \
#                --dim_e 128 --dim_k 128 --dim_correct 50 --dropout "${dropout}" \
#                --use_warm_up False --epoch_warm_up 4 \
#                --epoch_interval_generate 1 --epoch_generate 200 --weight_adv_pred_loss "${weight_adv_pred_loss}" --loop_adv 3 --adv_learning_rate "${adv_learning_rate}" --eta "${eta}" --gamma "${gamma}" \
#                --save_model False --debug_mode False --seed 0
#            done
#          done
#        done
#      done
#    done
#  done

  weights_adv_pred_loss='0.8 1 1.2 1.5'
  etas='5'
  gammas='10'
  dropouts='0.1 0.2 0.3 0.4'
  adv_learning_rates='5'
  weights_decay='0.000001'
  for weight_adv_pred_loss in ${weights_adv_pred_loss}
  do
    for eta in ${etas}
    do
      for gamma in ${gammas}
      do
        for dropout in ${dropouts}
        do
          for adv_learning_rate in ${adv_learning_rates}
          do
            for weight_decay in ${weights_decay}
            do
              echo -e "lr: 0.001, no lr decay, weight decay: ${weight_decay}, adv_learning_rate: ${adv_learning_rate}, dropout: ${dropout}, weight_adv_pred_loss: ${weight_adv_pred_loss}, eta: ${eta}, gamma: ${gamma}"
              CUDA_VISIBLE_DEVICES=0 python /home/xiongzj/myProjects/KT/dlkt/example/train/lpkt_max_entropy_aug.py \
                --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
                --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
                --optimizer_type "adam" --weight_decay "${weight_decay}" --momentum 0.9 \
                --train_strategy "valid_test" --num_epoch 200 \
                --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
                --main_metric "AUC" --use_multi_metrics False \
                --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "StepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
                --train_batch_size 64 --evaluate_batch_size 256 \
                --enable_clip_grad False --grad_clipped 10.0 \
                --num_concept 101 --num_question 2803 \
                --dim_e 128 --dim_k 128 --dim_correct 50 --dropout "${dropout}" \
                --use_warm_up False --epoch_warm_up 4 \
                --epoch_interval_generate 1 --epoch_generate 200 --weight_adv_pred_loss "${weight_adv_pred_loss}" --loop_adv 3 --adv_learning_rate "${adv_learning_rate}" --eta "${eta}" --gamma "${gamma}" \
                --save_model False --debug_mode False --seed 0
            done
          done
        done
      done
    done
  done
} >> /home/xiongzj/myProjects/KT/dlkt/example/results/lpkt_ME-ADA_our_setting_assist2017_fold_0_ob2.txt