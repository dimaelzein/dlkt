#!/usr/bin/env bash

{
  dataset_name="ednet-kt1"
  data_type="only_question"
  fold=0


  weight_decays='0.001 0.0001 0.00001 0.000001'
  dropouts='0.2 0.3 0.4 0.5'
  for weight_decay in ${weight_decays}
  do
    for dim_emb in 64 128
    do
        for dropout in ${dropouts}
        do
          echo -e "weight_decay: ${weight_decay}, dim_emb: ${dim_emb}, dropout: ${dropout}"
          python F:/code/myProjects/dlkt/example/train/lpkt.py \
            --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
            --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
            --optimizer_type "adam" --weight_decay "${weight_decay}" --momentum 0.9 \
            --train_strategy "valid_test" --num_epoch 200 \
            --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
            --main_metric "AUC" --use_multi_metrics False \
            --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "StepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
            --train_batch_size 64 --evaluate_batch_size 256 \
            --enable_clip_grad False --grad_clipped 10.0 \
            --num_concept 188 --num_question 11858 \
            --dim_e "${dim_emb}" --dim_k "${dim_emb}" --dim_correct 50 --dropout "${dropout}" \
            --save_model False --debug_mode False --seed 0
      done
    done
  done
} >> F:/code/myProjects/dlkt/example/result_local/lpkt_our_setting_ednet-kt1_fold_0_ob.txt