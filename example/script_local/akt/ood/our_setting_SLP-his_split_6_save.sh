#!/usr/bin/env bash

{
  setting_name="our_setting_ood_by_school"
  dataset_name="SLP-his"
  data_type="single_concept"
  split=6

  for seed in 0 1 2 3 4
  do
    echo -e "seed: ${seed}"
    python F:/code/myProjects/dlkt/example/train/akt.py \
      --setting_name "${setting_name}" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_split_${split}.txt" --valid_file_name "${dataset_name}_valid_iid_split_${split}.txt" --test_file_name "${dataset_name}_test_ood_split_${split}.txt" \
      --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
      --train_strategy "valid_test" --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric "AUC" --use_multi_metrics False \
      --learning_rate 0.0001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
      --train_batch_size 24 --evaluate_batch_size 128 \
      --enable_clip_grad True --grad_clipped 10.0 \
      --num_concept 22 --num_question 1251 \
      --dim_model 512 --key_query_same True --num_head 4 --num_block 1 --dim_ff 512 --dim_final_fc 512 --dropout 0.2 \
      --separate_qa False --seq_representation "encoder_output" --weight_rasch_loss 0.00001 \
      --save_model True --debug_mode False --use_cpu False --seed "${seed}"
  done

} >> F:/code/myProjects/dlkt/example/result_local/akt_our_setting_ood_SLP-his_split_6_save.txt