python /Users/dream/Desktop/code/projects/dlkt/example/train/simple_kt.py \
  --setting_name "our_setting" --dataset_name "assist2012" --data_type "single_concept" \
  --train_file_name "assist2012_train_fold_0.txt" --valid_file_name "assist2012_valid_fold_0.txt" --test_file_name "assist2012_test_fold_0.txt" \
  --optimizer_type "adam" --weight_decay 0.0001 --momentum 0.9 \
  --train_strategy "valid_test" --num_epoch 200 \
  --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
  --main_metric "AUC" --use_multi_metrics False \
  --learning_rate 0.002 --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
  --train_batch_size 64 --evaluate_batch_size 256 \
  --enable_clip_grad False --grad_clipped 10.0 \
  --num_concept 265 --num_question 53091 \
  --dim_model 256 --num_block 3 --num_head 8 --dim_ff 256 --dim_final_fc 64 --dim_final_fc2 64 --seq_len 200 \
  --dropout 0.2 --key_query_same True --separate_qa False --difficulty_scalar False --use_sample_weight False \
  --sample_weight_method "highlight_tail" --tail_weight 1.1 \
  --use_LLM_emb4question False --use_LLM_emb4concept False --train_LLM_emb True \
  --save_model False --debug_mode False --seed 1 