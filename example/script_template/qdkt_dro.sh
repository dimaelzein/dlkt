python /Users/dream/Desktop/code/projects/dlkt/example/train/qdkt_dro.py \
  --setting_name our_setting_ood_by_school --dataset_name SLP-phy --data_type single_concept --train_file_name SLP-phy_train_split_2.txt --valid_file_name SLP-phy_valid_iid_split_2.txt --test_file_name SLP-phy_test_ood_split_2.txt \
  --optimizer_type adam --weight_decay 0.001 --momentum 0.9 --train_strategy valid_test --num_epoch 200 \
  --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 --main_metric AUC \
  --use_multi_metrics False --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type MultiStepLR --lr_schedule_step 10 \
  --lr_schedule_milestones [5] --lr_schedule_gamma 0.5 --train_batch_size 64 --evaluate_batch_size 256 --enable_clip_grad False \
  --grad_clipped 10.0 --num_concept 54 --num_question 1915 --dim_concept 64 --dim_question 64 \
  --dim_correct 64 --dim_latent 64 --rnn_type gru --num_rnn_layer 1 --dropout 0.2 \
  --num_predict_layer 3 --dim_predict_mid 128 --activate_type relu --use_dro True --beta 5.0 \
  --alpha 0.001 --max_seq_len 200 --use_LLM_emb4question False --use_LLM_emb4concept False --train_LLM_emb True \
  --save_model False --debug_mode False --seed 0 