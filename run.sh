python pcnn_train.py \
--batch_size 16 \
--sample_batch_size 192 \
--sampling_interval 50 \
--save_interval 50 \
--dataset cpen455 \
--nr_resnet 1 \
--nr_filters 5 \
--nr_logistic_mix 5 \
--lr_decay 0.999995 \
--max_epochs 10 \
# --en_wandb True \
