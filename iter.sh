#!/bin/sh

num_epochs=3
acc_num_epochs=0
#python train.py -model_type audio -data /data/OpenMT/1h/libri1h.1 -save_model /data/OpenMT/libri_exp/models/libri-model -gpuid 0 -batch_size 16 -max_grad_norm 20 -learning_rate 0.1 -learning_rate_decay 0.98 -epochs $num_epochs -valid_batch_size 8 -enc_layers 5 -dec_layers 2 -rnn_size 500 -encoder_type brnn -report_every 50

last_model=$(ls -1 -t /data/OpenMT/libri_exp/models/ | head -1)
echo $last_model

for j in 1 2 3
do
for i in 1 2 3
do
acc_num_epochs=$(($acc_num_epochs + $num_epochs))
python train.py -model_type audio -data /data/OpenMT/1h/libri1h.$i -save_model /data/OpenMT/libri_exp/models/libri1-model -gpuid 0 -batch_size 16 -max_grad_norm 20 -learning_rate 0.1 -learning_rate_decay 0.98 -epochs $acc_num_epochs -valid_batch_size 8 -enc_layers 5 -dec_layers 2 -rnn_size 500 -encoder_type brnn -report_every 50 -train_from /data/OpenMT/libri_exp/models/$last_model
last_model=$(ls -1 -t /data/OpenMT/libri_exp/models/ | head -1)
echo $last_model
done
done
