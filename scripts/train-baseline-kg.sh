GPUID=3
export CUDA_VISIBLE_DEVICES=${GPUID}
python webnlg/finetune.py \
--data_dir=./webnlg/data/webnlg-kg \
--learning_rate=4e-5 \
--num_train_epochs 100 \
--task kg2text \
--model_name_or_path=../weights/bart/baseline-kg/ \
--train_batch_size=4 \
--eval_batch_size=4 \
--early_stopping_patience 5 \
--gpus 1 \
--N=0 \
--baseline \
--output_dir=./outputs/baseline-lr4/kg \
--max_source_length=384 \
--max_target_length=384 \
--val_max_target_length=384 \
--test_max_target_length=384 \
--eval_max_gen_length=384 \
--do_train --do_predict \
--eval_beams 3