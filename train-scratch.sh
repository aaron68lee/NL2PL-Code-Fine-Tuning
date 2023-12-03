lang=python #programming language
lr=5e-5
batch_size=32
beam_size=10
source_length=256
target_length=128
output_dir=model/$lang
data_dir=data
train_file=$data_dir/python-cleaned-train.jsonl
dev_file=$data_dir/python-cleaned-val.jsonl
# eval_steps=400 #400 for ruby, 600 for javascript, 1000 for others
# train_steps=20000 #20000 for ruby, 30000 for javascript, 50000 for others
eval_steps=2400 #400 for ruby, 600 for javascript, 1000 for others
train_steps=60000 #20000 for ruby, 30000 for javascript, 50000 for others
pretrained_model=microsoft/codebert-base #Roberta: roberta-base
#test_model=pytorch_model.bin #checkpoint for test

python code2nl/run.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --train_steps $train_steps --eval_steps $eval_steps
