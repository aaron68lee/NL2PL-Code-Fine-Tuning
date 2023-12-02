lang=python #programming language
beam_size=10
batch_size=128
source_length=256
target_length=128
output_dir=model/$lang
data_dir=data
# dev_file=$data_dir/$lang/valid.jsonl
test_file=$data_dir/python-cleaned-val.jsonl
test_model=pytorch_model.bin #checkpoint for test

python code2nl/run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path $test_model --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size
