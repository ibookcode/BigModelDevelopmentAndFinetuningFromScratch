# Soft Prompt的长度
PRE_SEQ_LEN=128
# 学习率
LR=2e-2

# 可以将train_file、validation_file和test_file修改为自己的数据集
# 并将prompt_column和response_column改为JSON文件中输入文本和输出文本对应的KEY
# 可能还需要增大max_source_length和max_target_length来匹配自己数据集中的最大输入输出长度
# P-Tuning-v2方法会冻结全部的模型参数，可通过调整quantization_bit来控制原始模型的量化等级，若不加则使用FP16精度加载模型
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_train \
    --train_file AdvertiseGen/train.json \
    --validation_file AdvertiseGen/dev.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm-6b \
    --output_dir output/adgen-chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

