
folder_path="/home/aiops/liuqian/TinyLlama/hf_dataset/qwen2_data_mixture/train"  # 替换为您要遍历的文件夹路径

# 遍历文件夹中的所有文件
find "$folder_path" -type f -print0 | while IFS= read -r -d $'\0' file_path; do
    # 调用Python脚本，并将文件路径作为参数传递
    command="sailctl job create qianlabelmistral -g 1 --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9 -p low --command-line-args \"pip install transformers==4.36; cd /home/aiops/liuqian/TinyLlama/TinyLlama/scripts; python label_mistral_ppl.py $file_path\""

    echo "$command"
    sleep 5
done