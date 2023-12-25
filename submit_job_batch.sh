
for i in {0..20}
do
  instance_name="n$i"
  command="sailctl job create qiantinyrandredlow$instance_name -g 1 --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9 -p low --high-vram --command-line-args \"cd /home/aiops/liuqian/TinyLlama/TinyLlama;./pretrain_tinyllama_config.sh $instance_name\""

  echo "Running command for $instance_name..."
  echo "$command"

  # Uncomment the line below to actually execute the command
  eval "$command"
done