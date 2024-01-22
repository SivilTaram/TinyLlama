
for i in {0..49}
do
  instance_name="n$i"
  command="sailctl job create qiantinyrandredhigh$instance_name -g 1 --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9 -p low --command-line-args \"cd /home/aiops/liuqian/TinyLlama/TinyLlama;./pretrain_tinyllama_config.sh $instance_name\" -f config.yaml"

  echo "Running command for $instance_name..."
  echo "$command"

  # Uncomment the line below to actually execute the command
  eval "$command"
  sleep 10
done