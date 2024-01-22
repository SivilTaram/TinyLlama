import numpy as np
import random

random.seed(42)
np.random.seed(42)

# Set the number of learning rates you want to try
num_learning_rates = 100

# Create a geometric sequence of learning rates
learning_rates = np.logspace(np.log10(0.00001), np.log10(0.0004), num=num_learning_rates)
learning_rates = list(learning_rates)
# Shuffle the array to randomize the order
random.shuffle(learning_rates)

def generate_train_group(weights, sample_folder=None):
    # groups = ['train_data_mixture_en_redpajama',
    #           'train_data_mixture_cleaned_cc100_ind_dedup',
    #           'train_data_mixture_cleaned_cc100_th_dedup', 
    #           'train_data_mixture_cleaned_cc100_vi_dedup']
    groups =  ["train_data_mixture_en_redpajama", 
               "train_data_mixture_cleaned_cc100_ind_dedup",
               "train_data_mixture_cleaned_cc100_lao_dedup", 
               "train_data_mixture_cleaned_cc100_ms_dedup", 
               "train_data_mixture_cleaned_cc100_th_dedup", 
               "train_data_mixture_cleaned_cc100_vi_dedup",
               "train_data_mixture_ebook_id_non_ocr", 
               "train_data_mixture_ebook_id_ocr", 
               "train_data_mixture_ebook_ms_non_ocr", 
               "train_data_mixture_ebook_th_non_ocr", 
               "train_data_mixture_ebook_vi_non_ocr", 
               "train_data_mixture_indonesian_madlad", 
               "train_data_mixture_malay_madlad", 
               "train_data_mixture_subtitle_id", 
               "train_data_mixture_subtitle_th",
               "train_data_mixture_subtitle_vi",
               "train_data_mixture_thai_madlad", 
               "train_data_mixture_translation_indonesian", 
               "train_data_mixture_translation_thai", 
               "train_data_mixture_translation_vietnamese",
               "train_data_mixture_vietnamese_madlad", 
               "train_data_mixture_vietnamese_sft_pretrain",
               "train_data_mixture_wikipedia_id_text", 
               "train_data_mixture_wikipedia_th_text",
               "train_data_mixture_wikipedia_vi_text"]

    assert len(groups) == len(weights)
    if sample_folder is None:
        output_group = [f"  {group}: {num}" for group, num in zip(groups, weights)]
    else:
        output_group = [f"  train_{sample_folder}_{group}: {num}" for group, num in zip(groups, weights)]
    return "\n".join(output_group)

def generate_valid_group(sample_folder=None):
    groups = ["en_redpajama",
            "id_wikipedia",
            "th_wikipedia",
            "vi_wikipedia",
            # "all_test",
            # "ms_cc100_all"
            ]
    weights = [1.0] * len(groups)
    if sample_folder is None:
        output_group = [f"  {group}: {num}" for group, num in zip(groups, weights)]
    else:
        output_group = [f"  valid_{sample_folder}_{group}: {num}" for group, num in zip(groups, weights)]
    return "\n".join(output_group)


def generate_weights_exponential(num_samples=100, num_elements=16, scale_parameter=1.0, custom_prior=None):
    if custom_prior is None or len(custom_prior) != num_elements:
        raise ValueError("Invalid custom_prior. It should be a list of length num_elements.")

    # 将 custom_prior 转换为 NumPy 数组
    custom_prior = np.array(custom_prior)
    
    # 归一化 custom_prior，使得每行之和为1
    prior_distribution = custom_prior / custom_prior.sum()

    # 采样指数分布
    samples = np.random.exponential(scale=scale_parameter, size=(num_samples, num_elements))

    # 对每一列进行归一化，使得每组数字之和为1
    normalized_samples = samples / samples.sum(axis=1)[:, np.newaxis]

    # 将先验分布与指数分布相结合
    combined_samples = 1.0 * normalized_samples + 9.0 * prior_distribution  # 调整权重以平衡两者
    return combined_samples

def generate_weights_with_custom_prior_and_relative_noise(num_samples=100, num_elements=16, prior_values=None, noise_factor_lower=0.5, noise_factor_upper=2.5):
    if prior_values is None or len(prior_values) != num_elements:
        raise ValueError("Invalid prior_values. It should be a list of length num_elements.")

    # 将 prior_values 转换为 NumPy 数组
    prior_values = np.array(prior_values)

    # 计算每个元素的上下浮动范围
    lower_bounds = prior_values * noise_factor_lower
    upper_bounds = prior_values * noise_factor_upper

    # 从均匀分布中采样随机噪声
    noise = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(num_samples, num_elements))

    # 构建先验分布
    prior_distribution = prior_values + noise

    # 使得每行之和为1
    prior_distribution /= prior_distribution.sum(axis=1)[:, np.newaxis]

    return prior_distribution


def generate_config(output_paths):
    prior_dist = [0.2650183233668776, 0.09434652311860842, 0.0008045956297418403, 0.00508835180864405, 0.011660806228142613, 0.08374579018393331, 0.003392234539096033, 0.0004526512963106269, 9.22263765316734e-05, 0.00013462930827037382, 0.0004219091708000691, 0.13356923497690631, 
                #   0.004028278515176539, 
                  0.012720879521610124,
                  0.000714489399797102,
                  0.0006572454419498564, 0.000205654218932697, 0.09975289691529272,
                #   0.003180219880402531,
                  0.0016961172695480165, 0.0012720879521610124, 0.0014841026108545145, 0.26809253591793336, 0.0037102565271362863, 0.0011660806228142614, 0.0010017692623267972, 0.0015901099402012655]
    all_weights = generate_weights_with_custom_prior_and_relative_noise(len(output_paths), len(prior_dist), prior_dist)
    for output_path, weights in zip(output_paths, all_weights):
        train_group = generate_train_group(weights, sample_folder=None)
        valid_group = generate_valid_group(sample_folder="data_mixture")
        with open(output_path, "w", encoding="utf8") as f:
            f.write("train:\n")
            f.write(train_group)
            f.write("\n")
            f.write("valid:\n")
            f.write(valid_group)
            print(f"Write to {output_path}")

def generate_random_learning_rate():
    # more than 1e-5, less than 4e-4
    # random_lr = np.random.uniform(low=0.00001, high=0.0004)
    random_lr = learning_rates.pop()
    return "learning_rate: " + str(random_lr)

def generate_random_batch_size():
    batch_size_candidates = [16 * i for i in range(1, 20)]
    return "global_batch_size: " + str(random.choice(batch_size_candidates))

def generate_random_max_step():
    max_step_candidates = [1000 * i for i in range(1, 11)]
    return "max_step: " + str(random.choice(max_step_candidates))

def generate_config_with_learning_rate(output_paths):
    # prior_dist = [0.25, 0.25, 0.25, 0.25]
    # all_weights = generate_weights_with_custom_prior_and_relative_noise(len(output_paths), len(prior_dist), prior_dist, noise_factor_lower=0.1, noise_factor_upper=10)
    prior_dist = [0.2650183233668776, 0.09434652311860842, 0.0008045956297418403, 0.00508835180864405, 0.011660806228142613, 0.08374579018393331, 0.003392234539096033, 0.0004526512963106269, 9.22263765316734e-05, 0.00013462930827037382, 0.0004219091708000691, 0.13356923497690631, 
                #   0.004028278515176539, 
                  0.012720879521610124,
                  0.000714489399797102,
                  0.0006572454419498564, 0.000205654218932697, 0.09975289691529272,
                #   0.003180219880402531,
                  0.0016961172695480165, 0.0012720879521610124, 0.0014841026108545145, 0.26809253591793336, 0.0037102565271362863, 0.0011660806228142614, 0.0010017692623267972, 0.0015901099402012655]
    all_weights = generate_weights_with_custom_prior_and_relative_noise(len(output_paths), len(prior_dist), prior_dist)
    for output_path, weights in zip(output_paths, all_weights):
        train_group = generate_train_group(weights, sample_folder=None)
        valid_group = generate_valid_group(sample_folder="data_mixture")
        with open(output_path, "w", encoding="utf8") as f:
            f.write("train:\n")
            f.write(train_group)
            f.write("\n")
            f.write("valid:\n")
            f.write(valid_group)
            f.write("\n")
            f.write(generate_random_learning_rate())
            f.write("\n")
            f.write("warmup_steps: 0\n")
            f.write(generate_random_max_step() + "\n")
            f.write("eval_step_interval: 100\n")
            f.write("save_step_interval: 3000\n")
            f.write("decay_lr: true\n")
            f.write("micro_batch_size: 16\n")
            f.write("global_batch_size: 128\n")
            # f.write(generate_random_batch_size() + "\n")
            f.write("model_name: tiny_LLaMA_mistral_120M\n")
            print(f"Write to {output_path}")


def generate_config_with_constant_english_metric(output_paths):
    en_magic_number = 36812
    # all_weights = generate_weights_with_custom_prior_and_relative_noise(len(output_paths), len(prior_dist), prior_dist, noise_factor_lower=0.1, noise_factor_upper=10)
    prior_dist = [0.2650183233668776, 0.09434652311860842, 0.0008045956297418403, 0.00508835180864405, 0.011660806228142613, 0.08374579018393331, 0.003392234539096033, 0.0004526512963106269, 9.22263765316734e-05, 0.00013462930827037382, 0.0004219091708000691, 0.13356923497690631, 
                #   0.004028278515176539, 
                  0.012720879521610124,
                  0.000714489399797102,
                  0.0006572454419498564, 0.000205654218932697, 0.09975289691529272,
                #   0.003180219880402531,
                  0.0016961172695480165, 0.0012720879521610124, 0.0014841026108545145, 0.26809253591793336, 0.0037102565271362863, 0.0011660806228142614, 0.0010017692623267972, 0.0015901099402012655]
    all_weights = generate_weights_with_custom_prior_and_relative_noise(len(output_paths), len(prior_dist), prior_dist)
    for output_path, weights in zip(output_paths, all_weights):
        train_group = generate_train_group(weights, sample_folder=None)
        valid_group = generate_valid_group(sample_folder="data_mixture")
        with open(output_path, "w", encoding="utf8") as f:
            f.write("train:\n")
            f.write(train_group)
            f.write("\n")
            f.write("valid:\n")
            f.write(valid_group)
            f.write("\n")
            f.write(generate_random_learning_rate())
            f.write("\n")
            f.write("warmup_steps: 0\n")
            f.write(generate_random_max_step() + "\n")
            f.write("eval_step_interval: 100\n")
            f.write("save_step_interval: 3000\n")
            f.write("decay_lr: true\n")
            f.write("micro_batch_size: 16\n")
            f.write("global_batch_size: 128\n")
            # f.write(generate_random_batch_size() + "\n")
            f.write("model_name: tiny_LLaMA_mistral_120M\n")
            print(f"Write to {output_path}")

if __name__ == "__main__":
    output_paths = []
    for i in range(0, 100):
        output_paths.append(f"../sea_config_lr_decay/n{i}.yaml")
    generate_config_with_learning_rate(output_paths)
