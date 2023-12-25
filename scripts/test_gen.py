import numpy as np

def generate_weights_exponential(num_samples=20, num_elements=17, scale_parameter=1.0):
    # 采样指数分布
    samples = np.random.exponential(scale=scale_parameter, size=(num_samples, num_elements))

    # 对每一列进行归一化，使得每组数字之和为1
    normalized_samples = samples / samples.sum(axis=1)[:, np.newaxis]

    return normalized_samples

# 调用函数并打印结果
weights_exponential = generate_weights_exponential()
for i, sample in enumerate(weights_exponential):
    print(f"Sample {i + 1}: {sample}, Sum: {sample.sum()}, Variance: {np.var(sample)}")
