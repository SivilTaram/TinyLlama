import json
import matplotlib.pyplot as plt

def extract_loss_from_line(line):
    try:
        data = json.loads(line)
        loss = data.get("loss")
        return loss
    except json.JSONDecodeError:
        return None

def extract_losses_from_file(file_path):
    loss_values = []
    with open(file_path, 'r') as file:
        for line in file:
            loss = extract_loss_from_line(line)
            if loss is not None:
                loss_values.append(loss)
    return loss_values

def main(file_paths):
    # Extract loss values from the first file
    all_loss_values = []
    for file_path in file_paths:
        loss_values = extract_losses_from_file(file_path)
        all_loss_values.append(loss_values)

    number_of_files = len(file_paths)
    # Create a figure with two subplots
    fig, axs = plt.subplots(number_of_files, 1, 
                            sharex=True, figsize=(4, 2 * number_of_files), 
                            gridspec_kw={'height_ratios': [1] * number_of_files})

    # Plot histogram for the first file
    for i in range(0, number_of_files):
        axs[i].hist(all_loss_values[i], bins=100, color='blue', alpha=0.5)
        # axs[i].set_title('File ' + file_paths[i].split('/')[-1])
        axs[i].set_xlabel('File ' + file_paths[i].split('/')[-1])
        axs[i].set_ylabel('Frequency')
    
    plt.savefig('loss_histogram.png', bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    # Replace 'your_file1.jsonl' and 'your_file2.jsonl' with the actual paths to your JSON line files
    main(["../ppl_data/en_chunk_data_with_loss.json", 
          "../ppl_data/cc100_id_chunk_data_with_loss.json",
        #   "../ppl_data/madlad_id_chunk_data_with_loss.json",
          "../ppl_data/cc100_ms_chunk_data_with_loss.json",
        #   "../ppl_data/cc100_th_chunk_data_with_loss.json",
          "../ppl_data/madlad_th_chunk_data_with_loss.json",
        #   "../ppl_data/cc100_vi_chunk_data_with_loss.json",
          "../ppl_data/madlad_vi_chunk_data_with_loss.json"])
