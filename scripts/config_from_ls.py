import re

input_str = """
250G    redpajama_120B_sample.jsonl
30G     cleaned_cc100_ind_dedup_part1.jsonl
31G     cleaned_cc100_ind_dedup_part2.jsonl
28G     cleaned_cc100_ind_dedup_part3.jsonl
759M    cleaned_cc100_lao_dedup.jsonl
4.8G    cleaned_cc100_ms_dedup.jsonl
11G     cleaned_cc100_th_dedup.jsonl
29G     cleaned_cc100_vi_dedup_part1.jsonl
29G     cleaned_cc100_vi_dedup_part2.jsonl
21G     cleaned_cc100_vi_dedup_part3.jsonl
3.2G    ebook_id_non_ocr.jsonl
427M    ebook_id_ocr.jsonl
87M     ebook_ms_non_ocr.jsonl
127M    ebook_th_non_ocr.jsonl
398M    ebook_vi_non_ocr.jsonl
27G     indonesian_madlad_1.jsonl
27G     indonesian_madlad_2.jsonl
27G     indonesian_madlad_3.jsonl
27G     indonesian_madlad_4.jsonl
18G     indonesian_madlad_5.jsonl
3.8G    indonesian_sft_pretrain.jsonl
12G     malay_madlad.jsonl
674M    subtitle_id.jsonl
620M    subtitle_th.jsonl
194M    subtitle_vi.jsonl
4.1G    thai_madlad_10.jsonl
10G     thai_madlad_1.jsonl
10G     thai_madlad_2.jsonl
10G     thai_madlad_3.jsonl
10G     thai_madlad_4.jsonl
10G     thai_madlad_5.jsonl
10G     thai_madlad_6.jsonl
10G     thai_madlad_7.jsonl
10G     thai_madlad_8.jsonl
10G     thai_madlad_9.jsonl
3.0G    thai_sft_pretrain.jsonl
1.6G    translation_indonesian.jsonl
1.2G    translation_thai.jsonl
1.4G    translation_vietnamese.jsonl
9.9G    vietnamese_madlad_10.jsonl
27G     vietnamese_madlad_1.jsonl
27G     vietnamese_madlad_2.jsonl
27G     vietnamese_madlad_3.jsonl
27G     vietnamese_madlad_4.jsonl
27G     vietnamese_madlad_5.jsonl
27G     vietnamese_madlad_6.jsonl
27G     vietnamese_madlad_7.jsonl
27G     vietnamese_madlad_8.jsonl
27G     vietnamese_madlad_9.jsonl
3.5G    vietnamese_sft_pretrain.jsonl
1.1G    wikipedia_id_text.jsonl
945M    wikipedia_th_text.jsonl
1.5G    wikipedia_vi_text.jsonl
"""

def parse_input(input_str):
    lines = input_str.strip().split("\n")
    lines = [line.strip().split() for line in lines]
    # if it is G, then multiple by 1000
    lines = [[float(line[0][:-1]) * 1000 if line[0][-1] == "G" else float(line[0][:-1]), line[1]] for line in lines]
    prefix_dict = {}
    # if the current line shares a similar starting prefix with the previous line, then merge them
    for line in lines:
        # prefix is the one removing "_part1.jsonl" or "_1.jsonl", using re to match
        prefix = re.match(r"(.*)_(part)?\d+\.jsonl", line[1])
        if prefix is None:
            prefix = line[1].replace(".jsonl", "")
        else:
            prefix = prefix.group(1)
        prefix = "train_data_mixture_" + prefix
        if len(prefix_dict) == 0:
            prefix_dict[prefix] = line[0]
        else:
            if prefix in prefix_dict:
                prefix_dict[prefix] += line[0]
            else:
                prefix_dict[prefix] = line[0]
    # print as A: 10
    for key in prefix_dict:
        print(key + ": " + str(prefix_dict[key]))
    
    # get the key list
    key_list = list(prefix_dict.keys())
    # get the value list
    value_list = list(prefix_dict.values())
    # normalize the value list into 1
    value_list = [float(value) / sum(value_list) for value in value_list]
    print(key_list)
    print(value_list)
parse_input(input_str)