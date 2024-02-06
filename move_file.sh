cd /nfs-share/sea_corpus/slimpajama/chunk2
for file in *; 
  do mv "$file" "slimpajama_chunk_2_${file}";
done
