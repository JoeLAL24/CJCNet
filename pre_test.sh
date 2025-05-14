EPOCH="ep49_Smeasure0.6677+ep50_Smeasure0.6720+ep34_Smeasure0.6599+ep41_Smeasure0.6604+ep47_Smeasure0.6594+ep43_Smeasure0.6564+ep48_Smeasure0.6655+ep46_Smeasure0.6682+ep37_Smeasure0.6556+ep44_Smeasure0.6615"

# 将EPOCH按"+"分割成数组
IFS='+' read -r -a epochs <<< "$EPOCH"

# 遍历数组，每次执行一次python命令
for epoch in "${epochs[@]}"
do
    python test.py --one_of_epoch "$epoch"
    wait
done
