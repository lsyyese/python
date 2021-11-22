1、文本转置
awk '{
    for (i=1;i<=NF;i++){
        if (NR==1){
            res[i]=$i
        }
        else{
            res[i]=res[i]" "$i
        }
    }
}END{
    for(j=1;j<=NF;j++){
        print res[j]
    }
}' file.txt

2、打印文件第10行
grep -n "" file.txt | grep -w '10' | cut -d: -f2
sed -n '10p' file.txt
awk '{if(NR==10){print $0}}' file.txt
######################
row_num=$(cat file.txt | wc -l)
echo $row_num
if [ $row_num -lt 10 ];then
    echo "The number of row is less than 10"
else
    awk '{if(NR==10){print $0}}' file.txt
fi
############################
i=0
while read line
do 
  if [ $i -eq 9 ];then
     echo $line  
     break
  fi
  i=$(($i+1))
done <file.txt

if [ $i -lt 9 ] ;then
  echo "null"
fi
3、匹配有效号码
(xxx) xxx-xxxx 或 xxx-xxx-xxxx
sed -rn "/^((\([0-9]{3}\) )|[0-9]{3}-)[0-9]{3}-[0-9]{4}$/p" file.txt
grep  -E '^([0-9]{3}-|\([0-9]{3}\) )[0-9]{3}-[0-9]{4}$' file.txt

4、统计词频
cat words.txt |awk '{for(i=1;i<=NF;i++){a[$i]=a[$i]+1}}END{for(v in a){print v ,a[v] |"sort -k2 -nr"}}'
cat words.txt |awk '{for(i=1;i<=NF;i++){a[$i]=a[$i]+1}}END{for(v in a){print v ,a[v]}}'|sort -k2 -nr
cat Words.txt| tr -s ' ' '\n' | sort | uniq -c | sort -r | awk '{print $2, $1}'
cat words.txt | xargs -n 1 | sort | uniq -c | sort -nr | awk '{print $2" "$1}
