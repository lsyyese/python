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
head -10 file|tail -1
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
cat words.txt | xargs -n 1 | sort | uniq -c | sort -nr | awk '{print $2" "$1}'

5、grep

6、find
find . -type d -o -type f -size +4k -mtime +10
find ./ ! -type d -exec chmod 644 \;
find /var/log -name "*.log" -mtime +3
# 编写个shell脚本将当前目录下大于100K的文件转移到/tmp目录下
find . -size +100K xargs -I {} mv {} /tmp
#  找出access.log中访问top 10的ip地址
find / -name ".log" -mtime +3 -exec rm fr {} ; find /log ! -mtime -3
# 找出系统内大于50k，小于100k的文件，并删除它们
find / -size +50k -size -100K -exec rm -rf {}\:
# 在 A 文件夹下有多个子文件夹（a1、b1、c1），每个子文件夹下有好几张 jpg 图片，要求写一段代码（用 Python or Shell），把这些图片全部拷贝并存在 B 文件夹下。
find  ./A/ -maxdepth 2  -name '*.jpg' -exec cp {} ./B \;
#查找/etc下以.conf结尾的文件并以时间命名打包到/tmp下（两种方法）
find /etc/ -type f -name '*.conf' -exec tar zcf /tmp/a.tar.gz {} +
find /etc/ -type f -name '*.conf'|xargs  tar zcf  /tmp/a.tar.gz {} +
tar zcf /tmp/.a.tar.gz `find /etc/ -type f -name '*.conf'`
#取出/etc/fstab 权限
16777282 -rw-r--r--. 1 root root 501 Mar 26 13:52 /etc/fstab
ll -id /etc/fstab |awk -F'[ .]' '{print $2}'
find /tmp/ -type f -size +1k -size -10M -mtime -7 |xargs -i cp {} /tmp/
find /tmp/ -type f -size +1k -size -10M -mtime -7 -exec cp {} /tmp/ \;




7、awk
# 假如文件中每行第一个元素是 FIND，如何获取第二个元素
awk'{ if ($1 == "FIND") print $2}'
# 找出access.log中访问top 10的ip地址
awk '{print $1}' nginx.log | grep -v "^$" | sort | uniq -c | sort -nr | head -n 10
# 如何显示文本file.txt中第二大列大于56789的行？
awk -F "," '{if($2>56789){print $0}}' file.txt
# 统计所有连接到shell服务器的外部IP数，以ip为准
netstat -tnp | awk '{print $5}' | awk -F: '{print $1}' | awk '{if(NR>2)print}' | sort | uniq -c |wc -l
# 假设qq. tel文件内容:
12334:13510014336
12345:12334555666
12334:12343453453
12099:13598989899
12334:12345454545
12099:12343454544
分类如下:
[12334]
13510014336
12343453453
...........
[12099]
13598989899
12343454544
 cat qq.tel | sort -r | awk -F: '{if (tmp!=$1) {tmp=$1;print "["tmp"]"} print $2}'
 
# 处理一 下文件内容，将域名取出并进行计数排数,如处理: ;
http: //www . baidu. com/ index. html
http: / / ww .baidu. com/1.html
http:/ / www . baidu. com/2. html
http: / /post . baidu. com/ index . html
http: / /mp3. baidu. com/ index. html
http:/ / www . baidu. com/3. html
http: / /post.baidu. com/2. html
得到如F结果:域名的出现次数,域名
4 www . baidu. com
2 post .baidu. com
1 mp3. baidu. com
awk -F/ '{print $3}' yuming.txt | sort -r | uniq -c 
# 将文件中的oldboy全部替换为oldgirl，同时将49000448改为31333741。
sed -e 's#oldboy#oldgirl#g;s#49000448#31333741#g' file.txt
#用awk获取reg.txt文件中第三行的倒数第二列字段
awk 'NR==3{print $(NF-1)}' reg.txt
cat reg.txt 
Zhang   Dandan      41117397    :250:100:175
Zhang   Xiaoyu      390320151   :155:90:201
Meng    Feixue      0042789     :250:60:50
Wu   Waiwai     70271111    :250:80:75
Liu     Bingbing    41117483    :250:100:175
Wang  Xiaoai        3515064655 :50:95:135
# 显示文件reg.txt所有以41开头的ID号码的人的全名和ID号码
awk  '$3~/^41/{print $1,$2,$3}' reg.txt
# 显示小雨的姓名与id号
awk '$2~/Xiaoyu/{print $1,$2,$3}' reg.txt 
# 显示Xiaoyu的捐款.每个值时都有以520135
awk '$NF{print $4}' reg.txt |tr ':' '$'
#计算第一次捐款的总额
awk -F: '{i=i+$2}END{print i}' reg.txt
#使用awk计算0加到100
seq 100|awk '{i=i+$1}END{print i}'
#调换/etc/passwd 第一列和最后一列内容（至少2种方法）
awk -F: -vOFS=":" '{u=$1;$1=$NF;$NF=u;print $0}' /etc/passwd
sed -r 's#(^.*:)(.*)(/.*)#\3 \1#g' /etc/passwd
#找出/oldboy下面以.txt结尾的文件把里面的oldboy替换为oldgirl(三种方法)
sed -i 's#oldboy#oldgirl#g' `find /oldboy/ -type f -name '*.txt'`
awk 'gsub(/oldboy/,"oldgirl"){print $0}' `find /oldboy/ -type f -name '*.txt'`
grep 'oldboy' `find /oldboy/ -type f -name '*.txt'`|sed 's#oldboy#oldgirl#g'




8、sed
# 用sed修改test.txt的23行test为tset；
sed –i ‘23s/test/tset/g’ test.txt
# 查看/web.log第25行第三列的内容
sed –n ‘25p’ /web.log | cut –d “ ” –f3
head –n25 /web.log | tail –n1 | cut –d “ ” –f3
awk –F “ ” ‘NR==23{print $3}’ /web.log
# 显示file.txt的1,3,5,7,10,15行
sed -n "1p;3p;5p;7p;10p;15p" file.txt
awk 'NR==1||NR==3||NR==5||…||NR=15{print $0}' file.txt
# 将file.txt的制表符，即tab，全部替换成"|"
sed -i "s#\t#\|#g" file.txt
# 找出nginx.log中的404的报错数据，统计下共有多少404报错
cat nginx.log | grep ' 404 ' | wc -l 
less nginx.log | grep ' 404 ' | wc -l
# 1、sed命令
123abc456
456def123
567abc789
789def567
要求输出：
456ABC123
123DEF456
789ABC567
567DEF789

sed -r -i 's#(...)(.)(.)(.)(...)#\5\u\2\u\3\u\4\1#g' 1.txt


for i in {0..100..3}; do echo $i; done

#!/bin/sh
STRING=
if [ -z "$STRING" ]; then
 echo "STRING is empty"
fi
if [ -n "$STRING" ]; then
 echo "STRING is not empty"
fi

# 截取字符串
variable="My name is Petras, and I am developer."
echo ${variable:11:6} # 会显示 Petras

$# #输入变量数量
$@ #分隔符的方式所有变量
$* #字符串的方式所有变量


# 晚上11点到早上8点之间每两个小时查看一次系统日期与时间，写出具体配置命令
echo 1 23,1-8/2 * * * root /tmp/walldate.sh >> /etc/crontab

#  把当前目录（包含子目录）下所有后缀为“.sh”的文件后缀变更为“.shell”
#!/bin/bash
str=`find ./ -name "*.sh"`
for i in $str
do
mv $i ${i%sh}shell
done

# 假设有一个脚本scan.sh，里面有1000行代码，并在vim模式下面，请按照如下要求写入对应的指令
1） 将shutdown字符串全部替换成reboot
:%s/shutdown/reboot/g
2） 清空所有字符
:%d
3） 不保存退出
q!

# 将A 、B、C目录下的文件A1、A2、A3文件，改名为AA1、AA2、AA3.使用shell脚本实现
#!/bin/bash
file=`ls [ABC]/A[123]`
for i in $file;do
mv $i ${i%/*}/A${i#*/}
done


