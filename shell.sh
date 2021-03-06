
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
find / -size +50k -size -100K -exec rm -rf {}\;
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
awk -v name="along" -F: '{print name":"$0}' awkdemo
awk -F: '$3>=0 && $3<=1000 {print $1,$3}' /etc/passwd
awk -F: '!($3==0) {print $1}' /etc/passwd
awk -F: '{if($3>10 && $3<1000)print $1,$3}' /etc/passwd
awk 'BEGIN{ test=100;if(test>90){print "very good"}else if(test>60){ print "good"}else{print "no pass"}}'
# 以along开头的行，以：为分隔，显示每一行的每个单词和其长度
awk -F: '/^along/{i=1;while(i<=NF){print $i,length($i); i++}}' awkdemo
awk -F: '{for(i=1;i<=NF;i++) {print$i,length($i)}}' awkdemo
# 计算1+2+3+...+100=5050
awk 'BEGIN{i=1;sum=0;while(i<=100){sum+=i;i++};print sum}'
#求男m、女f各自的平均
# cat sort.txt
xiaoming m 90
xiaohong f 93
xiaohei m 80
xiaofang f 99
[root@along ~]# awk '{m[$2]++;score[$2]+=$3}END{for(i in m){printf "%s:%6.2f\n",i,score[i]/m[i]}}' sort.txt
[root@along ~]# echo "2008:08:08 08:08:08" | awk 'sub(/:/,"-",$1)'
2008-08:08 08:08:08
[root@along ~]# echo "2008:08:08 08:08:08" | awk 'gsub(/:/,"-",$0)'
2008-08-08 08-08-08



8、sed
# 用sed修改test.txt的23行test为tset；
sed –i '23s/test/tset/g' test.txt
# 查看/web.log第25行第三列的内容
sed –n '25p' /web.log | cut –d " " –f3
head –n25 /web.log | tail –n1 | cut –d " " –f3
awk –F "" 'NR==23{print $3}' /web.log
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



牛客题
#写一个 bash脚本以输出数字 0 到 500 中 7 的倍数(0 7 14 21...)的命令
for i in {0..500..7}
do 
echo $i
done
# 写一个 bash脚本以输出一个文本文件 nowcoder.txt 中第5行的内容。
sed -n '5p'   nowcoder.txt
awk 'NR==5{print $0}' nowcoder.txt
# 写一个 bash脚本以输出一个文本文件 nowcoder.txt中空行的行号,可能连续,从1开始
awk '/^$/{print NR }'  nowcoder.txt
# 写一个 bash脚本以去掉一个文本文件 nowcoder.txt中的空行
sed '/^$/d' nowcoder.txt
#写一个 bash脚本以统计一个文本文件 nowcoder.txt中字母数小于8的单词
awk '
{
for(i=1;i<=NF;i++){
if (length($i) <8)
print $i
}
}' nowcoder.txt

#统计所有进程占用内存大小的和

awk '
{
total = total + $6
}END{print total}
' nowcoder.txt

sum=0;
while read p
do
    arr=($p)
    ((sum+=arr[5]))
done <nowcoder.txt
echo $sum

#  统计每个单词出现的个数
awk '
{
for(i=1;i<=NF;i++)
  {
  a[$i]=a[$i]+1
   }
}
END{
  for(v in a)
   {
     print v,a[v]
    }

}' nowcoder.txt | sort -n -k2
# 第二列是否有重复
#!/bin/bash
awk '
{
a[$2]=a[$2] + 1
}
END{
for(i in a)
  {
   if(a[i]>1)
   print a[i],i
   }
}' nowcoder.txt

# 转置文件的内容
awk '
{
for(i=1;i<=NF;i++)
  {
    if(NR==1)
      {
       a[i]=$i
      }
    else
      a[i] = a[i]" "$i
  }

}
END{
for(j in a)
 {
  print a[j]
 }

}
' nowcoder.txt
# 写一个 bash脚本以统计一个文本文件 nowcoder.txt中每一行出现的1,2,3,4,5数字个数并且要计算一下整个文档中一共出现了几个1,2,3,4,5数字数字总数
awk -F '[12345]' '
BEGIN{
  sum=0
}
{
  print "line"NR" number: "NF-1;
  sum= sum + NF-1
  } 
END{
  print "sum is "sum
}
' nowcoder.txt
# 写一个bash脚本以实现一个需求，求输入的一个的数组的平均值
awk '
{
if(NR==1)
  all=$0
if(NR>1)
  total+=$0
 }
END{
 printf "%.3f" ,total/all
 }'
 
# 写一个 bash脚本以实现一个需求，去掉输入中的含有B和b的单词
grep -vE "B|b"
# 判断输入的是否为IP地址
 awk '{
     if ($0 ~ /^((25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\.){3}(25[0-5]|2[0-4][0-9]|1[09][0-9]|[1-9][0-9]|[0-9])$/) {
         print("yes");
     } else if ($0 ~ /[0-9].[0-9].[0-9].[0-9]/){
         print "no";
     } else {
         print "error"
     }
 }' nowcoder.txt
 
 while read ip
 do
   if [[ $ip =~ ^((25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\.){3}(25[0-5]|2[0-4][0-9]|1[09][0-9]|[1-9][0-9]|[0-9])$ ]]; then
      echo "yes"
   elif [[ $ip =~ [0-9].[0-9].[0-9].[0-9] ]]; then
      echo "no"
   else 
      echo "error"
   fi
 
 done < nowcoder.txt
 
 # 将字段逆序输出文件nowcoder.txt的每一行，其中每一字段都是用英文冒号: 相分隔。
 awk -F: '
{
for(i=NF;i>=1;i--)
 {
  if(i==NF){
    a = $i
    continue
    }
  a = a":"$i
  }
 print a
}
' nowcoder.txt

# 域名进行计数排序处理
awk -F '/' '{print $3}'  nowcoder.txt |sort|uniq -c|sort -nr| awk '{print $1,$2}'
# 打印边长为5的等腰三角形。
你的脚本应该输出
    *
   * *
  * * *
 * * * *
* * * * *

for((i=1;i<=5;i++))
do

 for((j=5-i;j>=1;j--))
 do
   printf " "
 done
 
 for((k=1;k<=i;k++))
 do
   printf "* "
 done
 
 printf "\n"
 
done

#打印只有一个数字的行
awk -F "[0-9]" '{if(NF==2) print $0}' nowcoder.txt
 
 while read line
 do
     let count=0
     for (( i = 0; i < ${#line}; i++))
         do
             [[ ${line:i:1} =~ [0-9] ]] && ((count++))
			 # if [[ ${line:i:1} =~ [0-9] ]]; then 
             #     count=$(($count+1))
			 # fi
         done
     if [ $count -eq 1 ];then
         printf "$line\n"
     fi
 done < nowcoder.txt
 
awk -F "" '{ # ""表示以字符分割
    for (i = 1; i <= NF; i++) {
        if ($i ~ /[0-9]/) {
            count++
        }
    }
    if (count == 1) {
        print($0)
    }
    count = 0
}'  nowcoder.txt

# 我们有一个文件nowcoder.txt，里面的每一行都是一个数字串，假设数字串为“123456789”，那么我们要输出为123,456,789。
cat nowcoder.txt | xargs -n1  printf "%'d\n"

printf "%-10s %-8s %-4s\n" 姓名 性别 体重kg  
printf "%-10s %-8s %-4.2f\n" 郭靖 男 66.1234
printf "%-10s %-8s %-4.2f\n" 杨过 男 48.6543
printf "%-10s %-8s %-4.2f\n" 郭芙 女 47.9876
 
xargs -n1 -n num 后面加次数，表示命令在执行的时候一次用的argument的个数，默认是用所有的 
# cat test.txt
a b c d e f g
h i j k l m n
o p q
r s t
u v w x y z
# cat test.txt | xargs -n3
a b c
d e f
g h i
j k l
m n o
p q r
s t u
v w x
y z

awk -F "" '{
    number = "";
    for(i = 1; i<=NF;i++){
        j = NF + 1 - i
        number = $j number # 可以用空格
        if(i%3==0 && i!=NF) number = ","number
    }
    print number
}'
# 假设我们有一个nowcoder.txt，假设里面的内容如下
111:13443
222:13211
111:13643
333:12341
222:12123
现在需要你写一个脚本按照以下的格式输出
[111]
13443
13643
[222]
13211
12123
[333]
12341
awk -F: '{
 a[$1]=a[$1]"\n"$2
}
END{
 for( i in a){
   print "["i"]"a[i]
 }
}' nowcoder.txt

# 现在需要你统计出2020年4月23号的访问ip次数，并且按照次数降序排序。你的脚本应该输出：
grep '23/Apr/2020' nowcoder.txt | awk '{print $1}' | sort | uniq -c | sort -nr | awk '{print $1,$2}'
# 现在你需要统计2020年04月23日20-23点的去重IP访问量，你的脚本应该输出 5
grep   '23\/Apr\/2020:2[0-2]' nowcoder.txt  | awk '{print $1}' | sort | uniq | wc -l 

# 现在需要你写脚本统计访问3次以上的IP，你的脚本应该输出
awk '{print $1}' nowcoder.txt | sort | uniq -c | sort -nr | awk '{if($1>3) print $1,$2}'
awk '{print $1}' nowcoder.txt | sort | uniq -c | sort -nr | awk '$1>3{print $1,$2}'
# 现在需要你查询192.168.1.22的详细访问情况，按访问频率降序排序。你的脚本应该输出
grep '192.168.1.22' |  awk '{print $7 }' | sort | uniq -c | sort -nr |awk '{print $1,$2}'
# 现在需要你统计百度爬虫抓取404的次数，你的脚本应该输出
grep 'http://www.baidu.com/search/spider.html' | grep ' 404 ' | wc -l
# 现在需要你统计每分钟的请求数，并且按照请求数降序排序。你的脚本应该输出
awk '{print $4}' nowcoder.txt | awk -F: '{
 a[$2":"$3]++
}
END{
for(i in a)
 {
   print a[i],i
  
 }
}' | sort -nr -k1 
# 假设netstat命令运行的结果我们存储在nowcoder.txt里，格式如下：
grep -v 'Proto' nowcoder.txt | grep 'tcp'| awk '{print $6}' | sort | uniq -c | sort -nr | awk '{print $2,$1}' 
# 现在需要你查看和本机3306端口建立连接并且状态是established的所有IP，按照连接数降序排序。你的脚本应该输出
grep 'ESTABLISHED' nowcoder.txt | grep '3306' | awk '{print $5}' | awk -F: '{print $1}' | sort |uniq -c | sort -nr |awk '{print $1,$2}'
awk '{
if($0 ~"3306" && $6=="ESTABLISHED" ){
    a[$5]++
}}
END{
for(i in a){
   print a[i],i
}
}' | sed 's/:3306//' | sort -nr -k1
# 现在需要你输出每个IP的连接数，按照连接数降序排序。你的脚本应该输出
grep 'tcp' nowcoder.txt | awk '{print $5}' | awk -F: '{print $1}' | sort | uniq -c |sort -nr | awk '{print $2,$1}'

# 现在需要你输出和本机3306端口建立连接的各个状态的数目，按照以下格式输出
TOTAL_IP 3
ESTABLISHED 20
TOTAL_LINK 20
awk '{
if($1=="tcp" && $5 ~ "3306"){ # 必须是双引号
  a[$5]++;
  state[$6]++;
  n++;
}}
END{
print "TOTAL_IP "length(a);
for(i in state){
  print i,state[i]
}
print "TOTAL_LINK "n

}' nowcoder.txt
#假设我们的日志nowcoder.txt里，内容如下
12-May-2017 10:02:22.789 信息 [main] org.apache.catalina.startup.VersionLoggerListener.log Server version:Apache Tomcat/8.5.15
12-May-2017 10:02:22.813 信息 [main] org.apache.catalina.startup.VersionLoggerListener.log Server built:May 5 2017 11:03:04 UTC
12-May-2017 10:02:22.813 信息 [main] org.apache.catalina.startup.VersionLoggerListener.log Server number:8.5.15.0
12-May-2017 10:02:22.814 信息 [main] org.apache.catalina.startup.VersionLoggerListener.log OS Name:Windows, OS Version:10
12-May-2017 10:02:22.814 信息 [main] org.apache.catalina.startup.VersionLoggerListener.log Architecture:x86_64
现在需要你提取出对应的值，输出内容如下
serverVersion:Apache Tomcat/8.5.15
serverName:8.5.15.0
osName:Windows
osVersion:10

awk -F [:,] '{
if($3~"Server version"){
  print "serverVersion:"$4
  }
if($3~" Server number"){
  print "serverName:"$4
  }
if($3~"OS Name"){
  print "osName:"$4
  print "osVersion:"$6
  }

}' nowcoder.txt
# 现在需要你统计VSZ，RSS各自的总和（以M兆为统计），输出格式如下
awk '{
  vsz = vsz + $5;
  rss = rss + $6;
}
END{
print "MEM TOTAL";
print "VSZ_SUM:"vsz/1024"M,RSS_SUM:"rss/1024"M"
}'

awk '{
    v += $5
    r += $6
} END {
    printf("MEM TOTAL\nVSZ_SUM:%0.1fM,RSS_SUM:%0.3fM", v/1024, r/1024)
}'
