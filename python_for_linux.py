一、本机执行命令
1.os.system()
# 只返回命令执行状态(0:成功，非 0:失败)

2.os.popen()
# 会将结果保存在内存当中，可以用read()方法读取出来
res = os.popen("ls -l")
print res.read()

3.subprocess.getstatusoutput()
# 接受字符串形式的命令，返回 一个元组形式的结果，第一个元素是命令执行状态，第二个为执行结果
res = subprocess.getstatusoutput('pwd')
print res
(0, '/root')

4.subprocess.getoutput() 
# 接受字符串形式的命令，反回执行结果
res = subprocess.getstatusoutput('pwd')
print res
'/root'


二、远程执行命令
1.使用秘钥
import paramiko
transport=paramiko.Transport("192.168.1.106",22)
pkey=paramiko.RSAKey.from_private_key_file("/root/.ssh/id_rsa")
transport.connect(username="root",pkey=pkey)
 
ssh=paramiko.SSHClient()
ssh._transport=transport
stdio, stdout, stderr=ssh.exec_command("sh /soft/shell/luoshuyu.sh")
 
channel = stdout.channel
status = channel.recv_exit_status()
stdout = stdout.read().decode()
stderr = stderr.read().decode()
 
ssh.close()
transport.close()
 
print( "stdout is " + stdout )
print( "stderr is " + stderr )
print( "status is " + str(status) )
 
#stdout is sdl;fkd;lfk;as
#stderr is
#status is 0


2.使用密码
import paramiko
transport=paramiko.Transport("192.168.1.106",22)
transport.connect(username="root",password="123456")
 
ssh=paramiko.SSHClient()
ssh._transport=transport
stdio, stdout, stderr=ssh.exec_command("sh /soft/shell/luoshuyu.sh")
 
channel = stdout.channel
status = channel.recv_exit_status()
stdout = stdout.read().decode()
stderr = stderr.read().decode()
 
ssh.close()
transport.close()
 
print( "stdout is " + stdout )
print( "stderr is " + stderr )
print( "status is " + str(status) )
 
3.scp文件传输
import paramiko
transport=paramiko.Transport("192.168.1.106",22)
transport.connect(username="root",password="123456")
 
sftp=paramiko.SFTPClient.from_transport(transport)
#上传
sftp.put("/python_scripts/luoshuyu.txt","/tmp/luoshuyu.txt")
 
#下载
sftp.get("/soft/shell/luoshuyu.sh","/python_scripts/luoshuyu.sh")
 
sftp.close()
transport.close()
##################################################################
4.封装

#远程执行命令
import paramiko
def remote_exec_command(ip,port,username,password,command):
    try:
        transport = paramiko.Transport(ip, port)
        transport.connect(username=username, password=password)
 
        ssh = paramiko.SSHClient()
        ssh._transport = transport
        stdio, stdout, stderr = ssh.exec_command(command)
 
        channel = stdout.channel
        status = channel.recv_exit_status()
        stdout = stdout.read().decode()
        stderr = stderr.read().decode()
 
        return {"status":status,"stdout":stdout,"stderr":stderr}
 
        ssh.close()
        transport.close()
    except Exception as e:
        print(e)
    finally:
        try:
            ssh.close()
        except Exception as e:
            print(e)
        try:
            transport.close()
        except Exception as e:
            print(e)
 
 
#远程上传
def remote_put(ip,port,username,password,localpath,remotepath):
   try:
       transport = paramiko.Transport(ip, port)
       transport.connect(username=username, password=password)
 
       sftp = paramiko.SFTPClient.from_transport(transport)
       result=sftp.put(localpath,remotepath)
       return result
 
   except Exception as e:
       print(e)
   finally:
       try:
           sftp.close()
       except Exception as e:
           print(e)
       try:
           transport.close()
       except Exception as e:
           print(e)
 
 
#远程下载
def remote_get(ip,port,username,password,remotepath,localpath):
   try:
       transport = paramiko.Transport(ip, port)
       transport.connect(username=username, password=password)
 
       sftp = paramiko.SFTPClient.from_transport(transport)
       result=sftp.get(remotepath,localpath)
       return result
 
   except Exception as e:
       print(e)
   finally:
       try:
           sftp.close()
       except Exception as e:
           print(e)
       try:
           transport.close()
       except Exception as e:
           print(e)
 
if __name__ == "__main__":
    result1 = remote_exec_command("192.168.1.106",22,"root","123456","ls /nono")
    result2 = remote_put("192.168.1.106",22,"root","123456","/python_scripts/linux_001.py","/tmp/linux_001.py")
    result3 = remote_get("192.168.1.106",22,"root","123456","/tmp/kjkl.yy","/python_scripts/kjkl.yy")
    print(result1,result2,result3)

