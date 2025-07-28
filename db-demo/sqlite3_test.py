import sqlite3

# 连接sqlite3数据库
conn = sqlite3.connect("sqlite3.db")
cursor = conn.cursor()

# 创建数据表 user
try:
    cursor.execute("create table user(id varchar(20) primary key ,name varchar (20))")
except sqlite3.OperationalError:
    print("表已经建立")

# 插入数据 ('1','Agust')
try:
    cursor.execute("insert into  user (id,name) values ('1','Agust')")
    conn.commit()
except sqlite3.IntegrityError:
    print("主键重复")

print(cursor.rowcount)

cursor.execute("select * from user where id = ?", "1")
values = cursor.fetchall()  # 获取查询结构  结果集是一个list  每一个元素都是一个tuple（对应一行数据）
print(values)

cursor.close()
conn.close()
