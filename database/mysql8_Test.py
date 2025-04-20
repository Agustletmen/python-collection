import mysql.connector

conn = mysql.connector.connect(user='root', password='123', database='py')
cursor = conn.cursor()
cursor.execute("select * from user ")
values = cursor.fetchall()
for i in values:
    print(i)