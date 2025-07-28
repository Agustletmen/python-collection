import sqlite3

conn = sqlite3.connect('sqlite3.db')
cursor = conn.cursor()

cursor.execute('select * from user')
values = cursor.fetchall()
for value in values:
    print(value)