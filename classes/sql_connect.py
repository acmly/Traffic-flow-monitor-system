import string
from pprint import pprint
import pymysql

# DB_CONFIG = {
# 	"host": "127.0.0.1",
# 	"port": 3306,
# 	"user": "root",
# 	"passwd": "123456",
# 	"db": "test",
# 	"charset": "utf8"
# }

class SQLManager(object):

	def __init__(self,
				 host:str="127.0.0.1", port:int=3306, user:str="root",
				 passwd:str="123456", db:str="yolo", charset:str="utf8"):
		self.conn = None
		self.cursor = None

		self.host = host
		self.port = port
		self.user = user
		self.passwd = passwd
		self.db = db
		self.charset = charset

		self.connect()

	def connect(self):
		self.conn = pymysql.connect(
			host=self.host,
			port=self.port,
			user=self.user,
			passwd=self.passwd,
			db=self.db,
			charset=self.charset
		)
		self.cursor = self.conn.cursor(cursor=pymysql.cursors.DictCursor)

	def get_list(self, sql, args=None)->list:
		self.cursor.execute(sql, args)
		result = self.cursor.fetchall()
		return result

	def get_one(self, sql, args=None) ->dict:
		self.cursor.execute(sql, args)
		result = self.cursor.fetchone()
		return result

	def modify(self, sql, args=None):
		self.cursor.execute(sql, args)
		self.conn.commit()

	def multi_modify(self, sql, args=None):
		self.cursor.executemany(sql, args)
		self.conn.commit()

	def create(self, sql, args=None):
		self.cursor.execute(sql, args)
		self.conn.commit()
		last_id = self.cursor.lastrowid
		return last_id

	def close(self):
		self.cursor.close()
		self.conn.close()

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.close()

if __name__ == "__main__":
	db = SQLManager()
	show_data_db1 = db.get_list('select * from user ')
	pprint(show_data_db1)
	show_data_db1 = db.get_list('select * from auto ')
	pprint(show_data_db1)

	with SQLManager() as sql_manager:
		result = sql_manager.get_list("SELECT * FROM user")
		pprint(result)