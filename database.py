import MySQLdb as mdb
from datetime import datetime

class Database():
    def check_connection_to_db(self):
        """
        Check database connection
        """
        try:
            db = mdb.connect('localhost', 'admin', 'Coincard2@', 'traffic_density_estimation_log')
            print('Connect successfully!')
        except mdb.Error as e:
            print("Cannot connect to database!")

    def insert_data(self,date, time, bike, car, priority, status, image):
        """
        Insert traffic density estimation log to database
        """
        con = mdb.connect('localhost', 'admin', 'Coincard2@', 'traffic_density_estimation_log')
        with con:
            cur = con.cursor()
            sql = "INSERT INTO logs values"\
                        "(null,'%s','%s',%s, %s, %s, '%s', '%s');" % (''.join(date),
                                                    ''.join(time),
                                                    bike,
                                                    car,
                                                    priority,
                                                    ''.join(status),
                                                ''.join(image))
            cur.execute(sql)
            print("Insert successfully!")
            con.commit()
            cur.close()

    def get_data(self):
        con = mdb.connect('localhost', 'admin', 'Coincard2@', 'traffic_density_estimation_log')
        with con:
            cur = con.cursor()
            sql = "SELECT * FROM logs;"
            cur.execute(sql)
            rs = cur.fetchall()
            print(rs)
            for row_number, row_data in enumerate(rs):
                # print(row_number)
                print("")
                for column_number, data in enumerate(row_data):
                    print("%7s"%(str(data)), end='|')

    def get_last_image_path(self):
        con = mdb.connect('localhost', 'admin', 'Coincard2@', 'traffic_density_estimation_log')
        with con:
            cur = con.cursor()
            sql = "SELECT image FROM logs ORDER BY image DESC LIMIT 1, 1;"
            cur.execute(sql)
            rs = cur.fetchall()
            return rs[0][0]

if __name__ == '__main__':
    time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    time_temp = time_now.split(' ')
    date = time_temp[0]
    time = time_temp[1]
    print(date, time)
    db = Database()
    db.check_connection_to_db()
    # db.insert_data(date, time, 3, 2, 1, 'Traffic jam', 'F:/LVTN/DATA 3/Img_cut_video12_16/video12_16/video12B_001.jpg')
    # db.get_data()
    a = db.get_last_image_path()
    print(type(a))