import datetime
import pickle

import psycopg2
import os
import json
import pytz
from PIL import Image
from numpy import asarray
from pathlib import Path

source_path = Path(__file__).resolve()
source_dir = source_path.parent


def get_utc_now():
    return str(int(datetime.datetime.now(tz=pytz.utc).timestamp() * 1000))


class SqlDatabase:
    def __init__(self, dbname="image_database", user="postgres", password=5711):
        self.cur = None
        self.conn = None
        path = os.path.join(source_dir, 'db_credentials.json')
        with open(path, 'r') as f:
            credentials = json.load(f)
        self.config = {
            "dbname": credentials["dbname"],
            "user": credentials["user"],
            "password": credentials["password"]
        }
        self.connect()
        # self.create_table()
        pass

    def connect(self):
        """
        Connects to postgress SQL database
        :return:
        """
        self.conn = psycopg2.connect(
            f'dbname={self.config["dbname"]} user={self.config["user"]} host=\'localhost\' password={self.config["password"]}')
        self.cur = self.conn.cursor()
        return None

    def commit(self):
        """
        Commits our changes to the postgress database
        :param conn:
        :param cur:
        :return:
        """
        self.cur.close()
        self.conn.commit()
        self.connect()
        return None

    def create_table(self):

        sql_syntax = f'''
                    CREATE TABLE IF NOT EXISTS image_data (
                        image_id VARCHAR NOT NULL,
                        image_path VARCHAR NOT NULL,
                        style VARCHAR NOT NULL,
                        PRIMARY KEY (image_id)
                    )
                '''
        self.cur.execute(sql_syntax)
        self.commit()

        sql_syntax = f'''
                            CREATE TABLE IF NOT EXISTS vector_data (
                                image_id VARCHAR NOT NULL,
                                vector_folder_path VARCHAR NOT NULL,
                                PRIMARY KEY (image_id),
                                FOREIGN KEY(image_id) REFERENCES image_data(image_id)
                            )
                        '''
        self.cur.execute(sql_syntax)
        self.commit()
        self.cur.execute(sql_syntax)
        self.commit()

        sql_syntax = f'''
                                    CREATE TABLE IF NOT EXISTS average_vector_data (
                                        style VARCHAR NOT NULL,
                                        style_vector_folder_path VARCHAR NOT NULL,
                                        PRIMARY KEY (style)
                                    )
                                '''
        self.cur.execute(sql_syntax)
        self.commit()

        sql_syntax = f'''
                                            CREATE TABLE IF NOT EXISTS styles (
                                                style_id VARCHAR NOT NULL,
                                                style VARCHAR NOT NULL,
                                                PRIMARY KEY (style)
                                            )
                                        '''
        self.cur.execute(sql_syntax)
        self.commit()

        sql_syntax = f'''
                                    CREATE TABLE IF NOT EXISTS training_data (
                                        image_id VARCHAR NOT NULL,
                                        style_name VARCHAR NOT NULL,
                                        layer_stats VARCHAR NOT NULL,
                                        score_stats VARCHAR NOT NULL,
                                        PRIMARY KEY (image_id, style_name),
                                        FOREIGN KEY(image_id) REFERENCES vector_data(image_id)
                                    )
                                '''
        self.cur.execute(sql_syntax)
        self.commit()

    def insert_images(self, path, style):
        image_id = get_utc_now()
        image = Image.open(path)
        data = asarray(image)
        pickle_string = pickle.dumps(data)
        self.cur.execute('''INSERT INTO image_table VALUES (%s, %s, %s)''', (image_id, pickle_string, style))
        self.commit()

        style_id = 0
        styles = list()
        for e in self.fetch_data('styles'):
            styles.append(e[1])
            style_id += 1

        if style not in styles:
            print("Inserted style: " + str(style))
            sql_syntax = f'''
                                        INSERT INTO styles(style_id, style)
                                        VALUES('{style_id}', '{style}');
                                        '''
            self.cur.execute(sql_syntax)
            self.commit()
        return None

    def insert_vector_data(self, image_id, pickle_strings):
        try:
            self.cur.execute('''INSERT INTO vector_table VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                             (image_id,
                              pickle_strings['average'],
                              pickle_strings['block1_conv1'],
                              pickle_strings['block2_conv1'],
                              pickle_strings['block3_conv1'],
                              pickle_strings['block4_conv1'],
                              pickle_strings['block5_conv1'],
                              pickle_strings['block5_conv2'],
                              pickle_strings['block5_pool']))
            self.commit()
        except:
            self.cur.close()
            self.cur = self.conn.cursor()

    def insert_average_vector_data(self, style, style_average_vectors):
        try:
            self.cur.execute('''INSERT INTO average_vector_table VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                             (style,
                              style_average_vectors['average'],
                              style_average_vectors['block1_conv1'],
                              style_average_vectors['block2_conv1'],
                              style_average_vectors['block3_conv1'],
                              style_average_vectors['block4_conv1'],
                              style_average_vectors['block5_conv1'],
                              style_average_vectors['block5_conv2'],
                              style_average_vectors['block5_pool']))
            self.commit()
        except:
            self.cur.close()
            self.cur = self.conn.cursor()

    def insert_training_data(self, image_id, style_name, layer_stats, score_stats):
        try:
            sql_syntax = f'''INSERT INTO training_data(image_id, style_name, layer_stats, score_stats) 
            VALUES('{image_id}', '{style_name}', '{layer_stats}', '{score_stats}');'''
            self.cur.execute(sql_syntax)
            self.commit()
            return "Success"
        except Exception as e:
            curs = self.conn.cursor()
            curs.execute("ROLLBACK")
            self.commit()
            return "key Error" + str(e)

    def fetch_image_ids(self, table="image_table"):
        try:
            sql_syntax = f''' SELECT image_id FROM {table};'''
            self.cur.execute(sql_syntax)
            data = self.cur.fetchall()
            self.commit()
        except Exception as e:
            print("fetch data", e)
            curs = self.conn.cursor()
            curs.execute("ROLLBACK")
            self.commit()
            data = []
        return data

    def fetch_image_data(self, image_id):
        try:
            sql_syntax = f''' SELECT image_arr, style FROM image_table WHERE image_id = '{image_id}';'''
            self.cur.execute(sql_syntax)
            data = self.cur.fetchall()[0]
            self.commit()
        except Exception as e:
            print("fetch data", e)
            curs = self.conn.cursor()
            curs.execute("ROLLBACK")
            self.commit()
            data = []
        return data

    def fetch_data(self, table="image_table"):
        try:
            sql_syntax = f'''
                    SELECT * FROM {table};
                    '''
            self.cur.execute(sql_syntax)
            data = self.cur.fetchall()
            self.commit()
        except Exception as e:
            print("fetch data", e)
            curs = self.conn.cursor()
            curs.execute("ROLLBACK")
            self.commit()
            data = []
        return data
    
    def fetch_average_vector(self, style):
        sql_syntax = f'''
                SELECT style_vector_folder_path FROM average_vector_data WHERE style = '{style}';
                '''
        self.cur.execute(sql_syntax)
        data = self.cur.fetchall()[0][0]
        self.commit()
        return data

    def fetch_image_paths(self, imageid=None, style=None):
        if style is None:
            sql_syntax = f'''
                    SELECT style FROM image_data WHERE image_id = '{imageid}';
                    '''
            self.cur.execute(sql_syntax)
            style = self.cur.fetchall()
            self.commit()

            sql_syntax = f'''
                            SELECT image_path FROM image_data WHERE image_id = '{imageid}';
                            '''
            self.cur.execute(sql_syntax)
            image_paths = [self.cur.fetchall()[0][0]]
            self.commit()
        else:
            sql_syntax = f'''
                                SELECT image_id FROM image_data WHERE style = '{style}';
                                '''
            self.cur.execute(sql_syntax)
            image_ids = [e[0] for e in self.cur.fetchall()]
            self.commit()
            image_paths = []
            for image_id in image_ids:
                sql_syntax = f'''
                                    SELECT image_path FROM image_data WHERE image_id = '{image_id}';
                                '''
                self.cur.execute(sql_syntax)
                image_paths.append(self.cur.fetchall()[0][0])
                self.commit()

        return image_paths

    def fetch_vector_paths(self, img_id=None, style=None):
        vector_list = list()
        if style is None:
            sql_syntax = f'''SELECT average, block1_conv1, block2_conv1, 
            block3_conv1, block4_conv1, block5_conv1, block5_conv2, 
            block5_pool FROM vector_table WHERE image_id = '{img_id}';'''
            self.cur.execute(sql_syntax)
            result = self.cur.fetchall()[0]
            self.commit()
            entry = dict()
            entry['average'] = result[0]
            entry['block1_conv1'] = result[1]
            entry['block2_conv1'] = result[2]
            entry['block3_conv1'] = result[3]
            entry['block4_conv1'] = result[4]
            entry['block5_conv1'] = result[5]
            entry['block5_conv2'] = result[6]
            entry['block5_pool'] = result[7]
            vector_list.append(entry)
        else:
            sql_syntax = f'''SELECT image_id FROM image_table WHERE style = '{style}';'''
            self.cur.execute(sql_syntax)
            image_ids = [e[0] for e in self.cur.fetchall()]
            self.commit()
            for image_id in image_ids:
                sql_syntax = f'''SELECT average, block1_conv1, block2_conv1, 
                            block3_conv1, block4_conv1, block5_conv1, block5_conv2, 
                            block5_pool FROM vector_table WHERE image_id = '{image_id}';'''
                self.cur.execute(sql_syntax)
                result = self.cur.fetchall()[0]
                self.commit()
                entry = dict()
                entry['average'] = result[0]
                entry['block1_conv1'] = result[1]
                entry['block2_conv1'] = result[2]
                entry['block3_conv1'] = result[3]
                entry['block4_conv1'] = result[4]
                entry['block5_conv1'] = result[5]
                entry['block5_conv2'] = result[6]
                entry['block5_pool'] = result[7]
                vector_list.append(entry)
        return vector_list

    def fetch_style_name(self, styleid):
        sql_syntax = f'''
                SELECT style FROM styles WHERE style_id = '{styleid}';
                '''
        self.cur.execute(sql_syntax)
        style_name = self.cur.fetchall()[0][0]
        self.commit()
        return style_name
    
    def drop_all(self):
        sql_syntax = f'''
                DROP TABLE IF EXISTS training_data;
                DROP TABLE IF EXISTS vector_data;
                DROP TABLE IF EXISTS image_data;
                DROP TABLE IF EXISTS styles;
                DROP TABLE IF EXISTS average_vector_data;
                '''
        self.cur.execute(sql_syntax)
        self.commit()
        self.create_table()

    def drop_tables(self):
        sql_syntax = f'''
                DROP TABLE IF EXISTS vector_data;
                DROP TABLE IF EXISTS image_data;
                DROP TABLE IF EXISTS styles;
                DROP TABLE IF EXISTS average_vector_data;
                '''
        self.cur.execute(sql_syntax)
        self.commit()
        self.create_table()


sql = SqlDatabase()
