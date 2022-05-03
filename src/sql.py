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
        self.create_table()
        pass

    def connect(self):
        self.conn = psycopg2.connect(
            f'dbname={self.config["dbname"]} '
            f'user={self.config["user"]} '
            f'host=\'localhost\' '
            f'password={self.config["password"]}')
        self.cur = self.conn.cursor()
        return None

    def commit(self):
        self.cur.close()
        self.conn.commit()
        self.connect()
        return None

    def create_table(self):

        sql_syntax = f'''CREATE TABLE IF NOT EXISTS public.average_vector_table (
                style character NOT NULL,
                average bytea,
                block1_conv1 bytea,
                block2_conv1 bytea,
                block3_conv1 bytea,
                block4_conv1 bytea,
                block5_conv1 bytea,
                block5_conv2 bytea,
                block5_pool bytea,
                CONSTRAINT average_vector_table_pkey PRIMARY KEY (style))'''

        self.cur.execute(sql_syntax)
        self.commit()

        sql_syntax = f'''CREATE TABLE IF NOT EXISTS public.image_table (
               image_id character NOT NULL,
               image_arr bytea,
               style character,
               CONSTRAINT image_table_pkey PRIMARY KEY (image_id))'''

        self.cur.execute(sql_syntax)
        self.commit()

        sql_syntax = f'''CREATE TABLE IF NOT EXISTS public.styles (
               style_id character NOT NULL,
               style character NOT NULL,
               CONSTRAINT styles_pkey PRIMARY KEY (style))'''

        self.cur.execute(sql_syntax)
        self.commit()

        sql_syntax = f'''CREATE TABLE IF NOT EXISTS public.vector_table (
               image_id character NOT NULL,
               average bytea,
               block1_conv1 bytea,
               block2_conv1 bytea,
               block3_conv1 bytea,
               block4_conv1 bytea,
               block5_conv1 bytea,
               block5_conv2 bytea,
               block5_pool bytea,
               CONSTRAINT vector_table_pkey PRIMARY KEY (image_id))'''

        self.cur.execute(sql_syntax)
        self.commit()

        sql_syntax = f'''CREATE TABLE IF NOT EXISTS public.training_table (
               image_id character NOT NULL,
               style_name character NOT NULL,
               layer_stats character,
               score_stats character,
               CONSTRAINT training_table_pkey PRIMARY KEY (image_id, style_name),
               CONSTRAINT training_table_image_id_fkey FOREIGN KEY (image_id)
                   REFERENCES public.vector_table (image_id) MATCH SIMPLE
                   ON UPDATE NO ACTION
                   ON DELETE NO ACTION)'''

        self.cur.execute(sql_syntax)
        self.commit()

    def insert_images(self, path, style):
        image_id = get_utc_now()
        image = Image.open(path)
        data = asarray(image)
        pickle_string = pickle.dumps(data)
        try:
            self.cur.execute('''INSERT INTO image_table VALUES (%s, %s, %s)''', (image_id, pickle_string, style))
            self.commit()
        except Exception as e:
            print("INSERT INTO image_table", e)
            curs = self.conn.cursor()
            curs.execute("ROLLBACK")
            self.commit()

        style_id = 0
        styles = list()
        for e in self.fetch_data(params=['*'], table='styles'):
            styles.append(e[1])
            style_id += 1

        if style not in styles:
            print("Inserted style: " + str(style))
            sql_syntax = f''' INSERT INTO styles(style_id, style) VALUES('{style_id}', '{style}');'''
            try:
                self.cur.execute(sql_syntax)
                self.commit()
            except Exception as e:
                print("INSERT INTO styles", e)
                curs = self.conn.cursor()
                curs.execute("ROLLBACK")
                self.commit()

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
        except Exception as e:
            print("insert_vector_data", e)
            curs = self.conn.cursor()
            curs.execute("ROLLBACK")
            self.commit()
            return str(e)

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
        except Exception as e:
            print("insert_average_vector_data", e)
            curs = self.conn.cursor()
            curs.execute("ROLLBACK")
            self.commit()
            return str(e)

    def insert_training_data(self, image_id, style_name, layer_stats, score_stats):
        try:
            sql_syntax = f'''INSERT INTO training_table(image_id, style_name, layer_stats, score_stats) 
            VALUES('{image_id}', '{style_name}', '{layer_stats}', '{score_stats}');'''
            self.cur.execute(sql_syntax)
            self.commit()
            return "Success"
        except Exception as e:
            print("insert_training_data", e)
            curs = self.conn.cursor()
            curs.execute("ROLLBACK")
            self.commit()
            return str(e)

    def fetch_n_image_ids(self, n):
        try:
            sql_syntax = 'SELECT image_id FROM image_table limit ' + str(n) + ';'
            self.cur.execute(sql_syntax)
            data = self.cur.fetchall()
            self.commit()
        except Exception as e:
            print("fetch_image_data", e)
            curs = self.conn.cursor()
            curs.execute("ROLLBACK")
            self.commit()
            data = list()
        return data

    def fetch_image_data(self, image_id):
        try:
            sql_syntax = f''' SELECT image_arr, style FROM image_table WHERE image_id = '{image_id}';'''
            self.cur.execute(sql_syntax)
            data = self.cur.fetchall()[0]
            self.commit()
        except Exception as e:
            print("fetch_image_data", e)
            curs = self.conn.cursor()
            curs.execute("ROLLBACK")
            self.commit()
            data = list()
        return data

    def fetch_data(self, params, table):
        try:
            sql_syntax = 'SELECT '
            for param in params:
                sql_syntax += param + ', '
            sql_syntax = sql_syntax[:-2]
            sql_syntax += ' FROM ' + table + ';'
            self.cur.execute(sql_syntax)
            data = self.cur.fetchall()
            self.commit()
        except Exception as e:
            print("fetch_data", e)
            curs = self.conn.cursor()
            curs.execute("ROLLBACK")
            self.commit()
            data = list()
        return data
    
    def fetch_average_vector(self, style):
        try:
            sql_syntax = f'''SELECT average, block1_conv1, block2_conv1, 
            block3_conv1, block4_conv1, block5_conv1, block5_conv2, 
            block5_pool FROM average_vector_data WHERE style = '{style}';'''
            self.cur.execute(sql_syntax)
            data = self.cur.fetchall()[0]
            self.commit()
        except Exception as e:
            print("fetch_average_vector", e)
            curs = self.conn.cursor()
            curs.execute("ROLLBACK")
            self.commit()
            data = list()
        return data

    def fetch_image_ids_of_style(self, style):
        try:
            sql_syntax = f'''SELECT image_id FROM image_table WHERE style = '{style}';'''
            self.cur.execute(sql_syntax)
            data = [e[0] for e in self.cur.fetchall()]
            self.commit()
        except Exception as e:
            print("fetch_average_vector", e)
            curs = self.conn.cursor()
            curs.execute("ROLLBACK")
            self.commit()
            data = list()
        return data

    def fetch_image_vectors(self, img_id=None, style=None):
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
            image_ids = self.fetch_image_ids_of_style(style)
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

    def fetch_style_name(self, style_id):
        try:
            sql_syntax = f'''SELECT style FROM styles WHERE style_id = '{style_id}';'''
            self.cur.execute(sql_syntax)
            style_name = self.cur.fetchall()[0][0]
            self.commit()
        except Exception as e:
            print("fetch_style_name", e)
            curs = self.conn.cursor()
            curs.execute("ROLLBACK")
            self.commit()
            style_name = ''
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
