import datetime
import psycopg2
import os
import json


class SqlDatabase:
    def __init__(self, dbname="image_database", user="postgres", password=5711):
        self.cur = None
        self.conn = None
        credentials = json.load(open(os.path.join('src', 'db_credentials.json')))
        self.config = {
            "dbname": credentials["dbname"],
            "user": credentials["user"],
            "password": credentials["password"]
        }
        self.connect()
        self.create_table()
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
                        image_id TIMESTAMP NOT NULL,
                        image_path VARCHAR NOT NULL,
                        style VARCHAR NOT NULL,
                        PRIMARY KEY (image_id)
                    )
                '''
        self.cur.execute(sql_syntax)
        self.commit()

        sql_syntax = f'''
                            CREATE TABLE IF NOT EXISTS vector_data (
                                image_id TIMESTAMP NOT NULL,
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
                                                style VARCHAR NOT NULL,
                                                PRIMARY KEY (style)
                                            )
                                        '''
        self.cur.execute(sql_syntax)
        self.commit()

        sql_syntax = f'''
                                    CREATE TABLE IF NOT EXISTS training_data (
                                        image_id TIMESTAMP NOT NULL,
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
        timestamp = datetime.datetime.now()
        sql_syntax = f'''
                            INSERT INTO image_data(image_id, image_path, style)
                            VALUES('{timestamp}', '{path}', '{style}');
                            '''
        self.cur.execute(sql_syntax)
        self.commit()

        styles = [e[0] for e in self.fetch_data('styles')]
        if style not in styles:
            sql_syntax = f'''
                                        INSERT INTO styles(style)
                                        VALUES('{style}');
                                        '''
            self.cur.execute(sql_syntax)
            self.commit()
        return None

    def insert_vector_data(self, image_id, vector_path):
        sql_syntax = f'''
                                    INSERT INTO vector_data(image_id, vector_folder_path)
                                    VALUES('{image_id}', '{vector_path}');
                                    '''
        self.cur.execute(sql_syntax)
        self.commit()

    def insert_average_vector_data(self, style, style_vector_folder_path):
        styles = [e[0] for e in self.fetch_data('average_vector_data')]
        if style not in styles:
            sql_syntax = f'''
                                        INSERT INTO average_vector_data(style, style_vector_folder_path)
                                        VALUES('{style}', '{style_vector_folder_path}');
                                        '''
            self.cur.execute(sql_syntax)
            self.commit()

    def insert_training_data(self, image_id, style_name, layer_stats, score_stats):
        sql_syntax = f'''
                                            INSERT INTO training_data(image_id, style_name, layer_stats, score_stats)
                                            VALUES('{image_id}', '{style_name}', '{layer_stats}', '{score_stats}');
                                            '''
        self.cur.execute(sql_syntax)
        self.commit()

    def fetch_data(self, table="image_data"):
        try:
            sql_syntax = f'''
                    SELECT * FROM {table};
                    '''
            self.cur.execute(sql_syntax)
            data = self.cur.fetchall()
            self.commit()
        except Exception as e:
            print("fetch data", e)
            self.cur.close()
            self.cur = self.conn.cursor()
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

    def fetch_vector_paths(self, imageid=None, style=None):
        if style is None:
            sql_syntax = f'''
                    SELECT style FROM image_data WHERE image_id = '{imageid}';
                    '''
            self.cur.execute(sql_syntax)
            style = self.cur.fetchall()
            self.commit()

            sql_syntax = f'''
                            SELECT vector_folder_path FROM vector_data WHERE image_id = '{imageid}';
                            '''
            self.cur.execute(sql_syntax)
            vector_folder_paths = [self.cur.fetchall()[0][0]]
            self.commit()
        else:
            sql_syntax = f'''
                                SELECT image_id FROM image_data WHERE style = '{style}';
                                '''
            self.cur.execute(sql_syntax)
            image_ids = [e[0] for e in self.cur.fetchall()]
            self.commit()
            vector_folder_paths = []
            for image_id in image_ids:
                sql_syntax = f'''
                                            SELECT vector_folder_path FROM vector_data WHERE image_id = '{image_id}';
                                            '''
                self.cur.execute(sql_syntax)
                vector_folder_paths.append(self.cur.fetchall()[0][0])
                self.commit()

        return vector_folder_paths

    def drop_table(self, table_name):
        '''
        Table name can be
        "handwritting_data_to_train"
        "handwritting_data_trained"
        :param table_name:
        :return:
        '''
        sql_syntax = f'''
                                DROP TABLE {table_name};
                                '''
        self.cur.execute(sql_syntax)
        self.commit()
        return None
