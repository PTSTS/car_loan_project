import psycopg2
import getpass
import pandas as pd
from configparser import ConfigParser


type_conversion = {
    'int64': 'INTEGER',
    'float64': 'REAL',
}

# from Postgres documentation
def config(filename='database.ini', section='postgresql'):
    parser = ConfigParser()
    parser.read(filename)
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
    return db


def connect():
    conn = None
    try:
        params = config()

        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(
            **params,
            password=getpass.getpass(prompt='Password:',stream=None),
        )
        return conn

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


def create_table(connection: psycopg2.connect, data: pd.DataFrame, name: str, pk: str):
    """
    Create one single table from dataframe, no FK can be created
    :param connection:
    :param data:
    :param str name:
    :return:
    """

    assert pk in data.columns

    query = f"""
    CREATE TABLE {name} (
        {pk} {type_conversion[str(data[pk].dtypes)]} NOT NULL,
    """
    for column_name in data.columns:
        if column_name != pk:
            query += f"""\t{column_name} {type_conversion[str(data[column_name].dtypes)]},\n"""
    query += f"""\tPRIMARY KEY ({pk})
    )"""

    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        cursor.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    return


def insert_values(connection, database, key, values):
    cursor = connection.cursor()

    query = f"""INSERT INTO {database}({key}) VALUE(%s)"""
    try:
        cursor.executemany(query)

if __name__ == '__main__':
    conn = connect()
    # print(type(conn))
    create_table(conn, pd.read_csv('data/car_loan_trainset.csv'), 'car_loan_sql', 'customer_id')
