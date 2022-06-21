import psycopg2
import getpass
import pandas as pd
from configparser import ConfigParser


type_conversion = {
    'int64': 'INTEGER',
    'float64': 'NUMERIC',
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
            # password=getpass.getpass(prompt='Password:',stream=None),
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


def insert_row(connection, table, keys, values):
    cursor = connection.cursor()

    query = f"""INSERT INTO {table}({','.join([str(x) for x in keys])}) 
    VALUES ({','.join([str(x) for x in values])})""".replace('inf', 'NULL')
    # TODO: find a way to store infinity in Postgres
    # try:
    cursor.execute(query)
    connection.commit()
    cursor.close()
    # except (Exception, psycopg2.DatabaseError) as error:
    #     print(error)


def update_values(connection, table, pk_name, pks, key, values):
    cursor = connection.cursor()

    for i, value in enumerate(values):
        pk = pks[i]
        query = f"""UPDATE {table}
            SET {key} = {value}
            WHERE {pk_name} = {pk};"""
    # try:

        cursor.execute(query)
    connection.commit()
    cursor.close()
    # except (Exception, psycopg2.DatabaseError) as error:
    #     print(error)


if __name__ == '__main__':
    conn = connect()
    df = pd.read_csv('data/car_loan_trainset.csv')
    # print(df[df.isnull()])
    # print(type(conn))
    # create_table(conn, pd.read_csv('data/car_loan_trainset.csv'), 'car_loan_sql', 'customer_id')
    for row in df.values.tolist():
        keys = df.columns.tolist()

        insert_row(conn, 'car_loan_sql', keys, row)
    # for key in df.columns:
    #     if key != 'customer_id':
    #         insert_row(conn, 'car_loan_sql', key, df[key].tolist())
    #         update_values(conn, 'car_loan_sql', 'customer_id', df['customer_id'], key, df[key])
    # insert_values(conn, 'car_loan_sql', )

