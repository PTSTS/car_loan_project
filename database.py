import psycopg2
import pandas as pd
from configparser import ConfigParser


# data type conversions for creating database tables, all dtypes from loaded DF MUST be present here!
type_conversion = {
    'int64': 'INTEGER',
    'float64': 'NUMERIC',
}

# partially from Postgres documentation
def config(filename='database.ini', section='postgresql'):
    """
    Load specific database configuration for connecting
    :param str filename: file with stored database information
    :param str section: specify a section if multiple configurations are stored
    :return dict db: a dict with configuration values if successfully loaded, else None
    """
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
    """
    Connect to Postgres database as defined in database.ini
    :return conn or None: return psycopg2 connection instance if successful.
    """
    try:
        params = config()
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(
            **params,
        )
        return conn
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


def create_table(connection: psycopg2.connect, data: pd.DataFrame, name: str, pk: str):
    """
    Create one single table from dataframe with same columns, no FK can be created
    :param psycopg2.connect connection: established database connection
    :param pd.DataFrame data: dataframe, dtypes will be used to set data type in the table
    :param str name: name of the table
    :param str pk: primary key, for this function is mandatory
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


def insert_row(connection, table: str, keys, values):
    """
    Insert a single row into database, does not include validation or constraint check
    :param psycopg2.connect connection: psycopg2 connection instance
    :param str table: table name
    :param iterable keys: list of column keys in the table to insert
    :param iterable values: list of values to insert into specified keys
    """
    cursor = connection.cursor()

    # inf will be replaced with NULL
    query = f"""INSERT INTO {table}({','.join([str(x) for x in keys])}) 
    VALUES ({','.join([str(x) for x in values])})""".replace('inf', 'NULL')
    # TODO: find a way to store infinity in Postgres

    cursor.execute(query)
    connection.commit()
    cursor.close()


def update_values(connection, table: str, pk_name: str, pks, key: str, values):
    """
    Update the table on a given column with matching PK / PKs, can only update one column
    :param connection: psycopg2 connection instance
    :param str table: table name
    :param str pk_name: primary key column
    :param iterable pks: primary key values to look for
    :param str key: column name to be updated
    :param iterable values: list of new values
    """
    cursor = connection.cursor()

    for i, value in enumerate(values):
        pk = pks[i]
        query = f"""UPDATE {table}
            SET {key} = {value}
            WHERE {pk_name} = {pk};"""
        cursor.execute(query)
    connection.commit()
    cursor.close()


if __name__ == '__main__':
    conn = connect()
    df = pd.read_csv('data/car_loan_trainset.csv')
    for row in df.values.tolist():
        keys = df.columns.tolist()
        insert_row(conn, 'car_loan_sql', keys, row)
    conn.close()

