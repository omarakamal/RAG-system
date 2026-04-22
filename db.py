import psycopg2

def get_conn():
    return psycopg2.connect(
        dbname="ragdb",
        user="postgres",
        password="postgres",
        host="localhost",
        port="5433"
    )