import argparse
import psycopg2


# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-H", "--Host", help = "Database Host Domain/IP")
parser.add_argument("-P", "--Port", help = "Database Port")
parser.add_argument("-d", "--Dbname", help = "Database Name")
parser.add_argument("-u", "--Username", help = "Database Username")
parser.add_argument("-p", "--Password", help = "Database Password")
 
# Read arguments from command line
args = parser.parse_args()


def get_db_connection(db_config):
    try:
        db_connection = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            dbname=db_config['db_name'],
            user=db_config['username'],
            password=db_config['password']
        )

        return db_connection

    except Exception as e:
        print(e)
        return None


def init_db(args):
    if args.Host is None:
        print('-H/--Host: expected one argument')
        return False
        
    if args.Port is None:
        print('-P/--Port: expected one argument')
        return False

    if args.Dbname is None:
        print('-d/--Dbname: expected one argument')
        return False

    if args.Username is None:
        print('-u/--Username: expected one argument')
        return False

    if args.Password is None:
        print('-p/--Password: expected one argument')
        return False

    db_config = {
        'host': args.Host,
        'port': args.Port,
        'db_name': args.Dbname,
        'username': args.Username,
        'password': args.Password
    }

    # get db connection
    db_connection = get_db_connection(db_config)

    if db_connection is None:
        print('Database Connection Failed')
        return False

    # create table
    cur = db_connection.cursor()
    query = "DROP TABLE IF EXISTS sample_face_vectors; \
            CREATE TABLE sample_face_vectors ( \
                id SERIAL PRIMARY KEY, \
                created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, \
                modified TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, \
                sample_id TEXT NOT NULL, \
                name TEXT NOT NULL, \
                metadata TEXT, \
                action TEXT, \
                vector TEXT NOT NULL \
            );"
    cur.execute(query)
    db_connection.commit()

    cur.close()
    db_connection.close()
    
    return True


if __name__ == '__main__':
    # Run app in debug mode on port 6337
    res = init_db(args)
    
    if res:
        print('Database has been initialized successfully')
    else:
        print('Database initialization failed')
