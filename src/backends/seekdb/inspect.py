import os
import pymysql

conn = pymysql.connect(host=os.getenv("SEEKDB_HOST", "seekdb"),
                       port=int(os.getenv("SEEKDB_PORT", "2881")),
                       user=os.getenv("SEEKDB_USER", "root"),
                       password=os.environ["SEEKDB_PASSWORD"], autocommit=True)
with conn.cursor() as cur:
    for pattern in ("%ef%", "%vector%", "%hnsw%"):
        cur.execute("SHOW VARIABLES LIKE %s", (pattern,))
        print(pattern, cur.fetchall())
