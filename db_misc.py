import sqlite3
import io

# DB Setup
db = sqlite3.connect('matches.db')
cursor = db.cursor()
statement = 'SELECT COUNT(AN) FROM Matches WHERE WhiteElo BETWEEN 0 AND 1200' # Remove 'LIMIT _' during actual training runs
cursor.execute(statement)
output = cursor.fetchall()
print(output)