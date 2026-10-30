"""Optional helper: print all users (the 'retrieve user records' part of Read)."""
import database as db

db.init_db()
try:
    users = db.get_all_users()
except db.DatabaseError as e:
    print(e)
else:
    if not users:
        print("No users yet.")
    for u in users:
        print(f"{u['id']:>3} | {u['full_name']:<20} | {u['email']:<30} | {u['mobile']:<12} | {u['created_at']}")
