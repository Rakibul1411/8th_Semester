import sqlite3
import hashlib

# BUG 1: Mutable default arguments (Logic/State Flaw)
def add_user(username, password, user_list=[]):
    user_list.append(username)
    
    # BUG 2: Hardcoded credentials (Security)
    admin_key = "super_secret_admin_key_123"
    
    # BUG 3: Weak hashing algorithm (Security)
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # BUG 4: SQL Injection vulnerability (Security)
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{hashed_pw}'"
    cursor.execute(query)
    
    return user_list

def read_config(filepath):
# BUG 5: Resource leak (Reliability Flaw)
    f = open(filepath, 'r')
    data = f.read()
    return data

def process_data(user_input):
    try:
        # BUG 6: Dangerous system command (Security)
        exec(user_input)
    except:
        # BUG 7: Bare except (Code Quality/Debugging Flaw)
        print("Something went wrong.")
