import pymongo
import bcrypt
import uuid
import datetime # Import datetime

MONGO_URI = "mongodb+srv://shareenpan2:Fgouter55@cluster0.s3dpu.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

def get_connection():
    client = pymongo.MongoClient(MONGO_URI)
    db = client.get_database("summarization_paraphrasing_db")  # You might want to change the database name
    return db

def create_user(email, username, password, role='viewer'): # Removed 'name'
    db = get_connection()
    users_collection = db.users
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    user_id = str(uuid.uuid4())

    try:
        user_data = {
            "_id": user_id,
            "username": username,
            "email": email,
            "password": hashed_pw.decode('utf-8'),
            "role": role,
            "createdAt": datetime.datetime.now() # Set to current datetime
        }
        users_collection.insert_one(user_data)
        return True
    except pymongo.errors.DuplicateKeyError as err:
        print(f"Error: {err}")
        return False
    finally:
        db.client.close()

def validate_user(username, password):
    db = get_connection()
    users_collection = db.users
    user = users_collection.find_one({"username": username})
    db.client.close()

    if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
        return user
    return None
