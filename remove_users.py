from app import app  # Adjust this import according to your application structure
from extensions import db
from models import User

with app.app_context():  # Use the correct method to get the app context
    try:
        num_deleted = User.query.delete()
        db.session.commit()
        print(f"Deleted {num_deleted} users from the database.")
    except Exception as e:
        db.session.rollback()
        print(f"An error occurred: {e}")
