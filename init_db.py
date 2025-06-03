from app import app, db  # Make sure `app` and `db` are accessible from app.py

with app.app_context():
    db.create_all()
    print("✅ Database initialized!")
