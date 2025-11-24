from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy()
class User(db.Model):
    __tablename__ = 'rankifylogin'
    __table_args__ = {'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    verified = db.Column(db.Boolean, default=True)
    verification_code = db.Column(db.String(255), nullable=True)
    role = db.Column('Role', db.String(50), nullable=False, default='user')
    folders = db.Column('Folders', db.Text, nullable=True)
    
