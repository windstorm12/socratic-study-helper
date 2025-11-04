# models.py
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func, UniqueConstraint
from sqlalchemy.orm import relationship

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    role = db.Column(db.String(20), nullable=False, default="student")  # student|teacher
    full_name = db.Column(db.String(120))
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, server_default=func.now())
    taught_classrooms = relationship("Classroom", back_populates="teacher", foreign_keys="Classroom.teacher_id")

class Classroom(db.Model):
    __tablename__ = "classrooms"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    join_code = db.Column(db.String(16), unique=True, nullable=False, index=True)
    teacher_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    created_at = db.Column(db.DateTime, server_default=func.now())

    teacher = relationship("User", back_populates="taught_classrooms")
    enrollments = relationship("Enrollment", back_populates="classroom", cascade="all, delete-orphan")

class Enrollment(db.Model):
    __tablename__ = "enrollments"
    id = db.Column(db.Integer, primary_key=True)
    classroom_id = db.Column(db.Integer, db.ForeignKey("classrooms.id"), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    created_at = db.Column(db.DateTime, server_default=func.now())

    __table_args__ = (UniqueConstraint('classroom_id', 'student_id', name='uix_classroom_student'),)
    classroom = relationship("Classroom", back_populates="enrollments")
    student = relationship("User")

class Message(db.Model):
    __tablename__ = "messages"
    id = db.Column(db.Integer, primary_key=True)
    classroom_id = db.Column(db.Integer, db.ForeignKey("classrooms.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    role = db.Column(db.String(10), nullable=False)  # 'user' or 'ai'
    subject = db.Column(db.String(50))
    text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, server_default=func.now(), index=True)