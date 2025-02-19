# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker, declarative_base
# from config import DATABASE_URL

# # Create Engine
# engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# # ORM Base Class
# Base = declarative_base()

# # DB Session Factory
# SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# # Dependency Injection
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config.logging_config import logger

DATABASE_URL = "sqlite:///./chat_history.db"  # Change this to PostgreSQL if needed

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency to get DB Session
def get_db():
    db = SessionLocal()
    try:
        logger.info("🔗 New Database Session Created")
        yield db
    except Exception as e:
        logger.error(f"Database Connection Error: {str(e)}", exc_info=True)
    finally:
        logger.info("Closing Database Session")
        db.close()

