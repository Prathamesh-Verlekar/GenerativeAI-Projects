from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config.logging_config import logger

SQLALCHEMY_DATABASE_URL = "sqlite:///./chat_history.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        logger.info("New Database Session Created")
        yield db
    except Exception as e:
        logger.error(f"Database Connection Error: {str(e)}", exc_info=True)
    finally:
        logger.info("Closing Database Session")
        db.close()
