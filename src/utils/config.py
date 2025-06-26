import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GRIDSTATUS_API_KEY = os.getenv('GRIDSTATUS_API_KEY')
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///energy_data.db')
    API_HOST = os.getenv('API_HOST', 'localhost')
    API_PORT = int(os.getenv('API_PORT', 8000))
    
    @classmethod
    def validate(cls):
        """Validate required environment variables"""
        if not cls.GRIDSTATUS_API_KEY:
            raise ValueError("GRIDSTATUS_API_KEY not set")