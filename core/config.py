
from decouple import Config, RepositoryEnv
from pydantic_settings import BaseSettings
import os
from utils.logger import logger

config:str = Config(RepositoryEnv('.env'))

class Settings(BaseSettings): 
    ENV: str = os.getenv('ENV', 'PROD').upper()
    print(f"[CURRENT ENV]: {ENV}")
    
    # AI / LLM
    OPENAI_API_KEY: str = config("OPENAI_API_KEY", cast=str)
    OPENAI_MODEL: str = "gpt-5-mini" # or gpt-5
    
    GROQ_API_KEY: str = config("GROQ_API_KEY", cast=str)
    
    class Config:
        case_sensitive = True

settings = Settings()