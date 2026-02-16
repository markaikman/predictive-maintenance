from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    database_url: str
    model_path: str = "/artifacts/models/model.pkl"
    production_model_path: str = "/artifacts/models/production.pkl"
    model_version: str = "dev"
    admin_api_key: str = os.getenv("ADMIN_API_KEY", "")


settings = Settings()
