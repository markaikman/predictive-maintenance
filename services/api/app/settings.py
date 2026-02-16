from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    model_path: str = "/artifacts/models/model.pkl"
    model_version: str = "dev"


settings = Settings()
