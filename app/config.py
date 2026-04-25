from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_path: str = "./models/best_model.pth"
    model_name: str = "tf_efficientnetv2_s"
    model_version: str = "1.0.0"
    device: str = "cpu"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)
    
    @property
    def resolved_model_path(self) -> Path:
        """Resolve model path for both Docker and local environments"""
        path = Path(self.model_path)
        if path.is_absolute():
            return path
        # For relative paths, check if models exist relative to this file
        app_dir = Path(__file__).parent.parent
        resolved = app_dir / self.model_path
        if resolved.exists():
            return resolved
        # Fallback to the configured path as-is
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()
