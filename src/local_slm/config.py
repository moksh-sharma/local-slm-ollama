from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: Path = Path("data")
    results_dir: Path = Path("data/results")

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    mock_ollama: bool = False

    benchmark_models: str = "llama3.2:1b,llama3.2:3b,qwen2.5:3b"
    benchmark_warmup_runs: int = 1
    benchmark_measured_runs: int = 3
    benchmark_max_tokens: int = 256

    @property
    def model_list(self) -> list[str]:
        return [m.strip() for m in self.benchmark_models.split(",") if m.strip()]


settings = Settings()
