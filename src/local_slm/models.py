from pydantic import BaseModel, Field


class GenerationResult(BaseModel):
    model: str
    prompt: str
    response: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    time_to_first_token_ms: float | None = None
    total_latency_ms: float = 0.0
    tokens_per_second: float = 0.0


class PromptBenchmark(BaseModel):
    prompt_id: str
    category: str
    runs: list[GenerationResult]
    avg_ttft_ms: float | None = None
    avg_latency_ms: float = 0.0
    avg_tokens_per_second: float = 0.0
    avg_completion_tokens: float = 0.0


class ModelBenchmark(BaseModel):
    model: str
    hardware_note: str = ""
    mock: bool = False
    prompts: list[PromptBenchmark] = Field(default_factory=list)
    avg_ttft_ms: float | None = None
    avg_latency_ms: float = 0.0
    avg_tokens_per_second: float = 0.0


class QualityTaskResult(BaseModel):
    task_id: str
    category: str
    passed: bool
    response: str
    details: str = ""


class ModelQualityReport(BaseModel):
    model: str
    tasks: list[QualityTaskResult] = Field(default_factory=list)
    score: float = 0.0
    passed: int = 0
    total: int = 0


class ComparisonReport(BaseModel):
    hardware_note: str = ""
    mock: bool = False
    benchmarks: list[ModelBenchmark] = Field(default_factory=list)
    quality: list[ModelQualityReport] = Field(default_factory=list)
    generated_at: str = ""


class ChatMessage(BaseModel):
    role: str
    content: str
