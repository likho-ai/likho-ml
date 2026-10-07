"""Configuration, read from environment variables and the .env files of the current environment.

LIKHO_ENV (development, staging or production; default development) picks the files. They are
read in this order, each one overriding the one before, and a real environment variable wins
over all of them:

    .env  .env.local  .env.<LIKHO_ENV>  .env.<LIKHO_ENV>.local

The .env.<LIKHO_ENV> files are committed and hold no secrets; the .local files are ignored by
git and hold the secrets of that environment on this machine.
"""

import os

from pydantic_settings import BaseSettings, SettingsConfigDict

LIKHO_ENV = os.environ.get("LIKHO_ENV", "development")
ENV_FILES = (".env", ".env.local", f".env.{LIKHO_ENV}", f".env.{LIKHO_ENV}.local")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILES, extra="ignore")

    likho_env: str = LIKHO_ENV
    log_level: str = "INFO"

    grpc_port: int = 5080
    http_port: int = 4080  # /healthz, /readyz and /metrics

    # Defaults match the likho-infra local stack.
    database_url: str = "postgresql+asyncpg://likho_ml:likho_ml@localhost:5433/likho_ml"
    nats_url: str = "nats://localhost:4222"
    # How long the start keeps trying to reach NATS before going on without it.
    nats_connect_timeout_seconds: float = 120.0
    # Metrics are always at GET /metrics (Prometheus text); set this to also push them (OTLP/HTTP, e.g. http://localhost:4318).
    otel_exporter_otlp_endpoint: str = ""

    # Create or update the tables when the service starts.
    migrate_on_start: bool = True

    # The engine's published sizes, registered at start if missing (comma-separated registry ids).
    # The first becomes the default while there is none: what likho-transcription used before.
    seed_models: str = "faster-whisper/turbo,faster-whisper/large-v3,faster-whisper/medium,faster-whisper/small"

    # likho-transcription: the lines of corrected transcripts (audio spans and both layers).
    transcription_grpc_addr: str = "localhost:5020"
    rpc_timeout_seconds: float = 10.0

    # Keep people's corrections as training examples (likho.transcript.corrected, stream LIKHO_KEEP).
    consumers_enabled: bool = True
    # Instances with the same name share the corrections; "all" also takes every correction made
    # before this service existed (the stream keeps them for ever).
    corrected_durable: str = "likho-ml-corrected"
    corrected_start: str = "all"

    # The models bucket: datasets for fine-tuning go to s3://<bucket>/datasets/<id>/.
    s3_endpoint: str = "http://localhost:9000"
    s3_region: str = "us-east-1"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_bucket_models: str = "likho-models"

    # Where a training run is handed to: a GPU machine's agent or a cloud job's front, which gets
    # the run as JSON (POST) and reports back with MlService.ReportTrainingRun. Empty: a run is
    # recorded and waits as pending until someone starts it by hand.
    launcher_url: str = ""
    # The address the job reports back to (likho-ml's gRPC as the job can reach it).
    report_address: str = "likho-ml:5080"

    @property
    def seed_model_ids(self) -> list[str]:
        return [item.strip() for item in self.seed_models.split(",") if item.strip()]
