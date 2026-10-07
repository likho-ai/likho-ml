# likho-ml

The model service: which speech models exist and which one transcribes by default, how well each
does on a workspace's gold set, the training examples people's corrections make, and fine-tuning
runs. Owns the PostgreSQL database `likho_ml` and the `likho-models` bucket; speaks
`likho.ml.v1.MlService` (gRPC, [likho-contracts](https://github.com/likho-ai/likho-contracts)).

| | |
| --- | --- |
| gRPC | 5080 |
| Health, metrics | 4080: `/healthz`, `/readyz` (the database answers), `/metrics` (Prometheus) |
| Publishes | `likho.model.registered`, `likho.model.chosen`, `likho.evaluation.completed`, `likho.evaluation.failed` |
| Listens to | `likho.transcript.corrected` (each corrected line becomes a training example) |
| Calls | likho-transcription (`GetTranscript`, `Transcribe` for evaluations) |

## Run

Start the likho-infra stack (`.\stack.ps1 up`), then:

```powershell
uv run likho-ml
```

Settings come from `.env.development` (committed, no secrets) and `.env.development.local`
(ignored); `LIKHO_ENV` picks the environment. The tables are created or updated at start
(`MIGRATE_ON_START`).

## Develop

```powershell
uv run ruff check . ; uv run ruff format --check . ; uv run mypy ; uv run pytest
```

The integration tests run the real service against the stack; without it they are skipped, and
with `LIKHO_REQUIRE_STACK=1` (CI) they fail instead.

## How it works

This is being built in step 10 of the [roadmap](https://likho-ai.github.io/likho-docs/#/page/roadmap):
the registry and the default, training examples from corrections, the gold set and evaluations
(word and character error rates on both layers), the transcription service using the default,
the admin screen, and the fine-tuning launcher.
