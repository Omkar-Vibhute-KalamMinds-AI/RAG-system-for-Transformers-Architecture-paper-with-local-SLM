# ECS Fargate task definitions

Replace `123456789012` (account), `fs-xxxxxxxx` (EFS), and IAM role names before the first register.

| File | Family | Resources | Mounts |
| --- | --- | --- | --- |
| [task-definition.api.json](task-definition.api.json) | `llmops-api` | 4 vCPU / 16 GB | EFS → `/models/llm`, `/models/embed`, `/data/lance_db`, `/app/logs` |
| [task-definition.web.json](task-definition.web.json) | `llmops-web` | 0.25 vCPU / 0.5 GB | none |

Gemma, E5, and LanceDB are **not** in the image. Put them on one EFS filesystem using those directories (same layout as Compose).

nginx in web proxies to hostname `api:8000`. On Fargate that is **not** Docker Compose DNS. Enable **ECS Service Connect** (or Cloud Map) on both services in the same namespace, with the API port name `api` so web can resolve `api`.

`ci-cd.yml` renders these files with the image tag `github.sha` and registers a new revision on each deploy.
