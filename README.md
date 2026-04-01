# LLM Inference Router

Intelligent multi-model LLM router that optimizes cost and latency by routing queries based on complexity analysis.

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Client    │───▶│    Router    │───▶│   Models    │
└─────────────┘    └──────────────┘    └─────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │ Complexity   │
                   │ Analyzer     │
                   └──────────────┘
```

## Quick Start

```bash
# Start with Docker Compose
docker-compose up -d

# Test the API
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world"}'
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379` | Cache backend |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `LOCAL_MODEL_URL` | `http://localhost:11434` | Local Ollama endpoint |

## Monitoring

- Health: `GET /health`
- Metrics: `GET /metrics`
- Grafana: `http://localhost:3000`

## Troubleshooting

### High Latency
1. Check model availability: `kubectl get pods -l app=llm-router`
2. Verify Redis connection: `redis-cli ping`
3. Review circuit breaker status in metrics

### Routing Failures
1. Check complexity analyzer logs: `kubectl logs -l component=analyzer`
2. Verify model endpoints are responding
3. Check rate limits and quotas

## API Documentation

See `docs/api-spec.yaml` for complete OpenAPI specification.

## Skills Demonstrated

- **AI/ML**: Multi-model routing, complexity analysis
- **Infrastructure**: Kubernetes, Terraform, Redis
- **SRE**: Circuit breakers, monitoring, alerting
- **Backend**: FastAPI, async processing, load balancing
- **Database**: Redis caching, query optimization
- **DevOps**: Docker, K8s deployment, CI/CD