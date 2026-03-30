# LLM Inference Router

Multi-model LLM router that optimizes cost and latency by intelligently routing queries to local/cloud models based on complexity analysis.

## Architecture

- **Smart Routing**: Routes queries based on complexity analysis
- **Multi-Provider**: OpenAI, Anthropic, local models via Ollama
- **Caching**: Redis-based response caching
- **Monitoring**: Prometheus metrics + Grafana dashboards
- **Auto-scaling**: Kubernetes HPA with CPU, memory, and request-based scaling

## Skills Demonstrated

- **AI/ML**: LLM inference, complexity analysis, model serving
- **Infrastructure**: Kubernetes, Terraform, auto-scaling
- **SRE**: Circuit breakers, monitoring, reliability patterns
- **Backend**: FastAPI, async processing, distributed routing
- **Database**: Redis caching, query optimization
- **DevOps**: Docker, K8s manifests, CI/CD

## Quick Start

```bash
# Local development
docker-compose up -d

# Kubernetes deployment
kubectl apply -f k8s/

# Infrastructure
cd terraform && terraform apply
```

## Endpoints

- `POST /route` - Route LLM queries
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

## Auto-scaling

HPA scales pods (2-10) based on:
- CPU utilization (>70%)
- Memory utilization (>80%) 
- HTTP requests per second (>100 req/s)

## Monitoring

- Request latency and throughput
- Model performance metrics
- Cache hit rates
- Circuit breaker states