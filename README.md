# LLM Inference Router

Multi-model LLM router that optimizes cost and latency by intelligently routing queries to local/cloud models based on complexity analysis.

## Architecture

- **Complexity Analysis**: Analyzes query complexity to route between local/cloud models
- **Load Balancing**: Distributes requests across available model endpoints
- **Circuit Breaker**: Prevents cascade failures with automatic fallback
- **Redis Caching**: Response caching with configurable TTL
- **Monitoring**: Prometheus metrics + Grafana dashboards

## Skills Demonstrated

- **AI/ML**: Model routing, complexity analysis, LLM inference optimization
- **Infrastructure**: Kubernetes, Terraform, Ingress with rate limiting
- **Backend**: FastAPI, distributed routing, circuit breaker patterns
- **Database**: Redis caching, connection pooling
- **DevOps**: Docker, K8s deployments, CI/CD pipeline
- **SRE**: Prometheus metrics, health checks, observability

## Quick Start

```bash
# Local development
docker-compose up

# Kubernetes deployment
kubectl apply -f k8s/

# Test routing
curl -X POST localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

## Configuration

- **LOCAL_MODEL_URL**: Local model endpoint (default: ollama:11434)
- **CLOUD_MODEL_URL**: Cloud model endpoint
- **REDIS_URL**: Redis connection string
- **COMPLEXITY_THRESHOLD**: Routing threshold (0.0-1.0)

## Monitoring

- Health: `/health`
- Metrics: `/metrics`
- Grafana: Port 3000 (admin/admin)

## Infrastructure

- **Terraform**: GKE cluster, monitoring stack
- **Kubernetes**: Multi-replica deployment with HPA
- **Ingress**: Rate limiting, SSL termination
- **Monitoring**: Prometheus + Grafana + AlertManager