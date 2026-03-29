# LLM Inference Router

Multi-model LLM router that optimizes cost and latency by intelligently routing queries to local/cloud models based on complexity analysis.

## Features
- **Intelligent Routing**: Query complexity analysis routes to optimal model
- **Cost Optimization**: Prefer local models for simple queries
- **Circuit Breaker**: Automatic failover when models are unhealthy
- **Load Balancing**: Distribute load across model replicas
- **Redis Caching**: Cache responses for improved latency
- **Prometheus Metrics**: Comprehensive observability

## Architecture
```
Query → Complexity Analysis → Router → [Local Model | Cloud API]
                                    ↓
                              Redis Cache ← Metrics
```

## Skills Demonstrated
- **AI/ML**: Model serving, inference optimization, query analysis
- **Backend**: FastAPI, async processing, circuit breakers
- **Infrastructure**: Kubernetes, Terraform, monitoring
- **Database**: Redis caching, query optimization
- **SRE**: Prometheus metrics, health checks, observability
- **DevOps**: Docker, K8s manifests, CI/CD

## Quick Start
```bash
# Local development
docker-compose up

# Kubernetes deployment
kubectl apply -f k8s/

# Infrastructure
cd terraform && terraform apply
```

## API Endpoints
- `POST /inference` - Route LLM queries
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

## Monitoring
Prometheus ServiceMonitor automatically scrapes:
- Request latency and error rates
- Model performance metrics
- Circuit breaker states
- Cache hit rates

## Configuration
Set via environment variables:
- `REDIS_URL`: Redis connection string
- `LOCAL_MODEL_URL`: Local model endpoint
- `CLOUD_API_KEY`: Cloud provider API key