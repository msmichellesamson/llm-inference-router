# LLM Inference Router

Multi-model LLM router that optimizes cost and latency by intelligently routing queries to local/cloud models based on complexity analysis.

## Skills Demonstrated
- **AI/ML**: LLM routing, complexity analysis, model selection
- **Infrastructure**: Kubernetes, Terraform, GCP deployment
- **SRE**: Circuit breakers, health checks, observability, chaos engineering
- **Backend**: FastAPI microservice, gRPC, distributed routing
- **Database**: Redis caching, query optimization
- **DevOps**: CI/CD, containerization, GitOps
- **Data**: Request preprocessing, metrics pipeline

## Architecture

```
Client → API Gateway → Router → [Local Model | Cloud API]
                    ↓
               Redis Cache ← Metrics
```

## Features

### Core Routing
- **Intelligent Routing**: Routes based on query complexity and model availability
- **Load Balancing**: Distributes load across available models
- **Circuit Breaker**: Prevents cascade failures
- **Health Monitoring**: Continuous model health checks

### Performance
- **Redis Caching**: Sub-millisecond response caching
- **Batch Processing**: Efficient batch inference
- **Connection Pooling**: Optimized HTTP/gRPC connections
- **Timeout Handling**: Graceful timeout with fallbacks

### Reliability
- **Retry Logic**: Exponential backoff with jitter
- **Error Tracking**: Comprehensive error classification
- **Drift Detection**: Model performance monitoring
- **Graceful Degradation**: Fallback to simpler models

### Observability
- **Prometheus Metrics**: Request latency, error rates, model usage
- **OpenTelemetry Tracing**: Distributed request tracing
- **Health Endpoints**: Kubernetes-ready health checks
- **Performance Benchmarking**: Automated load testing

## Infrastructure

### Kubernetes
- **High Availability**: Pod disruption budgets, HPA scaling
- **Security**: Network policies, resource limits
- **Monitoring**: ServiceMonitor for Prometheus scraping
- **Persistence**: Redis with persistent volumes

### Terraform
- **GCP Resources**: GKE cluster, Redis instance, monitoring
- **Alerting**: Prometheus alertmanager rules
- **Networking**: VPC, firewalls, load balancers

## Quick Start

```bash
# Deploy infrastructure
cd terraform && terraform apply

# Deploy application
kubectl apply -f k8s/

# Port forward for testing
kubectl port-forward svc/llm-inference-router 8000:80
```

## API Usage

```bash
# Simple inference
curl -X POST http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is machine learning?", "max_tokens": 100}'

# Batch inference
curl -X POST http://localhost:8000/v1/batch \
  -H "Content-Type: application/json" \
  -d '{"requests": [{"prompt": "Hello"}, {"prompt": "World"}]}'
```

## Monitoring

- **Metrics**: `/metrics` endpoint for Prometheus
- **Health**: `/health` and `/ready` endpoints
- **Traces**: OpenTelemetry to configured backend

## Configuration

Environment variables:
- `REDIS_URL`: Redis connection string
- `MODEL_ENDPOINTS`: JSON array of model configurations
- `ENABLE_TRACING`: Enable OpenTelemetry tracing
- `LOG_LEVEL`: Logging verbosity

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run locally
python -m src.main
```

## Production Deployment

1. **Infrastructure**: Deploy GCP resources with Terraform
2. **Application**: Deploy to GKE with Kubernetes manifests
3. **Monitoring**: Configure Prometheus and Grafana
4. **Alerting**: Set up PagerDuty/Slack notifications

Built with Python, FastAPI, Redis, Kubernetes, and Terraform.