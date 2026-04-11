# LLM Inference Router

Multi-model LLM router that optimizes cost and latency by intelligently routing queries to local/cloud models based on complexity analysis.

## Features

- **Intelligent Routing**: Routes queries based on complexity analysis
- **Cost Optimization**: Balances cost vs performance automatically
- **Multi-Model Support**: Local (Llama) and cloud (OpenAI) models
- **Circuit Breakers**: Prevents cascade failures
- **Drift Detection**: Monitors model performance degradation
- **Cost Estimation**: Predict query costs before execution
- **Observability**: Comprehensive metrics and tracing
- **Auto-scaling**: Kubernetes HPA based on queue depth

## API Endpoints

### Inference
- `POST /inference/query` - Route and execute LLM queries
- `POST /inference/batch` - Batch processing with queuing

### Cost Management
- `POST /cost/estimate` - Estimate query cost and get model recommendations

### Monitoring
- `GET /health` - Health checks with dependency status
- `GET /metrics` - Prometheus metrics

## Architecture

```
Query → Preprocessing → Complexity Analysis → Router → Model Selection
  ↓                                                         ↓
Cost Estimation ← Circuit Breaker ← Load Balancer ← Response
```

## Quick Start

```bash
# Start with Docker Compose
docker-compose up -d

# Deploy to Kubernetes
kubectl apply -f k8s/

# Infrastructure (Terraform)
cd terraform && terraform apply
```

## Configuration

Set environment variables:
```bash
OPENAI_API_KEY=your-key
REDIS_URL=redis://localhost:6379
PROMETHEUS_ENDPOINT=http://localhost:9090
```

## Tech Stack

**AI/ML**: LLM routing, complexity analysis, model serving  
**Infrastructure**: Kubernetes, Terraform, GCP/Redis  
**Backend**: FastAPI, gRPC, async Python  
**Database**: Redis caching, query optimization  
**SRE**: Prometheus metrics, circuit breakers, health checks  
**DevOps**: Docker, K8s HPA, CI/CD pipeline  
**Data**: Batch processing, streaming inference

## Monitoring

Dashboards available at:
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- API Docs: http://localhost:8000/docs

## Testing

```bash
pytest tests/ -v
```