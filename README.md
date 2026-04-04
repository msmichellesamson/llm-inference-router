# LLM Inference Router

Multi-model LLM router that optimizes cost and latency by intelligently routing queries to local/cloud models based on complexity analysis.

## Architecture

- **Smart Routing**: Complexity analysis determines local vs cloud model routing
- **Circuit Breaker**: Fault tolerance with automatic fallback
- **Redis Caching**: Response caching and metrics storage
- **Load Balancing**: Distribute requests across model instances
- **Observability**: Prometheus metrics, health checks, alerts

## Skills Demonstrated

- **AI/ML**: Multi-model inference, complexity analysis, model serving
- **Infrastructure**: Terraform, GCP, Kubernetes, Redis
- **Backend**: FastAPI, gRPC, distributed routing
- **DevOps**: Docker, K8s manifests, CI/CD
- **SRE**: Circuit breakers, monitoring, alerting
- **Database**: Redis caching, query optimization

## Infrastructure

```bash
# Deploy infrastructure
cd terraform
terraform init
terraform plan
terraform apply

# Deploy to Kubernetes
kubectl apply -f k8s/
```

## Local Development

```bash
# Start services
docker-compose up

# Run tests
pytest tests/

# API available at http://localhost:8000
```

## API Endpoints

- `POST /route` - Route query to optimal model
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

## Configuration

- Model thresholds via environment variables
- Redis connection and caching settings
- Circuit breaker parameters
- Prometheus monitoring configuration