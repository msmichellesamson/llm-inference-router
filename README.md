# LLM Inference Router

Multi-model LLM router that optimizes cost and latency by intelligently routing queries to local/cloud models based on complexity analysis.

## Features

- **Intelligent Routing**: Analyzes query complexity to route to optimal model
- **Cost Optimization**: Balances performance vs cost across local/cloud models
- **Redis Caching**: Caches responses to reduce latency and costs
- **Health Monitoring**: Tracks model performance and availability
- **Adaptive Load Balancing**: Routes requests to healthiest endpoints
- **Prometheus Metrics**: Comprehensive observability and alerting

## Architecture

```
Client → Router → Load Balancer → [Local Models, OpenAI, Claude]
           ↓
        Redis Cache
           ↓
      Metrics/Alerts
```

## Quick Start

```bash
# Start with Docker Compose
docker-compose up -d

# Or run locally
pip install -r requirements.txt
python src/main.py
```

## API Endpoints

- `POST /v1/chat/completions` - Route LLM requests
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

## Configuration

Set environment variables:
- `REDIS_URL` - Redis connection string
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `LOCAL_MODEL_URLS` - Comma-separated local model endpoints

## Deployment

```bash
# Deploy to GCP with Terraform
cd terraform/
terraform init
terraform apply
```

## Monitoring

- Grafana dashboards for request metrics
- Alertmanager for SLA violations
- Health checks for all model endpoints

## Skills Demonstrated

- **AI/ML**: Multi-model routing, complexity analysis
- **Backend**: FastAPI, async processing, gRPC
- **Database**: Redis caching, query optimization
- **Infrastructure**: Terraform, GCP, Kubernetes
- **SRE**: Prometheus monitoring, health checks
- **DevOps**: Docker, CI/CD, automated deployment