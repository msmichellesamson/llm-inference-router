# LLM Inference Router

Multi-model LLM router that optimizes cost and latency by intelligently routing queries to local/cloud models based on complexity analysis.

## Architecture

- **Smart Routing**: Complexity analysis determines optimal model selection
- **Multi-Provider**: Supports local models (Ollama) and cloud APIs (OpenAI, Anthropic)
- **Cost Optimization**: Real-time cost estimation and budget controls
- **High Availability**: Circuit breakers, retries, fallbacks, and health checks
- **Observability**: Prometheus metrics, distributed tracing, structured logging
- **Caching**: Redis-backed response caching with TTL management
- **Rate Limiting**: Per-client rate limiting and quota management

## Infrastructure

### Deployment
```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Or use Docker Compose for local development
docker-compose up
```

### Terraform (GCP)
```bash
cd terraform/
terraform init
terraform apply
```

### Components
- **API Gateway**: FastAPI with async request handling
- **Router Engine**: ML-based complexity analysis for model selection
- **Cache Layer**: Redis with intelligent TTL and eviction
- **Monitoring**: Prometheus + Grafana dashboards
- **Load Balancing**: Multiple model endpoint management
- **Security**: RBAC, network policies, rate limiting

## Features

### Routing Intelligence
- Query complexity scoring (tokens, semantics, domain)
- Cost-aware routing with budget tracking
- Latency optimization based on SLA requirements
- Fallback chains for reliability

### Reliability
- Circuit breakers with configurable thresholds
- Exponential backoff retry logic
- Health checks for all model endpoints
- Performance degradation detection

### Observability
- Request/response metrics and latencies
- Model performance tracking
- Cost analytics and budget alerts
- Distributed tracing with correlation IDs
- Error tracking and categorization

### Caching Strategy
- Semantic similarity caching
- Cost-based cache prioritization
- Adaptive TTL based on query patterns
- Cache warming for popular queries

## API Endpoints

- `POST /chat/completions` - Main inference endpoint
- `POST /batch` - Batch processing with queueing
- `GET /models` - Available model information
- `GET /health` - Service health status
- `GET /metrics` - Prometheus metrics
- `POST /benchmark` - Model performance testing

## Configuration

Environment variables:
- `REDIS_URL` - Redis connection string
- `MODEL_CONFIGS` - JSON config for model endpoints
- `COST_LIMITS` - Per-client cost budgets
- `LOG_LEVEL` - Logging verbosity

## Monitoring

Key metrics:
- `llm_requests_total` - Request count by model/client
- `llm_request_duration_seconds` - Response latencies
- `llm_cost_total` - Running cost by model/client
- `llm_cache_hit_ratio` - Cache effectiveness
- `llm_circuit_breaker_state` - Circuit breaker status

## Stack

- **Languages**: Python 3.11+
- **Framework**: FastAPI with async/await
- **Cache**: Redis 7.x
- **Monitoring**: Prometheus, Grafana
- **Infrastructure**: Kubernetes, Terraform (GCP)
- **CI/CD**: GitHub Actions with automated deployments

Built for production scale with focus on cost efficiency, reliability, and observability.