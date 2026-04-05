# LLM Inference Router

Multi-model LLM router that optimizes cost and latency by intelligently routing queries to local/cloud models based on complexity analysis.

## Skills Showcased
- **AI/ML**: LLM routing, complexity analysis, model serving optimization
- **Backend**: FastAPI, async processing, distributed routing
- **Infrastructure**: Redis caching, Kubernetes deployment, monitoring
- **SRE**: Circuit breakers, retry logic, comprehensive observability
- **Database**: Redis for caching and session management
- **DevOps**: Docker containerization, Terraform IaC, CI/CD pipeline

## Architecture
```
Client → API Gateway → Complexity Analyzer → Router → [Local LLM | Cloud API]
                    ↓
                Redis Cache ← Metrics Collector ← Circuit Breaker
```

## Quick Start
```bash
# Start with docker-compose
docker-compose up -d

# Or deploy to Kubernetes
kubectl apply -f k8s/
```

## API Examples

### Basic Query Routing
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is Python?"}],
    "max_tokens": 100
  }'
```

### Force Specific Model
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Preferred-Model: gpt-4" \
  -d '{
    "messages": [{"role": "user", "content": "Complex analysis task"}],
    "temperature": 0.7
  }'
```

### Streaming Response
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

### Health Check
```bash
curl http://localhost:8000/health
# Returns: {"status": "healthy", "models": {...}}
```

### Metrics Endpoint
```bash
curl http://localhost:8000/metrics
# Prometheus format metrics
```

## Configuration

### Environment Variables
```bash
# Model Configuration
LOCAL_MODEL_URL=http://ollama:11434
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...

# Redis
REDIS_URL=redis://localhost:6379
CACHE_TTL=3600

# Routing Thresholds
COMPLEXITY_THRESHOLD=0.7
LATENCY_THRESHOLD_MS=2000
COST_WEIGHT=0.3
```

### Model Routing Logic
- **Simple queries** (complexity < 0.5) → Local model
- **Complex queries** (complexity > 0.7) → GPT-4
- **Medium queries** → GPT-3.5-turbo or Claude
- **Code/technical** → Specialized models

## Monitoring

### Key Metrics
- `llm_requests_total` - Total requests by model
- `llm_request_duration_seconds` - Response latency
- `llm_routing_decisions_total` - Routing choices
- `llm_cache_hits_total` - Cache performance
- `llm_circuit_breaker_state` - Circuit breaker status

### Alerts
- High error rate (>5%)
- Circuit breaker open
- Cache miss rate >80%
- Response time >5s

## Troubleshooting

### Common Issues

**503 Service Unavailable**
```bash
# Check circuit breaker status
curl http://localhost:8000/health
# Reset if needed
curl -X POST http://localhost:8000/admin/circuit-breaker/reset
```

**Slow Responses**
```bash
# Check model health
kubectl logs deployment/llm-router | grep "model_health"
# Monitor routing decisions
curl http://localhost:8000/metrics | grep llm_routing
```

**Cache Issues**
```bash
# Redis connection
redis-cli -h localhost ping
# Cache stats
curl http://localhost:8000/metrics | grep cache
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
# View routing decisions
kubectl logs deployment/llm-router -f | grep "routing_decision"
```

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Local development
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

## Deployment

### Terraform
```bash
cd terraform/
terraform init
terraform plan
terraform apply
```

### Kubernetes
```bash
# Apply configurations
kubectl apply -f k8s/

# Check deployment
kubectl get pods -l app=llm-router
kubectl logs deployment/llm-router
```

## Technology Stack
- **API**: FastAPI with async support
- **Models**: OpenAI, Anthropic, local Ollama
- **Cache**: Redis with TTL
- **Monitoring**: Prometheus + Grafana
- **Infrastructure**: Kubernetes, Terraform
- **Languages**: Python with type hints