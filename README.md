# LLM Inference Router

Multi-model LLM router that optimizes cost and latency by intelligently routing queries to local/cloud models based on complexity analysis.

## Architecture

```
Query → Preprocessing → Complexity Analysis → Routing Decision → Model Inference
  ↓         ↓              ↓                    ↓              ↓
Cache    Rate Limit    Cost/Latency         Load Balance   Response
```

## Core Features

- **Smart Routing**: Routes queries based on complexity, cost, and latency requirements
- **Multi-Model Support**: OpenAI, Anthropic, local models via Ollama
- **Performance Optimization**: Caching, circuit breakers, load balancing
- **Observability**: Prometheus metrics, distributed tracing, health checks
- **Production Ready**: Rate limiting, retries, graceful degradation

## Quick Start

### Local Development
```bash
# Start dependencies
docker-compose up -d redis prometheus grafana

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=your_key
export REDIS_URL=redis://localhost:6379

# Run the service
python src/main.py
```

### Kubernetes Deployment
```bash
# Apply configurations
kubectl apply -f k8s/

# Check status
kubectl get pods -l app=llm-router
```

## API Usage

See [API Examples](docs/api-examples.md) for detailed usage examples.

### Basic Request
```python
import requests

response = requests.post('http://localhost:8000/route', json={
    'query': 'Explain quantum computing',
    'max_tokens': 200,
    'temperature': 0.7
})

print(response.json())
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379` | Redis connection string |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `LOCAL_MODEL_URL` | `http://localhost:11434` | Ollama API endpoint |
| `MAX_CONCURRENT_REQUESTS` | `100` | Request concurrency limit |
| `CACHE_TTL_SECONDS` | `3600` | Response cache TTL |

## Monitoring

- **Metrics**: Available at `/metrics` (Prometheus format)
- **Health**: Available at `/health`
- **Tracing**: Jaeger-compatible spans
- **Logs**: Structured JSON logging

Key metrics:
- `llm_requests_total`: Total requests by model/status
- `llm_request_duration_seconds`: Request latency histogram
- `llm_cost_total`: Total API costs by model
- `llm_cache_hits_total`: Cache hit/miss rates

## Model Routing Logic

1. **Complexity Analysis**: Query length, keywords, semantic complexity
2. **Cost Estimation**: Per-token costs for different models
3. **Latency Requirements**: User preferences vs model capabilities
4. **Health Checking**: Circuit breakers for unhealthy models
5. **Load Balancing**: Distribute load across healthy instances

## Development

### Running Tests
```bash
pytest tests/ -v --cov=src/
```

### Infrastructure
```bash
# Deploy infrastructure
cd terraform
terraform init
terraform apply
```

## Technologies

- **Language**: Python 3.11+
- **Framework**: FastAPI
- **Cache**: Redis
- **Monitoring**: Prometheus, Grafana, Jaeger
- **Infrastructure**: Terraform, Kubernetes
- **Testing**: pytest, pytest-cov

## License

MIT