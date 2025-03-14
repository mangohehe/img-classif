# Ray Serve LLM Service on GCP

This repository deploys an **OpenAI-compatible language model (LLM) service** using **Ray Serve**, **vLLM**, and **FastAPI**. The service runs on a **Ray cluster** configured on **Google Cloud Platform (GCP)** and integrates **Prometheus** and **Grafana** for real-time monitoring. A **command-line interface (CLI)** is also provided for managing the service.

---

## 🚀 Overview

- **Ray Cluster on GCP:**  
  Deploy and manage the Ray cluster using **automated provisioning scripts**.

- **LLM Service:**  
  Uses **vLLM** as the backend for **asynchronous chat** and **text completions**, exposed via **FastAPI**.

- **Monitoring with Prometheus & Grafana:**  
  - **Prometheus** collects real-time system and application metrics.
  - **Grafana** provides a **dashboard UI** for visualizing cluster and service metrics.

- **Automated Environment Setup:**  
  - **Conda environment creation** and **dependency installation** are fully automated in setup scripts.
  - No manual setup is required when launching the cluster.

- **Command-Line Interface (CLI):**  
  Provides commands to **start the service** and **send queries**.

---

## 📁 Project Structure

```
.
├── __init__.py
├── pyproject.toml               # Project build configuration
├── ray_config                   # Ray cluster setup scripts and YAML configurations
│   ├── gpc-ray-cluster-cpus-only.yaml  # Main Ray cluster configuration file
│   ├── head_setup.sh
│   ├── head_start_ray.sh
│   ├── setup_commands.sh
│   ├── setup_ray_env.sh
│   └── setup_vllm.sh
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation script
├── src                          # Main source code for the LLM service
│   ├── __init__.py
│   ├── cli.py                   # CLI for starting and querying the service
│   ├── config.py                # Configuration (MODEL_NAME, TENSOR_PARALLEL_SIZE, etc.)
│   ├── llm.py                   # Ray Serve deployment & API implementations
│   ├── main.py                  # Main entry point for the service
│   ├── query.py                 # Utility to query the deployed LLM service
│   └── utils.py                 # Utility functions (logging setup, etc.)
├── tests                        # Test suite
│   ├── test_model_support.py
│   └── test_vllm.py
└── vllm_app.egg-info            # Packaging metadata
```

---

## ⚙️ Starting the Ray Cluster on GCP

### 🚀 **Launch the Cluster**
To start the **Ray cluster** on **Google Cloud Platform (GCP)**, run:

```bash
ray up ray_config/gpc-ray-cluster-cpus-only.yaml -y
```

📌 This command:
- Automatically provisions the cluster based on the provided YAML configuration.
- Creates & sets up Conda environments for Ray, vLLM, and dependencies.
- Executes setup scripts for configuring the environment.

### 🔍 Verify Cluster Status
Once the cluster is running, verify its status with:

```bash
ray status
```

### 🛑 Stop the Cluster
To shut down the cluster:

```bash
ray down ray_config/gpc-ray-cluster-cpus-only.yaml -y
```

---

## 📊 Monitoring with Grafana & Prometheus

### ✅ Prometheus
- Used for collecting real-time system and application metrics.
- Ray provides built-in Prometheus metric exporters.
- Runs on port **9090**.

Check Prometheus metrics with:

```bash
curl http://localhost:9090/api/v1/query?query=up
```

### ✅ Grafana
- Provides a dashboard UI for monitoring Ray cluster and application metrics.
- Runs on port **3000**.
- Uses Prometheus as its data source.

Start Grafana:

```bash
grafana-server --homepath /usr/share/grafana --config /tmp/ray/session_latest/metrics/grafana/grafana.ini
```

Access the Grafana dashboard at:

[http://localhost:3000/d/rayDefaultDashboard/?var-datasource=Prometheus](http://localhost:3000/d/rayDefaultDashboard/?var-datasource=Prometheus)

---

## 🚀 Running the Service

### 🏁 Deploying the Service
The main entry point is `src/main.py`. When executed, it:

1. Initializes Ray:

```python
ray.init(address="auto", num_cpus=8)
```

2. Builds the Ray Serve application using parameters from `src/config.py` and `src/llm.py`.
3. Deploys the application using `serve.run(app)`.

To manually start the service on the head node, run:

```bash
python src/main.py
```

### 🔗 Service Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat completions API |
| `POST /v1/completions` | Text completions API |

### 📝 Example Query

Test the text completions endpoint with:

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer NOT_A_REAL_KEY" \
  -d '{
  "model": "gpt2",
  "prompt": "Question: What is the capital of France?\nAnswer: The capital of France is",
  "max_tokens": 100
}'
```

---

## 🖥️ CLI Usage

A CLI is provided in `src/cli.py` with the following commands:

### ▶️ Start the Service

```bash
python -m src.cli start
```
🔹 Builds and deploys the Ray Serve application.

### ❓ Query the Service

```bash
python -m src.cli query
```
🔹 Sends a test query to the deployed service.

---

## 🧪 Testing

Run the test suite with:

```bash
pytest tests/
```

✅ Ensure all tests pass before deploying to production.

---

## 🔧 Troubleshooting

### 🔹 Cluster Connectivity
Verify that the head node’s external IP is accessible on required ports:

| Service | Port |
|---------|------|
| Grafana | 3000 |
| Prometheus | 9090 |
| Ray | 6379, 8265, 8076 |

📌 Adjust GCP firewall rules if necessary.

### 🔹 Check Logs
Monitor errors by checking `/tmp/init.log`:

```bash
cat /tmp/init.log
```

---

## 🤝 Contributing

🚀 We welcome contributions!
To contribute:
1. Fork the repository
2. Create a new branch
3. Submit a pull request (PR)

🔹 Issues & suggestions? Feel free to open an issue!

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🙌 Acknowledgements

This project utilizes:
- Ray
- Grafana
- Prometheus
- vLLM
- Miniforge

💡 Inspired by modern LLM deployment architectures!
