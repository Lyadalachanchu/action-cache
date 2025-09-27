# Docker Setup Instructions

## Install Docker Desktop

### Windows:
1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop/
2. Run the installer
3. Restart your computer
4. Open Docker Desktop and wait for it to start

### Alternative: Use Weaviate Cloud

If you don't want to install Docker, you can use Weaviate Cloud instead:

1. Go to https://console.weaviate.cloud/
2. Create a free account
3. Create a new cluster
4. Get your cluster URL and API key
5. Update the memory system to use the cloud URL

## Test Docker Installation

After installing Docker Desktop:

```bash
docker --version
docker compose --version
```

## Run Weaviate Setup

Once Docker is installed:

```bash
python setup_weaviate.py
```

## Alternative: Manual Docker Compose

If the setup script doesn't work, you can run manually:

```bash
docker compose up -d
```

## Check if Weaviate is Running

```bash
docker ps
```

You should see a container named `action-cache-weaviate-1` running.

