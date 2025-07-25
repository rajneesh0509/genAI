# client_example.py
import requests
import json
from typing import Optional

class LogAnalyzerClient:
    """Client class for interacting with the iLogAnalyze API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> dict:
        """Check if the API is healthy"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def upload_log_file(self, file_path: str) -> dict:
        """Upload a log file for analysis"""
        with open(file_path, 'rb') as file:
            files = {'file': (file_path, file, 'text/plain')}
            response = self.session.post(f"{self.base_url}/upload", files=files)
            response.raise_for_status()
            return response.json()
    
    def analyze_issue(self, question: str, session_id: Optional[str] = None) -> dict:
        """Analyze an issue using the ReAct agent"""
        payload = {
            "question": question,
            "session_id": session_id
        }
        response = self.session.post(
            f"{self.base_url}/analyze", 
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        return response.json()
    
    def get_session_info(self, session_id: str) -> dict:
        """Get information about a session"""
        response = self.session.get(f"{self.base_url}/sessions/{session_id}")
        response.raise_for_status()
        return response.json()
    
    def delete_session(self, session_id: str) -> dict:
        """Delete a session and its data"""
        response = self.session.delete(f"{self.base_url}/sessions/{session_id}")
        response.raise_for_status()
        return response.json()

# Example usage
def main():
    # Initialize client
    client = LogAnalyzerClient("https://your-app.azurewebsites.net")
    
    try:
        # 1. Health check
        print("🔍 Checking API health...")
        health = client.health_check()
        print(f"✅ API Status: {health['status']}")
        
        # 2. Upload log file
        print("\n📁 Uploading log file...")
        upload_result = client.upload_log_file("sample_log.txt")
        session_id = upload_result['session_id']
        print(f"✅ Upload successful! Session ID: {session_id}")
        print(f"📊 Processed {upload_result['chunks_created']} text chunks")
        
        # 3. Analyze issues
        questions = [
            "What errors are present in my log file?",
            "Why is my application crashing?",
            "How can I fix the connection timeout issues?",
            "Analyze the performance problems in my logs"
        ]
        
        for question in questions:
            print(f"\n❓ Question: {question}")
            print("🤖 Analyzing...")
            
            result = client.analyze_issue(question, session_id)
            
            print(f"📝 Answer:\n{result['answer']}")
            print(f"🛠️ Tools used: {', '.join(result['tools_used'])}")
            print(f"⏰ Timestamp: {result['timestamp']}")
            print("-" * 80)
        
        # 4. Get session info
        print(f"\n📊 Session Info:")
        session_info = client.get_session_info(session_id)
        print(json.dumps(session_info, indent=2))
        
        # 5. Clean up (optional)
        print(f"\n🗑️ Cleaning up session...")
        client.delete_session(session_id)
        print("✅ Session deleted successfully")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

---

# test_api.py - Unit tests for the API
import pytest
import tempfile
import os
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data

def test_upload_log_file():
    """Test log file upload"""
    # Create a temporary log file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        f.write("2024-01-01 10:00:00 INFO Application started\n")
        f.write("2024-01-01 10:01:00 ERROR Connection timeout\n")
        f.write("2024-01-01 10:02:00 WARN Memory usage high\n")
        temp_file = f.name
    
    try:
        with open(temp_file, 'rb') as f:
            response = client.post("/upload", files={"file": ("test.log", f, "text/plain")})
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "filename" in data
        assert "chunks_created" in data
        
        return data["session_id"]  # Return for other tests
    finally:
        os.unlink(temp_file)

def test_analyze_without_session():
    """Test analysis without uploading a file first"""
    response = client.post("/analyze", json={
        "question": "What errors are in my log?",
        "session_id": "non_existent_session"
    })
    assert response.status_code == 404

def test_analyze_with_session():
    """Test analysis with a valid session"""
    # First upload a file
    session_id = test_upload_log_file()
    
    # Then analyze
    response = client.post("/analyze", json={
        "question": "What errors are present in the log file?",
        "session_id": session_id
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "session_id" in data
    assert "timestamp" in data
    assert "tools_used" in data

def test_invalid_file_type():
    """Test upload with invalid file type"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.exe', delete=False) as f:
        f.write("invalid content")
        temp_file = f.name
    
    try:
        with open(temp_file, 'rb') as f:
            response = client.post("/upload", files={"file": ("test.exe", f, "application/octet-stream")})
        
        assert response.status_code == 400
    finally:
        os.unlink(temp_file)

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

---

# deployment_guide.md
# iLogAnalyze API - Azure Deployment Guide

## 🚀 Quick Start

### Prerequisites
- Azure subscription
- Azure CLI installed
- Docker installed (for local testing)
- Python 3.11+
- Google Generative AI API key

### 1. Local Development Setup

```bash
# Clone your repository
git clone <your-repo>
cd iloganalyze-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run locally
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Test the API Locally

```bash
# Health check
curl http://localhost:8000/health

# Upload a log file
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_log.txt"

# Analyze an issue
curl -X POST "http://localhost:8000/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"question": "What errors are in my log?", "session_id": "your-session-id"}'
```

### 3. Azure Container Registry Setup

```bash
# Login to Azure
az login

# Create resource group
az group create --name iloganalyze-rg --location eastus

# Create Azure Container Registry
az acr create --resource-group iloganalyze-rg \
  --name iloganalyzeacr --sku Basic

# Login to ACR
az acr login --name iloganalyzeacr

# Build and push Docker image
docker build -t iloganalyzeacr.azurecr.io/iloganalyze-api:latest .
docker push iloganalyzeacr.azurecr.io/iloganalyze-api:latest
```

### 4. Azure App Service Deployment

#### Option A: Using Azure CLI

```bash
# Create App Service Plan
az appservice plan create --name iloganalyze-plan \
  --resource-group iloganalyze-rg \
  --sku B1 --is-linux

# Create Web App
az webapp create --resource-group iloganalyze-rg \
  --plan iloganalyze-plan \
  --name iloganalyze-api \
  --deployment-container-image-name iloganalyzeacr.azurecr.io/iloganalyze-api:latest

# Configure app settings
az webapp config appsettings set --resource-group iloganalyze-rg \
  --name iloganalyze-api \
  --settings GOOGLE_API_KEY="your-api-key" \
             GOOGLE_CSE_ID="your-cse-id" \
             PORT="8000" \
             WEBSITES_PORT="8000"

# Enable container registry authentication
az webapp config container set --name iloganalyze-api \
  --resource-group iloganalyze-rg \
  --container-image-name iloganalyzeacr.azurecr.io/iloganalyze-api:latest \
  --container-registry-url https://iloganalyzeacr.azurecr.io
```

#### Option B: Using Bicep Template

```bash
# Deploy infrastructure using Bicep
az deployment group create \
  --resource-group iloganalyze-rg \
  --template-file bicep/main.bicep \
  --parameters webAppName=iloganalyze-api \
               googleApiKey="your-api-key" \
               googleCseId="your-cse-id"
```

### 5. Container Instances Deployment (Alternative)

```bash
# Create container instance
az container create --resource-group iloganalyze-rg \
  --name iloganalyze-container \
  --image iloganalyzeacr.azurecr.io/iloganalyze-api:latest \
  --cpu 1 --memory 2 \
  --registry-login-server iloganalyzeacr.azurecr.io \
  --registry-username iloganalyzeacr \
  --registry-password $(az acr credential show --name iloganalyzeacr --query "passwords[0].value" -o tsv) \
  --dns-name-label iloganalyze-api \
  --ports 8000 \
  --environment-variables GOOGLE_API_KEY="your-api-key" GOOGLE_CSE_ID="your-cse-id"
```

### 6. Set Up CI/CD Pipeline

#### GitHub Actions Workflow

Create `.github/workflows/azure-deploy.yml`:

```yaml
name: Deploy to Azure

on:
  push:
    branches: [ main ]

env:
  AZURE_WEBAPP_NAME: iloganalyze-api
  CONTAINER_REGISTRY: iloganalyzeacr.azurecr.io
  IMAGE_NAME: iloganalyze-api

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Login to Azure
      uses: azure/login@v1
      with:
        creds: ${{ secrets.AZURE_CREDENTIALS }}
    
    - name: Login to ACR
      run: az acr login --name iloganalyzeacr
    
    - name: Build and push Docker image
      run: |
        docker build -t ${{ env.CONTAINER_REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} .
        docker push ${{ env.CONTAINER_REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
    
    - name: Deploy to Azure Web App
      uses: azure/webapps-deploy@v2
      with:
        app-name: ${{ env.AZURE_WEBAPP_NAME }}
        images: ${{ env.CONTAINER_REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
```

### 7. Environment Configuration

#### Required Environment Variables:
- `GOOGLE_API_KEY`: Your Google Generative AI API key (Required)
- `GOOGLE_CSE_ID`: Google Custom Search Engine ID (Optional)
- `PORT`: Application port (default: 8000)

#### Azure App Service Configuration:
1. Go to Azure Portal
2. Navigate to your App Service
3. Go to Configuration > Application settings
4. Add the required environment variables
5. Save and restart the application

### 8. Monitoring and Logging

#### Enable Application Insights:

```bash
# Create Application Insights
az monitor app-insights component create \
  --app iloganalyze-insights \
  --location eastus \
  --resource-group iloganalyze-rg

# Get instrumentation key
INSTRUMENTATION_KEY=$(az monitor app-insights component show \
  --app iloganalyze-insights \
  --resource-group iloganalyze-rg \
  --query instrumentationKey -o tsv)

# Configure app setting
az webapp config appsettings set --resource-group iloganalyze-rg \
  --name iloganalyze-api \
  --settings APPINSIGHTS_INSTRUMENTATIONKEY="$INSTRUMENTATION_KEY"
```

### 9. Security Best Practices

1. **API Keys**: Store sensitive data in Azure Key Vault
2. **CORS**: Configure proper CORS settings for production
3. **Authentication**: Add API authentication if needed
4. **HTTPS**: Ensure HTTPS-only access
5. **Rate Limiting**: Implement rate limiting for API endpoints

#### Azure Key Vault Integration:

```bash
# Create Key Vault
az keyvault create --name iloganalyze-kv \
  --resource-group iloganalyze-rg \
  --location eastus

# Add secrets
az keyvault secret set --vault-name iloganalyze-kv \
  --name google-api-key --value "your-api-key"

# Grant access to App Service
az webapp identity assign --resource-group iloganalyze-rg \
  --name iloganalyze-api

# Set access policy
az keyvault set-policy --name iloganalyze-kv \
  --object-id $(az webapp identity show --resource-group iloganalyze-rg --name iloganalyze-api --query principalId -o tsv) \
  --secret-permissions get
```

### 10. Scaling and Performance

#### Auto-scaling rules:

```bash
# Create auto-scale profile
az monitor autoscale create --resource-group iloganalyze-rg \
  --resource /subscriptions/{subscription-id}/resourceGroups/iloganalyze-rg/providers/Microsoft.Web/serverfarms/iloganalyze-plan \
  --name iloganalyze-autoscale \
  --min-count 1 --max-count 10 --count 2

# Add scale-out rule
az monitor autoscale rule create --resource-group iloganalyze-rg \
  --autoscale-name iloganalyze-autoscale \
  --condition "Percentage CPU > 70 avg 5m" \
  --scale out 2
```

### 11. Testing the Deployed API

```python
# test_production.py
import requests

API_BASE_URL = "https://iloganalyze-api.azurewebsites.net"

def test_production_api():
    # Health check
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Health check: {response.status_code}")
    
    # Upload test file
    with open("sample_log.txt", "rb") as f:
        response = requests.post(f"{API_BASE_URL}/upload", files={"file": f})
        if response.status_code == 200:
            session_id = response.json()["session_id"]
            print(f"Upload successful: {session_id}")
            
            # Test analysis
            analysis_response = requests.post(f"{API_BASE_URL}/analyze", json={
                "question": "What errors are in this log file?",
                "session_id": session_id
            })
            print(f"Analysis: {analysis_response.status_code}")

if __name__ == "__main__":
    test_production_api()
```

## 🔧 Troubleshooting

### Common Issues:

1. **Container fails to start**: Check environment variables and port configuration
2. **API timeouts**: Increase timeout settings in Azure App Service
3. **Memory issues**: Upgrade to higher SKU (B2, S1, etc.)
4. **Authentication errors**: Verify Google API key is correctly set

### Useful Commands:

```bash
# View logs
az webapp log tail --name iloganalyze-api --resource-group iloganalyze-rg

# Restart app
az webapp restart --name iloganalyze-api --resource-group iloganalyze-rg

# Check app settings
az webapp config appsettings list --name iloganalyze-api --resource-group iloganalyze-rg
```

## 📚 API Documentation

Once deployed, access the interactive API documentation at:
- Swagger UI: `https://your-app.azurewebsites.net/docs`
- ReDoc: `https://your-app.azurewebsites.net/redoc`

## 💰 Cost Optimization

1. Use Azure Dev/Test pricing for development
2. Configure auto-shutdown for development environments
3. Monitor usage with Azure Cost Management
4. Use reserved instances for production workloads