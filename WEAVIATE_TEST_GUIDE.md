# How to Run weaviate_test.py

## 🔧 **Setup Required**

### **1. Get Weaviate Cloud Credentials**

If you don't have Weaviate Cloud yet:
1. Go to https://console.weaviate.cloud/
2. Create a free account
3. Create a new cluster
4. Get your cluster URL and API key

### **2. Set Environment Variables**

#### **Method A: PowerShell (Temporary)**
```powershell
$env:WEAVIATE_URL = "https://your-cluster-url.weaviate.network"
$env:WEAVIATE_API_KEY = "your-weaviate-api-key-here"
```

#### **Method B: Create .env file (Recommended)**
Create a file named `.env` in your project root:
```
WEAVIATE_URL=https://your-cluster-url.weaviate.network
WEAVIATE_API_KEY=your-weaviate-api-key-here
OPENAI_API_KEY=sk-your-openai-api-key-here
```

### **3. Run the Test**

```bash
# Activate virtual environment
venv\Scripts\activate

# Run the test
python weaviate_test.py
```

## 🎯 **Expected Output**

If successful:
```
Connected: True
```

If failed:
```
Error: [connection error details]
```

## 🔍 **Troubleshooting**

### **Error: "WEAVIATE_URL not found"**
- Make sure you set the environment variable
- Check that the .env file exists and has the right format

### **Error: "Authentication failed"**
- Check your API key is correct
- Make sure your Weaviate cluster is running

### **Error: "Connection timeout"**
- Check your internet connection
- Verify the cluster URL is correct
- Try increasing the timeout in the script

## 🚀 **Quick Test Without Weaviate**

If you want to test without setting up Weaviate Cloud, you can modify the script to use a mock connection:

```python
# Replace the connection code with:
print("Mock connection test - skipping actual Weaviate connection")
print("Connected: True (mock)")
```
