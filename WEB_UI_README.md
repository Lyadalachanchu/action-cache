# 🧠 Weaviate Agent Web UI

A beautiful web interface for exploring the AI agent's thinking process, similarity detection, and execution planning.

## 🎯 Features

### 🔍 **Question Analysis**
- **Real-time similarity detection** with existing tasks
- **AI thinking process** visualization
- **Execution plan generation** based on similarity
- **Learning system** that gets smarter over time

### 📊 **Visual Interface**
- **Modern, responsive design** with gradient backgrounds
- **Real-time status indicators** for Weaviate connection
- **Interactive similarity scores** with color coding
- **Step-by-step execution plans** with time estimates

### 🚀 **Smart Features**
- **Pattern reuse** for similar questions (saves time and costs)
- **Adaptive planning** for related questions
- **Cross-topic learning** (works with any subject)
- **Confidence scoring** for execution plans

## 🛠️ Setup

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Set Environment Variables**
Create a `.env` file with your API keys:
```bash
WEAVIATE_URL=your_weaviate_url
WEAVIATE_API_KEY=your_weaviate_api_key
OPENAI_API_KEY=your_openai_api_key
```

### 3. **Start the Web UI**
```bash
python start_web_ui.py
```

### 4. **Open in Browser**
Navigate to: `http://localhost:5000`

## 🎭 Demo

Run the demo to see the functionality:
```bash
python demo_web_ui.py
```

## 📱 How to Use

### 1. **Ask a Question**
- Type any question in the search box
- Examples:
  - "Who is older, Chris Martin or Matt Damon?"
  - "How many people does Paris have?"
  - "What is the capital of France?"

### 2. **View AI Thinking**
The interface shows:
- **🧠 AI Thinking Process**: How the AI analyzes your question
- **🔍 Similarity Analysis**: Similar tasks found in the database
- **⚡ Execution Plan**: Step-by-step plan with time estimates

### 3. **Understand the Results**
- **High Similarity (≥80%)**: Reuses existing patterns (fastest)
- **Medium Similarity (70-80%)**: Adapts existing patterns
- **Low Similarity (<70%)**: Creates new plans

## 🎯 Example Workflow

### **First Question**: "Who is older, Chris Martin or Matt Damon?"
- **Result**: No similar tasks found → Creates new plan
- **Time**: 5-7 minutes
- **Confidence**: Medium

### **Second Question**: "How old is Chris Martin?"
- **Result**: Finds first question (80% similar) → Reuses pattern
- **Time**: 2-3 minutes
- **Confidence**: High

## 🔧 Technical Details

### **Architecture**
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Backend**: Flask (Python)
- **Database**: Weaviate (Vector Database)
- **AI**: OpenAI GPT models

### **API Endpoints**
- `POST /search` - Analyze a question
- `GET /similarity/<question>` - Get similarity analysis
- `GET /history` - Get search history

### **Similarity Thresholds**
- **≥80%**: Reuse existing pattern (fastest)
- **70-80%**: Adapt existing pattern
- **<70%**: Create new plan

## 🎨 UI Components

### **Status Indicator**
- 🟢 **Green**: Weaviate connected and ready
- 🔴 **Red**: Weaviate offline or connection error

### **Similarity Scores**
- 🟢 **High (≥80%)**: Green background
- 🟡 **Medium (60-80%)**: Yellow background
- 🔴 **Low (<60%)**: Red background

### **Execution Plans**
- **REUSE**: Use existing proven pattern
- **ADAPT**: Modify existing pattern
- **CREATE_NEW**: Generate new plan from scratch

## 🚀 Benefits

### **For Users**
- **Faster answers** through pattern reuse
- **Transparent AI thinking** process
- **Cost savings** through efficient planning
- **Learning system** that improves over time

### **For Developers**
- **RESTful API** for integration
- **Modular design** for easy extension
- **Real-time analysis** capabilities
- **Scalable architecture**

## 🔮 Future Enhancements

- **Real-time execution** with live progress updates
- **Question history** and analytics
- **Custom similarity thresholds**
- **Export results** to various formats
- **Multi-language support**

## 🐛 Troubleshooting

### **Common Issues**
1. **"Weaviate not available"**: Check your environment variables
2. **"Connection error"**: Verify Weaviate URL and API key
3. **"No similar tasks"**: Database is empty, first questions will create new plans

### **Debug Mode**
Run with debug logging:
```bash
FLASK_DEBUG=1 python start_web_ui.py
```

## 📞 Support

For issues or questions:
1. Check the demo: `python demo_web_ui.py`
2. Verify environment variables
3. Check Weaviate connection status
4. Review the console output for errors

---

**🎉 Enjoy exploring the AI agent's thinking process!**
