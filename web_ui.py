#!/usr/bin/env python3
"""
Web UI for the Weaviate-integrated agent
Shows thinking logic, similarity detection, and execution path
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import asyncio
import json
import os
from datetime import datetime
from weaviate_service import WeaviateService

app = Flask(__name__)

# Global Weaviate service
weaviate_service = None

def get_weaviate_service():
    """Get or create Weaviate service"""
    global weaviate_service
    if weaviate_service is None:
        weaviate_service = WeaviateService()
    return weaviate_service

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_question():
    """Search for a question and return thinking process"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Please provide a question'}), 400
        
        # Get Weaviate service
        service = get_weaviate_service()
        
        # Create response structure
        response = {
            'question': question,
            'timestamp': datetime.now().isoformat(),
            'thinking_process': {},
            'similarity_analysis': {},
            'execution_plan': {},
            'weaviate_status': service.is_available()
        }
        
        # Step 1: Similarity Analysis
        if service.is_available():
            similar_tasks = service.search_similar_tasks(question, limit=5, similarity_threshold=0.5)
            response['similarity_analysis'] = {
                'found_similar': len(similar_tasks) > 0,
                'similar_tasks': similar_tasks,
                'best_match': similar_tasks[0] if similar_tasks else None,
                'reuse_decision': None
            }
            
            if similar_tasks:
                best_similarity = similar_tasks[0]['similarity']
                if best_similarity >= 0.8:
                    response['similarity_analysis']['reuse_decision'] = 'REUSE_EXISTING_PATTERN'
                    response['thinking_process']['similarity_reasoning'] = f"Found highly similar task (similarity: {best_similarity:.2f}) - will reuse existing pattern for faster execution"
                elif best_similarity >= 0.7:
                    response['similarity_analysis']['reuse_decision'] = 'ADAPT_EXISTING_PATTERN'
                    response['thinking_process']['similarity_reasoning'] = f"Found similar task (similarity: {best_similarity:.2f}) - will adapt existing pattern"
                else:
                    response['similarity_analysis']['reuse_decision'] = 'CREATE_NEW_PLAN'
                    response['thinking_process']['similarity_reasoning'] = f"Found similar tasks but similarity too low ({best_similarity:.2f} < 0.7) - will create new plan"
            else:
                response['similarity_analysis']['reuse_decision'] = 'CREATE_NEW_PLAN'
                response['thinking_process']['similarity_reasoning'] = "No similar tasks found - will create new plan"
        else:
            response['thinking_process']['similarity_reasoning'] = "Weaviate not available - will create new plan"
        
        # Step 2: Execution Plan Generation
        if response['similarity_analysis']['reuse_decision'] == 'REUSE_EXISTING_PATTERN':
            response['execution_plan'] = {
                'type': 'REUSE',
                'steps': [
                    "1. Load existing pattern from similar task",
                    "2. Adapt pattern for current question",
                    "3. Execute adapted pattern",
                    "4. Extract answer from results"
                ],
                'estimated_time': '2-3 minutes',
                'confidence': 'High (using proven pattern)'
            }
        elif response['similarity_analysis']['reuse_decision'] == 'ADAPT_EXISTING_PATTERN':
            response['execution_plan'] = {
                'type': 'ADAPT',
                'steps': [
                    "1. Analyze similar task pattern",
                    "2. Modify pattern for current question",
                    "3. Execute modified pattern",
                    "4. Extract answer from results"
                ],
                'estimated_time': '3-4 minutes',
                'confidence': 'Medium (adapting existing pattern)'
            }
        else:
            response['execution_plan'] = {
                'type': 'CREATE_NEW',
                'steps': [
                    "1. Break down question into subgoals",
                    "2. Create action plan for each subgoal",
                    "3. Execute Wikipedia research",
                    "4. Extract and synthesize answer"
                ],
                'estimated_time': '5-7 minutes',
                'confidence': 'Medium (creating new plan)'
            }
        
        # Step 3: Save task for future learning
        if service.is_available():
            task_data = {
                'title': f"Research: {question}",
                'actions': [
                    {"action": "read_page"},
                    {"action": "done"}
                ]
            }
            result = service.save_subtask_with_actions(task_data)
            response['thinking_process']['task_saved'] = f"Task saved with ID: {result}" if result else "Failed to save task"
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Search failed: {str(e)}'}), 500

@app.route('/similarity/<question>')
def get_similarity(question):
    """Get similarity analysis for a question"""
    try:
        service = get_weaviate_service()
        if not service.is_available():
            return jsonify({'error': 'Weaviate not available'}), 500
        
        similar_tasks = service.search_similar_tasks(question, limit=10, similarity_threshold=0.3)
        
        return jsonify({
            'question': question,
            'similar_tasks': similar_tasks,
            'total_found': len(similar_tasks)
        })
        
    except Exception as e:
        return jsonify({'error': f'Similarity analysis failed: {str(e)}'}), 500

@app.route('/history')
def get_history():
    """Get search history"""
    try:
        service = get_weaviate_service()
        if not service.is_available():
            return jsonify({'error': 'Weaviate not available'}), 500
        
        # Get recent tasks (this is a simplified version)
        # In a real implementation, you'd want to track search history
        return jsonify({
            'message': 'Search history feature coming soon',
            'weaviate_status': service.is_available()
        })
        
    except Exception as e:
        return jsonify({'error': f'History retrieval failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("🚀 Starting Web UI for Weaviate Agent")
    print("📱 Open your browser to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
