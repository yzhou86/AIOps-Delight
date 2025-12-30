from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.chat_models.tongyi import ChatTongyi

app = Flask(__name__)
CORS(app)

# 配置通义千问模型
chat_model = ChatTongyi(
    api_key="sk-c003b7a1f93d4067a64e690a4b020bdb",
    model="qwen-turbo"
)

@app.route('/')
def hello():
    return 'Hello, AI Agent!'

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
            
        # 使用LangChain调用模型
        response = chat_model.invoke([user_message])
        
        return jsonify({
            'message': user_message,
            'response': response.content
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)