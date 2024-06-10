from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

app = Flask(__name__)

# 禁用符号链接警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 加载模型和tokenizer
model_name = "gpt2"  # 模型的名称
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def chat_with_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    attention_mask = inputs['attention_mask']
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=attention_mask,
        max_length=100,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.2,  # 控制生成文本的随机性
        top_p=0.9,  # 控制采样方法的多样性
        num_return_sequences=1
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    prompt = data.get("prompt", "")
    response = chat_with_model(prompt)

    # 构建仿OpenAI格式的响应
    completion = {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "text": response
            }
        ]
    }

    return jsonify(completion)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
