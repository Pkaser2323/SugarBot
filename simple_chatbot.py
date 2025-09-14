import os
import json
import io
import re
from typing import List, Tuple
from dotenv import load_dotenv
from flask import Flask, request
from pyngrok import ngrok
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import (
    MessageEvent,
    TextMessage,
    TextSendMessage,
    ImageMessage,
    FlexSendMessage,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
import logging
from sentence_transformers import CrossEncoder
import torch
import numpy as np
import hmac
import hashlib
import base64
from PIL import Image
from flexMessage import generate_carousel_flex, generate_flex_message, generate_calorie_source_flex_message
from FatSecret.FatAPI import search_food_with_fatsecret

# Load environment variables
load_dotenv()
API_KEY = os.environ.get("GOOGLE_API_KEY")
LINE_ACCESS_TOKEN = os.environ.get("LINE_ACCESS_TOKEN")
LINE_SECRET = os.environ.get("LINE_SECRET")

# Configure Gemini AI
genai.configure(api_key=API_KEY)

generation_config = {
    "temperature": 0.2,
    "max_output_tokens": 512,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)

# 模型配置
EMBED_MODEL_NAME = "DMetaSoul/sbert-chinese-general-v2"
SAS_MODEL_DIR = os.path.join(os.path.dirname(__file__), "sas_model")

# 載入 SAS 模型參數
with open(os.path.join(SAS_MODEL_DIR, "best_params.json"), "r", encoding="utf-8") as f:
    SAS_PARAMS = json.load(f)

# 初始化 SAS 模型
sas_model = CrossEncoder(SAS_MODEL_DIR)
sas_model.model = sas_model.model.to("cpu")  # 預設使用 CPU

def predict_pos_prob(
    model,
    questions: List[str],
    answers: List[str],
    temperature: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """預測正類機率"""
    # 空輸入檢查
    if not questions or not answers or len(questions) != len(answers):
        return np.array([]), np.array([])
    
    # 過濾無效輸入
    valid_pairs = []
    for q, a in zip(questions, answers):
        if not (isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip()):
            continue
        valid_pairs.append([q.strip(), a.strip()])
    
    if not valid_pairs:
        return np.array([]), np.array([])
    
    try:
        # 取得 logits
        logits = model.predict(valid_pairs, apply_softmax=False)
        logits = np.array(logits)
        
        # 檢查數值範圍
        if np.any(np.isnan(logits)) or np.any(np.isinf(logits)):
            print("⚠️ 檢測到 NaN 或 Inf 值，將替換為 0")
            logits = np.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # 應用溫度校準
        scaled_logits = logits / temperature
        
        # 轉換為機率
        if scaled_logits.ndim == 2:
            # 對於二分類，使用 softmax
            scaled_logits = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
            exp_scores = np.exp(scaled_logits)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            pos_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        else:
            # 對於單一分數，使用 sigmoid
            pos_probs = 1 / (1 + np.exp(-scaled_logits))
        
        # 確保機率在 [0,1] 範圍內
        pos_probs = np.clip(pos_probs, 0, 1)
        
        return pos_probs.copy(), pos_probs.copy()
        
    except Exception as e:
        print(f"⚠️ 預測過程發生錯誤: {e}")
        return np.array([]), np.array([])

def generate_subqueries(question: str, k: int = 2) -> List[str]:
    """使用 GPT 將問題拆解為子問題"""
    prompt = f"""請將以下糖尿病相關問題拆解成 {k} 個核心子問題。

原問題：{question}

要求：
1. 每個子問題必須針對原問題的不同核心要點
2. 子問題之間不能重複
3. 子問題要簡潔直接，不要贅句
4. 直接列出子問題，每行一個，不要加編號或符號

請列出子問題："""

    try:
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.1, "max_output_tokens": 300}
        )
        
        if not response or not hasattr(response, "text"):
            return [question]
            
        text = response.text.strip()
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        
        # 過濾並去重子問題
        subqs = []
        seen = set()
        for line in lines:
            # 移除可能的編號和符號
            line = line.lstrip("0123456789. )-•").strip()
            # 檢查長度和重複
            if len(line) > 5 and line not in seen:
                subqs.append(line)
                seen.add(line)
            if len(subqs) >= k:
                break
        
        # 如果子問題不夠，補充預設問題
        while len(subqs) < k:
            if not subqs:
                subqs.append(question)
            else:
                default_q = f"{question}的{len(subqs)+1}個面向是什麼？"
                if default_q not in seen:
                    subqs.append(default_q)
                    seen.add(default_q)
        
        return subqs[:k]
    except Exception as e:
        print(f"GPT拆解子問題失敗: {e}")
        return [question]

def initialize_vector_db():
    """初始化向量資料庫，如果不存在則從 CSV 創建"""
    import pandas as pd
    
    # 設置路徑
    base_dir = os.path.dirname(__file__)
    db_dir = os.path.join(base_dir, "vector_DB")
    db_path = os.path.join(db_dir, "diabetes_comprehensive_db")
    csv_path = os.path.join(base_dir, "datacsv", "a_topic_analyzed_processed.csv")
    
    # 確保目錄存在
    os.makedirs(db_dir, exist_ok=True)
    
    # 檢查向量資料庫是否已存在
    if os.path.exists(db_path):
        print("✓ 向量資料庫已存在")
        return db_path
        
    print("⚙️ 向量資料庫不存在，開始創建...")
    
    # 檢查 CSV 文件
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到資料文件：{csv_path}")
    
    # 讀取 CSV
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    
    # 檢查必要欄位
    required_cols = ["對應子問題", "回答"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV 缺少必要欄位: {', '.join(missing_cols)}")
    
    # 過濾無效行
    df = df.dropna(subset=["對應子問題", "回答"])
    
    # 準備文本
    texts = []
    for _, row in df.iterrows():
        # 組合問題和答案
        text = f"問題：{row['對應子問題']}\n答案：{row['回答']}"
        texts.append(text)
    
    print(f"✓ 載入了 {len(texts)} 筆問答對")
    
    # 創建嵌入
    model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs=model_kwargs
    )
    
    # 創建向量資料庫
    db = FAISS.from_texts(texts, embeddings)
    
    # 保存資料庫
    db.save_local(db_path)
    print(f"✓ 向量資料庫已保存至：{db_path}")
    
    return db_path

def generate_retriever():
    """生成檢索器"""
    print("Loading vector DB...")
    
    # 初始化/載入向量資料庫
    db_path = initialize_vector_db()
    
    # 創建嵌入模型
    model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
    embedding = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs=model_kwargs
    )
    
    # 載入資料庫
    db = FAISS.load_local(db_path, embedding, allow_dangerous_deserialization=True)
    print("Done loading vector DB!")
    
    return db.as_retriever(search_kwargs={"k": 5})

retriever = generate_retriever()

def search_related_content(retriever, query):
    """檢索相關文本
    
    Args:
        retriever: 檢索器
        query: 查詢文本
        
    Returns:
        Tuple[str, List]: (合併後的文本, 文檔列表)
    """
    docs = retriever.invoke(query)
    return "\n---\n".join([doc.page_content for doc in docs]), docs

def generate_answer(query: str, docs=None):
    """生成回答，使用 Fast/Slow path 機制"""
    # 如果有檢索結果，先用 Fast path 評估
    if docs:
        print("🚀 Fast path: 評估檢索結果...")
        # 使用 SAS 評估每個檢索結果
        _, probs = predict_pos_prob(
            sas_model,
            [query] * len(docs),
            [doc.page_content for doc in docs],
            temperature=SAS_PARAMS.get("temperature", 2.0)
        )
        
        # 檢查是否有段落通過高門檻
        high_thr = SAS_PARAMS.get("high_threshold", 0.6)
        passed_indices = np.nonzero(probs >= high_thr)[0]
        
        if len(passed_indices) > 0:
            print(f"\n✅ 找到 {len(passed_indices)} 個通過高門檻的段落：")
            # 顯示所有通過門檻的段落及其分數
            for i, idx in enumerate(passed_indices):
                print(f"\n段落 {i+1} (相關度分數: {probs[idx]:.3f}):")
                print("-" * 50)
                print(docs[idx].page_content)
                print("-" * 50)
            
            # 選擇最多3個最高分的段落
            top_indices = passed_indices[np.argsort(-probs[passed_indices])[:3]]
            print(f"\n🔍 選擇前 {len(top_indices)} 個最高分段落用於生成回答")
            evidence = "\n".join([docs[i].page_content for i in top_indices])
            
            # 使用 GPT 生成回答
            template = f"""
任務: 
1. 你是一位在台灣的糖尿病領域的專業護理師，需要以專業嚴謹但親切的態度回答病患的問題。

2. 請仔細分析下方的「相關文本」，並按照以下步驟回答：
   a. 從「相關文本」中提取可靠且相關的醫療資訊
   b. 確保所提供的每一項建議都有文獻依據
   c. 整合資訊時，需明確區分：
      - 確定的醫療建議（有明確依據）
      - 一般性建議（基於專業知識）
   d. 使用準確的醫療術語，並提供清晰的解釋

3. 回答要求：
   - 字數限制：最多100字，且回答須清晰易懂
   - 不需列出文獻來源，只根據「相關文本」作答  
   - 使用繁體中文，語氣親切清晰  
   - 分段呈現，提高可讀性

------
「相關文本」：
{evidence}
------
「病患的提問」：
{query}

請基於上述相關文本，提供專業且實用的回答：
"""
            response = model.generate_content(template)
            return response.text if response else "不好意思，我不清楚這個問題，建議您諮詢專業醫師。"
    
    # 如果 Fast path 失敗，嘗試 Slow path
    print("🐢 Slow path: 拆解子問題...")
    subqs = generate_subqueries(query)
    print(f"✓ 生成 {len(subqs)} 個子問題")
    
    # 為每個子問題檢索並評估
    all_evidence = []
    low_thr = SAS_PARAMS.get("low_threshold", 0.3)
    
    for sq in subqs:
        # 檢索相關文本
        sq_docs = retriever.invoke(sq)
        if not sq_docs:
            continue
            
        # 評估每個檢索結果
        _, probs = predict_pos_prob(
            sas_model,
            [sq] * len(sq_docs),
            [doc.page_content for doc in sq_docs],
            temperature=SAS_PARAMS.get("temperature", 2.0)
        )
        
        # 收集通過低門檻的段落
        passed_indices = np.nonzero(probs >= low_thr)[0]
        if len(passed_indices) > 0:
            # 選擇最多2個最高分的段落
            top_indices = passed_indices[np.argsort(-probs[passed_indices])[:2]]
            all_evidence.extend([sq_docs[i].page_content for i in top_indices])
    
    # 如果沒有找到任何有效證據
    if not all_evidence:
        return "這個問題需要更多專業資訊才能完整回答，建議您諮詢主治醫師。"
    
    # 使用 GPT 整合所有證據生成回答
    evidence_text = "\n".join(all_evidence)
    template = f"""
任務: 
1. 你是一位在台灣的糖尿病領域的專業護理師，需要以專業嚴謹但親切的態度回答病患的問題。

2. 請仔細分析下方的「相關文本」，並按照以下步驟回答：
   a. 從「相關文本」中提取可靠且相關的醫療資訊
   b. 確保所提供的每一項建議都有文獻依據
   c. 整合資訊時，需明確區分：
      - 確定的醫療建議（有明確依據）
      - 一般性建議（基於專業知識）
   d. 使用準確的醫療術語，並提供清晰的解釋

3. 回答要求：
   - 字數限制：最多100字，且回答須清晰易懂
   - 不需列出文獻來源，只根據「相關文本」作答  
   - 使用繁體中文，語氣親切清晰  
   - 分段呈現，提高可讀性

------
「相關文本」：
{evidence_text}
------
「病患的提問」：
{query}

請基於上述相關文本，提供專業且實用的回答：
"""
    response = model.generate_content(template)
    answer = response.text if response else "不好意思，我不清楚這個問題，建議您諮詢專業醫師。"
    
    # 最後用原問題評估生成的答案
    _, probs = predict_pos_prob(
        sas_model,
        [query],
        [answer],
            temperature=SAS_PARAMS.get("temperature", 2.0)
    )
    
    # 如果最終答案未通過高門檻，建議諮詢醫師
    if probs[0] < SAS_PARAMS.get("high_threshold", 0.6):
        return "這個問題需要更多專業資訊才能完整回答，建議您諮詢主治醫師。"
    
    return answer

# Flask app setup
app = Flask(__name__)
port = 5000

# LINE Bot setup
line_bot_api = LineBotApi(LINE_ACCESS_TOKEN)
handler = WebhookHandler(LINE_SECRET)

# 測試 LINE API 連線
try:
    print("✅ Testing LINE API connection")
    print("✅ LINE API setup completed")
except Exception as e:
    print(f"❌ LINE API setup error: {str(e)}")

# 健康檢查路由
@app.route("/health", methods=["GET"])
def health_check():
    return "OK", 200

# LINE Bot webhook
@app.route("/callback", methods=["POST"])
def callback():
    # 獲取 X-Line-Signature header 值
    signature = request.headers.get("X-Line-Signature", "")
    
    # 獲取請求內容
    body = request.get_data(as_text=True)
    print(f"✅ Received webhook: {body[:100]}...")  # 只打印前100個字符
    
    try:
        # 驗證簽名
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("❌ Invalid signature")
        return "Invalid signature", 400
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return str(e), 500
        
    return "OK", 200

# 全局數據存儲，用於保存食物分析結果
global_data_store = {}

def translate_to_chinese(english_text):
    """翻譯英文食物名稱為繁體中文"""
    translation_prompt = f"""請將以下食物名稱翻譯為繁體中文，精準翻譯，只回傳食物名稱，不要其他描述或多餘的詞彙。
{english_text}
"""
    response = model.generate_content(translation_prompt)
    if not response or not hasattr(response, "text"):
        return english_text
    return response.text.strip()

def analyze_nutrition_for_flex(nutrition_data):
    """分析營養數據，提取優點、風險和建議"""
    analysis_prompt = f"""任務:
1. 你是一位專業營養師，請根據以下食物的營養資訊進行分析：
2. 分析結果必須包含這三個區塊：優點、潛在風險、建議（針對糖尿病患者）
3. 每個區塊提供 1-2 點簡潔的分析，每點不超過15字
4. 使用繁體中文

【營養數據】：
{nutrition_data}

請用以下JSON格式回答：
{{"優點":["優點1", "優點2"], "潛在風險":["風險1", "風險2"], "建議":["建議1", "建議2"]}}
"""
    try:
        gemini_response = model.generate_content(analysis_prompt)
        if not gemini_response or not hasattr(gemini_response, "text"):
            return {"優點": [], "潛在風險": [], "建議": []}

        # 解析 JSON 格式的回應
        analysis_text = gemini_response.text.strip()
        match = re.search(r"(\{.*\})", analysis_text, re.DOTALL)
        if match:
            analysis_json = match.group(1)
            try:
                return json.loads(analysis_json)
            except:
                return {"優點": [], "潛在風險": [], "建議": []}
        return {"優點": [], "潛在風險": [], "建議": []}
    except Exception as e:
        print(f"分析營養數據時出錯: {str(e)}")
        return {"優點": [], "潛在風險": [], "建議": []}

def calculate_calorie_sources(nutrition_data_list):
    """計算熱量來源佔比"""
    total_carb_calories = 0
    total_protein_calories = 0
    total_fat_calories = 0
    total_sugar_calories = 0
    total_calories = 0

    # 熱量換算：碳水4卡/克，蛋白質4卡/克，脂肪9卡/克，糖分4卡/克
    for data in nutrition_data_list:
        carb = float(data.get("carbohydrate", 0) or 0)
        protein = float(data.get("protein", 0) or 0)
        fat = float(data.get("fat", 0) or 0)
        sugar = float(data.get("sugar", 0) or 0)

        carb_cal = carb * 4
        protein_cal = protein * 4
        fat_cal = fat * 9
        sugar_cal = sugar * 4

        total_carb_calories += carb_cal
        total_protein_calories += protein_cal
        total_fat_calories += fat_cal
        total_sugar_calories += sugar_cal
        total_calories += float(data.get("calories", 0) or 0)

    return {
        "carbs_calories": round(total_carb_calories, 0),
        "protein_calories": round(total_protein_calories, 0),
        "fat_calories": round(total_fat_calories, 0),
        "sugar_calories": round(total_sugar_calories, 0),
        "total_calories": round(total_calories, 0),
        "is_estimated": total_calories == 0,
    }

def analyze_food_image(image_path):
    """
    使用 Gemini Vision 分析食物圖片，並生成營養分析 Flex Message
    """
    try:
        # 讀取圖片並轉換為 Base64
        with Image.open(image_path) as image:
            buffered = io.BytesIO()
            image_format = image.format
            image.save(buffered, format=image_format)
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Gemini Vision 分析圖片內容
        vision_prompt = """請擷取圖片中所有主要的食物名稱（英文），用逗號分隔，例如：
"apple, banana, sandwich"
"""
        vision_response = model.generate_content(
            [{"mime_type": f"image/{image_format.lower()}", "data": image_base64}, vision_prompt]
        )

        if not vision_response or not hasattr(vision_response, "text"):
            return TextSendMessage(text="⚠️ 無法辨識圖片，請試試另一張！")

        food_list = [food.strip().lower() for food in vision_response.text.strip().split(",")]
        if not food_list:
            return TextSendMessage(text="⚠️ 無法識別主要食物，請提供更清晰的圖片！")

        print(f"🔍 提取的食物名稱: {food_list}")

        # 查詢 FatSecret API 並分析
        nutrition_data_list = []
        food_chinese_names = []
        api_data_found = False

        for food in food_list:
            nutrition_data = search_food_with_fatsecret(food)
            if not isinstance(nutrition_data, dict):
                print(f"⚠️ FatSecret API 回傳錯誤數據: {nutrition_data}")
                continue

            food_chinese_name = translate_to_chinese(food.capitalize())
            food_chinese_names.append(food_chinese_name)

            nutrition_data["food_name"] = food
            nutrition_data["food_chinese_name"] = food_chinese_name
            nutrition_data_list.append(nutrition_data)

            if "calories" in nutrition_data and nutrition_data.get("calories"):
                api_data_found = True

        if not nutrition_data_list:
            return TextSendMessage(text="⚠️ 無法獲取食物的營養資訊，請稍後再試。")

        # 計算熱量來源佔比
        calorie_sources = calculate_calorie_sources(nutrition_data_list)
        calorie_sources["is_estimated"] = not api_data_found

        # 生成熱量來源分析 Flex Message
        flex_message = generate_calorie_source_flex_message(food_chinese_names, calorie_sources)

        # 確保返回的是 LINE 的消息對象
        if isinstance(flex_message, dict):
            return FlexSendMessage(alt_text=f"{food_chinese_names[0]} 的熱量來源分析", contents=flex_message)
        else:
            return flex_message

    except Exception as e:
        print(f"🚨 圖片分析時發生錯誤: {str(e)}")
        return TextSendMessage(text="⚠️ 無法分析圖片，請稍後再試。")

# 處理文字訊息
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    """處理文字訊息事件"""
    print(f"✅ 收到訊息：{event.message.text}")
    message_text = event.message.text.strip()
    
    try:
        # 檢索相關文本並生成回答
        print(f"💬 處理一般文字訊息: {message_text}")
        _, docs = search_related_content(retriever, message_text)
        response = generate_answer(message_text, docs)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=response))
    
    except LineBotApiError as e:
        print(f"❌ Failed to reply message: {str(e)}")
    except Exception as e:
        print(f"❌ Error in handle_message: {str(e)}")
        import traceback
        traceback.print_exc()

# 處理圖片訊息
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    """處理圖片訊息事件"""
    try:
        print("✅ 收到圖片訊息")
        
        # 從 LINE 獲取圖片內容
        message_content = line_bot_api.get_message_content(event.message.id)
        
        # 創建臨時文件夾（如果不存在）
        temp_dir = "temp_images"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # 保存圖片到臨時文件
        image_path = os.path.join(temp_dir, f"{event.message.id}.jpg")
        with open(image_path, "wb") as f:
            for chunk in message_content.iter_content():
                f.write(chunk)
        
        try:
            # 分析圖片並獲取 Flex Message
            flex_message = analyze_food_image(image_path)
            
            # 回覆 Flex Message
            line_bot_api.reply_message(
                event.reply_token,
                flex_message
            )
            
        finally:
            # 確保無論如何都會刪除臨時文件
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
            except Exception as e:
                print(f"⚠️ 無法刪除臨時文件: {str(e)}")
        
    except LineBotApiError as e:
        print(f"❌ LINE API 錯誤: {str(e)}")
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="⚠️ 圖片處理失敗，請稍後再試。")
        )
    except Exception as e:
        print(f"❌ 處理圖片時發生錯誤: {str(e)}")
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="⚠️ 系統錯誤，請稍後再試。")
        )

def start_ngrok():
    """啟動 ngrok 服務"""
    try:
        # 確保使用 HTTPS
        public_url = ngrok.connect(port, bind_tls=True).public_url
        print(f' * ngrok tunnel "{public_url}" -> "http://127.0.0.1:{port}" ')
        print(f' * LINE Bot webhook URL: {public_url}/callback')
        return public_url
    except Exception as e:
        print(f"⚠️ ngrok 啟動失敗: {str(e)}")
        print("🔧 Bot 仍可以在本地運行，但需要手動設置外部連接")
        return None

if __name__ == "__main__":
    print("啟動糖尿病諮詢 LINE Bot")
    print("功能包含：")
    print("   - RAG 檢索")
    print("   - SAS 答案評估")
    print("   - Fast/Slow path 機制")
    print("   - 食物圖片分析")
    
    # 在 Flask 啟動前先啟動 ngrok
    public_url = start_ngrok()
    
    # 啟動 Flask 應用（生產環境中建議關閉 debug 模式）
    app.run(host="0.0.0.0", port=port, debug=False)
