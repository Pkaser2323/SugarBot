import os
import time
import json
from dotenv import load_dotenv
from flask import Flask, request
from pyngrok import ngrok
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import (
    MessageEvent,
    TextMessage,
    TextSendMessage,
    FlexSendMessage,
    PostbackEvent,
    QuickReply,
    QuickReplyButton,
    DatetimePickerAction,
    PostbackAction,
    ImageSendMessage,
    ImageMessage,
    AudioMessage,
    CameraAction,
    CameraRollAction,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
import requests
import mimetypes
import io
import base64
import logging
from PIL import Image
from sentence_transformers import CrossEncoder
import torch
import numpy as np
import json
import os
from FatSecret.FatAPI import search_food_with_fatsecret
import re
from flexMessage import (
    generate_carousel_flex,
    generate_flex_message,
    generate_calorie_source_flex_message,
    get_calorie_source_analysis,
)
from datetime import datetime, timedelta
import pytz
import hmac
import hashlib
import uuid

# 設置 matplotlib 後端（避免 tkinter 衝突）
import matplotlib
matplotlib.use('Agg')

# 導入 blood_sugar.py 的函數
from blood_sugar import (
    record_blood_sugar,
    get_blood_sugar_by_date,
    update_blood_sugar,
    delete_blood_sugar,
    generate_blood_sugar_chart,
    create_blood_sugar_message,
    create_report_menu_message,
    handle_blood_sugar_report,
    show_records_for_edit,
    show_records_for_delete,
    FIREBASE_AVAILABLE as BLOOD_SUGAR_AVAILABLE,
)

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

# 添加全局變量來保存查詢結果
global_data_store = {}

# 儲存使用者狀態（判斷是否要記錄血糖）
user_states = {}

def predict_pos_prob(
    model,
    questions: List[str],
    answers: List[str],
    temperature: float = 2.0,
    penalty: float = -1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """預測正類機率
    
    Args:
        model: SAS 模型
        questions: 問題列表
        answers: 答案列表
        temperature: 溫度校準參數
        penalty: logits 空間的懲罰值
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (scores, probs)
    """
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
    """使用 GPT 將問題拆解為子問題
    
    Args:
        question: 原始問題
        k: 子問題數量
        
    Returns:
        List[str]: 子問題列表
    """
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
    required_cols = ["問題", "回答"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV 缺少必要欄位: {', '.join(missing_cols)}")
    
    # 過濾無效行
    df = df.dropna(subset=["問題", "回答"])
    
    # 準備文本
    texts = []
    for _, row in df.iterrows():
        # 組合問題和答案
        text = f"問題：{row['問題']}\n答案：{row['回答']}"
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


def generate_answer(query, related_context, docs=None):
    """生成回答，使用 Fast/Slow path 機制
    
    Args:
        query: 用戶問題
        related_context: 檢索到的相關文本
        docs: 檢索到的文檔列表（可選）
        
    Returns:
        str: 生成的回答
    """
    # 如果有檢索結果，先用 Fast path 評估
    if docs:
        print("🚀 Fast path: 評估檢索結果...")
        # 使用 SAS 評估每個檢索結果
        _, probs = predict_pos_prob(
            sas_model,
            [query] * len(docs),
            [doc.page_content for doc in docs],
            temperature=SAS_PARAMS.get("temperature", 2.0),
            penalty=SAS_PARAMS.get("penalty", -1.0)
        )
        
        # 檢查是否有段落通過高門檻
        high_thr = SAS_PARAMS.get("high_threshold", 0.6)
        passed_indices = np.where(probs >= high_thr)[0]
        
        if len(passed_indices) > 0:
            print(f"✅ 找到 {len(passed_indices)} 個通過高門檻的段落")
            # 選擇最多3個最高分的段落
            top_indices = passed_indices[np.argsort(-probs[passed_indices])[:3]]
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
            temperature=SAS_PARAMS.get("temperature", 2.0),
            penalty=SAS_PARAMS.get("penalty", -1.0)
        )
        
        # 收集通過低門檻的段落
        passed_indices = np.where(probs >= low_thr)[0]
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
        temperature=SAS_PARAMS.get("temperature", 2.0),
        penalty=SAS_PARAMS.get("penalty", -1.0)
    )
    
    # 如果最終答案未通過高門檻，建議諮詢醫師
    if probs[0] < SAS_PARAMS.get("high_threshold", 0.6):
        return "這個問題需要更多專業資訊才能完整回答，建議您諮詢主治醫師。"
    
    return answer


def clean_markdown(text):
    """
    去除 Gemini AI 生成的 Markdown 標記，例如 **加粗**、*斜體*
    """
    return re.sub(r"[\*\_]", "", text).strip()


def extract_food_names_english(text):
    """
    從 Gemini Vision 生成的描述中擷取 **英文** 的食物名稱，回傳 list
    """
    extraction_prompt = f"""請從以下文字中找出 **所有主要的食物名稱（英文）**：
{text}

**輸出格式：**
- 只回傳英文食物名稱，不要其他描述或多餘的詞彙。
- 如果有多個食物，請用逗號分隔，例如：「apple, banana, sandwich」。
- 例如：
  「圖片顯示一個漢堡和薯條」 → 「burger, fries」
  「這是一碗白飯和一塊雞肉」 → 「rice, chicken」
"""

    response = model.generate_content(extraction_prompt)

    if not response or not hasattr(response, "text"):
        logging.error("Gemini AI 未能擷取食物名稱")
        return None

    # 取得 AI 回傳的文字並拆分為 list
    food_text = response.text.strip()
    food_list = [food.strip().lower() for food in food_text.split(",")]

    return food_list if food_list else None


def analyze_food_with_gemini(image_path):
    """
    1️⃣ 使用 Gemini Vision 擷取 **英文** 食物名稱
    2️⃣ 使用 FatSecret API 查詢營養資訊
    3️⃣ 使用 Gemini 解析營養數據（英文），然後翻譯成繁體中文
    4️⃣ 輸出 簡潔且易讀 的結果
    """
    try:
        # **讀取圖片並轉換為 Base64**
        with Image.open(image_path) as image:
            buffered = io.BytesIO()
            image_format = image.format
            image.save(buffered, format=image_format)
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # **Gemini Vision 分析圖片內容**
        vision_prompt = """請擷取圖片中 所有主要的食物名稱（英文），用逗號分隔，例如：
"apple, banana, sandwich"
"""

        vision_response = model.generate_content(
            [{"mime_type": f"image/{image_format.lower()}", "data": image_base64}, vision_prompt]
        )

        # **檢查 Vision AI 回應**
        if not vision_response or not hasattr(vision_response, "text"):
            logging.error("Gemini Vision AI 未回傳有效的結果")
            return TextSendMessage(text="⚠️ 無法辨識圖片，請試試另一張！")

        food_list = [food.strip().lower() for food in vision_response.text.strip().split(",")]

        if not food_list:
            return TextSendMessage(text="⚠️ 無法識別主要食物，請提供更清晰的圖片！")

        logging.info(f"🔍 提取的食物名稱: {food_list}")

        # **查詢 FatSecret API 並分析**
        all_results = []
        nutrition_data_list = []
        food_chinese_names = []
        api_data_found = False  # 標記是否找到API數據
        food_english_names = []  # 保存成功查詢到的英文食物名稱，用於後續詳細信息查詢

        for food in food_list:
            nutrition_data = search_food_with_fatsecret(food)

            # 確保 API 回傳的數據是字典
            if not isinstance(nutrition_data, dict):
                logging.error(f"FatSecret API 回傳錯誤數據: {nutrition_data}")
                continue  # 跳過錯誤數據

            food_chinese_name = translate_to_chinese(food.capitalize())
            food_chinese_names.append(food_chinese_name)

            # 添加食物名稱到營養數據中，用於後續處理
            nutrition_data["food_name"] = food
            nutrition_data["food_chinese_name"] = food_chinese_name
            food_english_names.append(food)

            # 分析並取得優點、風險、建議
            analysis_data = analyze_nutrition_for_flex(nutrition_data)

            # 保存營養數據，供後續生成 FlexMessage 使用
            nutrition_data_list.append(nutrition_data)

            # 檢查是否找到有效的API數據
            if "calories" in nutrition_data and nutrition_data.get("calories"):
                api_data_found = True

            # **格式化輸出**
            formatted_result = f"""
📊 {food_chinese_name} 的營養資訊
🔥 卡路里: {nutrition_data.get('calories', 'N/A')} kcal
🍞 碳水化合物: {nutrition_data.get('carbohydrate', 'N/A')} g
🍗 蛋白質: {nutrition_data.get('protein', 'N/A')} g
🥑 脂肪: {nutrition_data.get('fat', 'N/A')} g
🍬 糖: {nutrition_data.get('sugar', 'N/A')} g
🌾 纖維: {nutrition_data.get('fiber', 'N/A')} g
🧂 鈉: {nutrition_data.get('sodium', 'N/A')} mg
"""
            all_results.append(formatted_result.strip())

        # 計算熱量來源佔比
        calorie_sources = calculate_calorie_sources(nutrition_data_list)

        # 檢查營養數據是否為空
        if not nutrition_data_list:
            return TextSendMessage(text="⚠️ 無法獲取食物的營養資訊，請稍後再試。")

        # 將查詢狀態添加到全局數據存儲中，用於詳細頁面顯示
        global_data_store[",".join(food_english_names)] = {
            "api_data_found": api_data_found,
            "nutrition_data_list": nutrition_data_list,
            "food_chinese_names": food_chinese_names,
        }

        # 更新熱量來源數據，添加API數據標記
        calorie_sources["is_estimated"] = not api_data_found

        # 創建詳細營養數據結構（兼容新的緊湊型設計）
        detailed_nutrition = {
            "total_calories": calorie_sources.get("total_calories", 0),
            "carbs": {
                "total": sum([float(data.get("carbohydrate", 0) or 0) for data in nutrition_data_list]),
                "sugar": {"total": sum([float(data.get("sugar", 0) or 0) for data in nutrition_data_list])},
            },
            "protein": {"total": sum([float(data.get("protein", 0) or 0) for data in nutrition_data_list])},
            "fat": {"total": sum([float(data.get("fat", 0) or 0) for data in nutrition_data_list])},
        }

        # 使用原始的熱量來源分析 Flex Message
        flex_message = generate_calorie_source_flex_message(food_chinese_names, calorie_sources)

        # 確保返回的是 LINE 的消息對象
        if isinstance(flex_message, dict):
            # 如果是字典，轉換為 FlexSendMessage
            return FlexSendMessage(alt_text=f"{food_chinese_names[0]} 的熱量來源分析", contents=flex_message)
        else:
            # 如果已經是 FlexSendMessage 或其他 LINE 消息對象，直接返回
            return flex_message

    except Exception as e:
        logging.error(f"🚨 圖片分析時發生錯誤: {str(e)}")
        return TextSendMessage(text="⚠️ 無法分析圖片，請稍後再試。")


def analyze_nutrition_for_flex(nutrition_data):
    """
    分析營養數據，提取優點、風險和建議，以便生成 FlexMessage
    """
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
        # 確保只提取 JSON 部分
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
    """
    計算熱量來源佔比（碳水化合物、蛋白質、脂肪、糖分）
    """
    total_carb_calories = 0
    total_protein_calories = 0
    total_fat_calories = 0
    total_sugar_calories = 0  # 新增糖分熱量計算
    total_calories = 0

    # 熱量換算：碳水4卡/克，蛋白質4卡/克，脂肪9卡/克，糖分4卡/克
    for data in nutrition_data_list:
        carb = float(data.get("carbohydrate", 0) or 0)
        protein = float(data.get("protein", 0) or 0)
        fat = float(data.get("fat", 0) or 0)
        sugar = float(data.get("sugar", 0) or 0)  # 獲取糖分含量

        carb_cal = carb * 4
        protein_cal = protein * 4
        fat_cal = fat * 9
        sugar_cal = sugar * 4  # 糖分熱量計算（同碳水化合物）

        total_carb_calories += carb_cal
        total_protein_calories += protein_cal
        total_fat_calories += fat_cal
        total_sugar_calories += sugar_cal  # 累加糖分熱量
        total_calories += float(data.get("calories", 0) or 0)

    # 計算佔比
    if total_calories > 0:
        carb_percentage = (total_carb_calories / total_calories) * 100
        protein_percentage = (total_protein_calories / total_calories) * 100
        fat_percentage = (total_fat_calories / total_calories) * 100
        sugar_percentage = (total_sugar_calories / total_calories) * 100  # 計算糖分比例
    else:
        # 如果沒有熱量資訊，使用大語言模型尋找建議值
        food_names = []
        for data in nutrition_data_list:
            if "food_name" in data and data["food_name"]:
                food_names.append(data["food_name"])

        # 如果有食物名稱，使用大語言模型估算
        if food_names:
            estimated_values = estimate_nutrition_with_gemini(food_names)
            total_calories = estimated_values.get("total_calories", 100)
            total_carb_calories = estimated_values.get("carbs_calories", 50)
            total_protein_calories = estimated_values.get("protein_calories", 20)
            total_fat_calories = estimated_values.get("fat_calories", 30)
            total_sugar_calories = estimated_values.get("sugar_calories", 10)
        else:
            # 若無法獲取食物名稱，使用預設值
            total_calories = 100
            total_carb_calories = 50
            total_protein_calories = 20
            total_fat_calories = 30
            total_sugar_calories = 10

    return {
        "carbs_calories": round(total_carb_calories, 0),  # 改為直接返回熱量值而非百分比
        "protein_calories": round(total_protein_calories, 0),
        "fat_calories": round(total_fat_calories, 0),
        "sugar_calories": round(total_sugar_calories, 0),  # 添加糖分熱量值
        "total_calories": round(total_calories, 0),
        "is_estimated": total_calories == 0,  # 添加標記，表示是否為估算值
    }


def estimate_nutrition_with_gemini(food_names):
    """
    使用Gemini獲取食物的估計營養成分

    Args:
        food_names: 食物名稱列表

    Returns:
        包含估計營養值的字典
    """
    # 組合所有食物名稱
    food_list = "、".join(food_names)

    # 構建提示詞
    prompt = f"""請根據營養學知識，估算以下食物的大致熱量來源分佈：{food_list}

請提供以下信息的估計值：
1. 總熱量（大卡）
2. 碳水化合物熱量（大卡）
3. 蛋白質熱量（大卡）
4. 脂肪熱量（大卡）
5. 糖分熱量（大卡）

請使用以下JSON格式回應：
{{"total_calories": 數值, "carbs_calories": 數值, "protein_calories": 數值, "fat_calories": 數值, "sugar_calories": 數值}}

注意：這些只是估計值，非精確數據。
"""

    try:
        # 呼叫Gemini模型
        response = model.generate_content(prompt)

        if not response or not hasattr(response, "text"):
            return {
                "total_calories": 100,
                "carbs_calories": 50,
                "protein_calories": 20,
                "fat_calories": 30,
                "sugar_calories": 10,
            }

        # 從回應中提取JSON
        result_text = response.text.strip()
        # 尋找JSON部分
        match = re.search(r"(\{.*\})", result_text, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                estimated_values = json.loads(json_str)
                # 確保所有必要的鍵存在
                required_keys = [
                    "total_calories",
                    "carbs_calories",
                    "protein_calories",
                    "fat_calories",
                    "sugar_calories",
                ]
                for key in required_keys:
                    if key not in estimated_values:
                        estimated_values[key] = 0
                return estimated_values
            except:
                # JSON解析失敗，返回預設值
                return {
                    "total_calories": 100,
                    "carbs_calories": 50,
                    "protein_calories": 20,
                    "fat_calories": 30,
                    "sugar_calories": 10,
                }

        # 未找到有效JSON，返回預設值
        return {
            "total_calories": 100,
            "carbs_calories": 50,
            "protein_calories": 20,
            "fat_calories": 30,
            "sugar_calories": 10,
        }
    except Exception as e:
        print(f"估算營養值時出錯: {str(e)}")
        return {
            "total_calories": 100,
            "carbs_calories": 50,
            "protein_calories": 20,
            "fat_calories": 30,
            "sugar_calories": 10,
        }


def translate_to_chinese(english_text):
    """
    翻譯分析結果為繁體中文
    """
    translation_prompt = f"""請將以下內容翻譯為繁體中文，精準翻譯，只回傳食物名稱，不要其他描述或多餘的詞彙。
{english_text}
"""

    response = model.generate_content(translation_prompt)

    if not response or not hasattr(response, "text"):
        logging.error("Gemini AI 未回傳翻譯結果")
        return "⚠️ 無法翻譯分析結果，請稍後再試。"

    return response.text.strip()


# Flask app setup
app = Flask(__name__)
port = 5000

# LINE Bot setup
line_bot_api = LineBotApi(LINE_ACCESS_TOKEN)
handler = WebhookHandler(LINE_SECRET)

# 測試 LINE API 連線（啟動時發送一條測試訊息）
try:
    print("✅ Testing LINE API connection")
    # 這裡可以添加測試訊息，但需要有效的 user_id
    print("✅ LINE API setup completed")
except Exception as e:
    print(f"❌ LINE API setup error: {str(e)}")


# 健康檢查路由
@app.route("/health", methods=["GET"])
def health_check():
    return "OK", 200


# 支援原本的 callback 路由
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    print(f"✅ Server time: {datetime.now(pytz.timezone('Asia/Taipei')).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✅ Received request with signature: {signature}")

    # 手動計算簽名驗證
    hash = hmac.new(LINE_SECRET.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).digest()
    calculated_signature = base64.b64encode(hash).decode("utf-8")

    if signature != calculated_signature:
        print(f"❌ Signature mismatch! Expected: {calculated_signature}, Received: {signature}")
        return "Invalid signature", 400

    try:
        handler.handle(body, signature)
    except InvalidSignatureError as e:
        print(f"❌ InvalidSignatureError: {str(e)}")
        return "Invalid signature", 400

    print("✅ Returning HTTP 200 response")
    return "OK", 200


@app.route("/", methods=["POST"])
def linebot():
    body = request.get_data(as_text=True)
    signature = request.headers.get("X-Line-Signature", "")

    try:
        # 使用 handler 處理所有事件
        handler.handle(body, signature)
        return "OK"
    except InvalidSignatureError:
        print("❌ Invalid signature")
        return "Invalid signature", 400
    except Exception as e:
        print(f"❌ Error in linebot: {str(e)}")
        print(f"Raw Body: {body}")
        return "OK"


# 註冊事件處理函數 - 使用 handler 處理訊息
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    """處理文字訊息事件"""
    print(f"✅ 收到訊息：{event.message.text}")
    user_id = event.source.user_id
    message_text = event.message.text.strip()

    try:
        # 🩸 最優先處理血糖相關狀態 - 必須在其他任何處理之前
        print(f"🔍 檢查用戶狀態: user_id={user_id}, states={user_states.get(user_id)}")

        # 處理血糖值輸入狀態
        if user_states.get(user_id) == "waiting_for_bloodsugar":
            print(f"🩸 檢測到血糖輸入狀態，用戶 {user_id} 輸入: {message_text}")
            if BLOOD_SUGAR_AVAILABLE:
                try:
                    blood_sugar_value = int(message_text)
                    print(f"🩸 嘗試記錄血糖值: {blood_sugar_value}")
                    response_text = record_blood_sugar(user_id, blood_sugar_value)
                    print(f"🩸 血糖記錄結果: {response_text}")

                    if response_text.startswith("✅"):
                        tz = pytz.timezone("Asia/Taipei")
                        today = datetime.now(tz).strftime("%Y-%m-%d")
                        today_records_message = create_blood_sugar_message(user_id, today)
                        final_message = TextSendMessage(
                            text=f"已記錄！\n-------------\n{today_records_message.text}",
                            quick_reply=today_records_message.quick_reply,
                        )
                        if user_id in user_states:
                            del user_states[user_id]  # 清除狀態
                        print(f"🩸 血糖記錄成功，清除用戶狀態")
                        line_bot_api.reply_message(event.reply_token, final_message)
                    else:
                        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=response_text))
                    return  # 重要：處理完血糖輸入後直接返回
                except ValueError:
                    print(f"❌ 血糖值格式錯誤: {message_text}")
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 請輸入有效的數字！"))
                    return  # 重要：處理完錯誤後直接返回
            else:
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 血糖記錄功能暫時不可用"))
                return  # 重要：處理完後直接返回

        # 處理血糖值修改狀態
        elif user_states.get(user_id) and isinstance(user_states.get(user_id), dict) and user_states[user_id].get("state") == "editing_bloodsugar":
            print(f"🩸 檢測到血糖修改狀態，用戶 {user_id} 輸入: {message_text}")
            if BLOOD_SUGAR_AVAILABLE:
                try:
                    new_value = int(message_text)
                    state = user_states[user_id]
                    date_str = state["date"]
                    record_index = state["index"]
                    print(f"🩸 嘗試修改血糖值: 日期={date_str}, 索引={record_index}, 新值={new_value}")
                    response_text = update_blood_sugar(user_id, date_str, record_index, new_value)
                    print(f"🩸 血糖修改結果: {response_text}")

                    if response_text.startswith("✅"):
                        today_records_message = create_blood_sugar_message(user_id, date_str)
                        final_message = TextSendMessage(
                            text=f"已修改！\n-------------\n{today_records_message.text}",
                            quick_reply=today_records_message.quick_reply,
                        )
                        if user_id in user_states:
                            del user_states[user_id]  # 清除狀態
                        print(f"🩸 血糖修改成功，清除用戶狀態")
                        line_bot_api.reply_message(event.reply_token, final_message)
                    else:
                        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=response_text))
                    return  # 重要：處理完血糖修改後直接返回
                except ValueError:
                    print(f"❌ 血糖值格式錯誤: {message_text}")
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 請輸入有效的數字！"))
                    return  # 重要：處理完錯誤後直接返回
            else:
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 血糖記錄功能暫時不可用"))
                return  # 重要：處理完後直接返回

        # 🩺 處理血糖功能觸發訊息 - 在選單過濾之前處理
        if message_text in ["我來輸入血糖值囉～", "我來輸入血糖值囉~"]:
            tz = pytz.timezone("Asia/Taipei")
            today = datetime.now(tz).strftime("%Y-%m-%d")
            message = create_blood_sugar_message(user_id, today)
            line_bot_api.reply_message(event.reply_token, message)
            return

        elif message_text == "我看看最近的血糖變化，有沒有進步啊！":
            message = create_report_menu_message()
            line_bot_api.reply_message(event.reply_token, message)
            return

        # 📋 選單訊息過濾 - 忽略自動選單選項 (支援模糊匹配)
        menu_keywords = [
            "查看我的血糖紀錄",
            "語音轉文字",
            "詢問飲食建議",
            "我來輸入血糖值囉～",
            "我來輸入血糖值囉~",
            "我看看最近的血糖變化，有沒有進步啊！",
            "可以幫我看一下這餐的熱量多少嗎",
            "生活習慣也會影響血糖嗎",
            "吃藥的事我有點搞不清楚",
            "今天可以吃點什麼不會讓血糖飆高啊",
            "請糖小護回答我的問題吧",
            "小護您好！我想知道關於血糖管理室的資訊~",  # 支援模糊匹配血糖中心/血糖管理室
        ]

        # 檢查是否為選單訊息 (使用模糊匹配)
        is_menu_message = False
        for keyword in menu_keywords:
            if keyword in message_text:
                is_menu_message = True
                print(f"🔇 忽略選單訊息 (匹配關鍵字: {keyword}): {message_text}")
                break

        # 處理語音轉文字功能
        if "語音轉文字" in message_text:
            message = create_voice_input_message()
            line_bot_api.reply_message(event.reply_token, message)
            return

        # 處理飲食建議功能
        if "今天可以吃點什麼不會讓血糖飆高啊" in message_text:
            message = create_diet_advice_message()
            line_bot_api.reply_message(event.reply_token, message)
            return

        # 處理用藥建議功能
        if "吃藥的事我有點搞不清楚" in message_text:
            message = create_medication_advice_message()
            line_bot_api.reply_message(event.reply_token, message)
            return

        # 處理生活習慣建議功能
        if "生活習慣也會影響血糖嗎" in message_text:
            message = create_lifestyle_advice_message()
            line_bot_api.reply_message(event.reply_token, message)
            return

        # 處理熱量分析功能
        if "可以幫我看一下這餐的熱量多少嗎" in message_text:
            message = create_calorie_analysis_message()
            line_bot_api.reply_message(event.reply_token, message)
            return



        # 如果是選單訊息，直接返回不回應
        if is_menu_message:
            return

        # 🤖 預設處理文字訊息 - 營養諮詢
        print(f"💬 處理一般文字訊息: {message_text}")
        related_context, docs = search_related_content(retriever, message_text)
        response = generate_answer(message_text, related_context, docs)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=response))

    except LineBotApiError as e:
        print(f"❌ Failed to reply message: {str(e)}")
    except Exception as e:
        print(f"❌ Error in handle_message: {str(e)}")
        import traceback

        traceback.print_exc()


# 添加圖片訊息處理器
@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    """處理圖片訊息事件"""
    print(f"✅ 收到圖片訊息")
    user_id = event.source.user_id

    try:
        # 獲取圖片 ID
        image_id = event.message.id

        # 下載圖片
        image_url = f"https://api-data.line.me/v2/bot/message/{image_id}/content"
        headers = {"Authorization": f"Bearer {LINE_ACCESS_TOKEN}"}
        response = requests.get(image_url, headers=headers, stream=True)

        if response.status_code == 200:
            # 立即回覆分析中的訊息
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text="📸 照片分析中，請稍候...\n🔍 正在識別食物並查詢營養資訊")
            )

            image_path = f"temp_{image_id}.jpg"
            with open(image_path, "wb") as f:
                for chunk in response.iter_content():
                    f.write(chunk)

            # 傳給 Gemini AI 進行分析
            result = analyze_food_with_gemini(image_path)

            # 檢查結果類型並回應
            if isinstance(result, str):
                push_message = TextSendMessage(text=result)
            else:
                # 回傳 FlexMessage（已經是分析後的結果）
                push_message = result

            # 使用 push_message 發送分析結果
            line_bot_api.push_message(user_id, push_message)

            # 清理暫存檔案
            try:
                os.remove(image_path)
            except:
                pass
        else:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="無法下載圖片，請稍後再試。"))

    except LineBotApiError as e:
        print(f"❌ Failed to reply image message: {str(e)}")
    except Exception as e:
        print(f"❌ Error in handle_image_message: {str(e)}")


# 添加語音訊息處理器
@handler.add(MessageEvent, message=AudioMessage)
def handle_audio_message(event):
    """處理語音訊息事件"""
    print(f"✅ 收到語音訊息")
    user_id = event.source.user_id

    try:
        # 獲取語音訊息 ID
        audio_id = event.message.id
        duration = event.message.duration  # 語音長度（毫秒）

        # 檢查語音長度（超過60秒的語音可能處理困難）
        if duration > 60000:  # 60秒
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text="❌ 語音訊息過長（超過60秒），請重新錄製較短的語音訊息")
            )
            return

        # 下載語音文件
        audio_url = f"https://api-data.line.me/v2/bot/message/{audio_id}/content"
        headers = {"Authorization": f"Bearer {LINE_ACCESS_TOKEN}"}
        response = requests.get(audio_url, headers=headers, stream=True)

        if response.status_code == 200:
            # 保存語音文件
            audio_path = f"temp_audio_{audio_id}.m4a"
            with open(audio_path, "wb") as f:
                for chunk in response.iter_content():
                    f.write(chunk)

            # 先發送處理中訊息
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="🎤 正在處理您的語音訊息，請稍候..."))

            # 使用 Gemini AI 轉換語音為文字
            converted_text = process_audio_with_gemini(audio_path)

            # 如果轉換成功，進一步處理文字內容
            if not converted_text.startswith("❌"):
                # 搜尋相關內容並生成回答
                related_context = search_related_content(retriever, converted_text)
                ai_response = generate_answer(converted_text, related_context)

                # 組合回應訊息
                final_response = f"🎤 語音轉文字結果：\n「{converted_text}」\n\n📝 糖小護的回答：\n{ai_response}"
            else:
                final_response = converted_text

            # 發送最終回應
            line_bot_api.push_message(user_id, TextSendMessage(text=final_response))

            # 清理暫存檔案
            try:
                os.remove(audio_path)
                print(f"✅ 已清理暫存音頻檔案：{audio_path}")
            except:
                pass

        else:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 無法下載語音檔案，請稍後再試"))

    except LineBotApiError as e:
        print(f"❌ Failed to reply audio message: {str(e)}")
    except Exception as e:
        print(f"❌ Error in handle_audio_message: {str(e)}")
        try:
            line_bot_api.push_message(user_id, TextSendMessage(text=f"❌ 語音處理失敗：{str(e)}"))
        except:
            pass


@handler.add(PostbackEvent)
def handle_postback(event):
    """處理 Postback 事件"""
    user_id = event.source.user_id
    postback_data = event.postback.data
    print(f"✅ Handling postback: {postback_data}")

    try:
        # 處理語音輸入相關的 Postback
        if postback_data == "action=start_voice_recording":
            message = TextSendMessage(
                text="🎤 語音錄製指南\n\n📱 **手機操作步驟：**\n\n1️⃣ 點擊聊天室下方的「+」按鈕\n2️⃣ 選擇「語音訊息」或麥克風圖示\n3️⃣ 按住錄音按鈕開始說話\n4️⃣ 放開按鈕完成錄音\n5️⃣ 點擊發送按鈕\n\n💡 **錄音小技巧：**\n• 在安靜環境下錄音效果更好\n• 說話清晰，語速適中\n• 建議錄音時間在10-30秒內\n\n✨ 錄音完成後，我會自動將語音轉為文字並提供專業建議！"
            )
            line_bot_api.reply_message(event.reply_token, message)
            return

        elif postback_data == "action=voice_tutorial":
            message = TextSendMessage(
                text="📱 **語音功能使用教學**\n\n🎯 **兩種使用方式：**\n\n**方式1：直接發送語音**\n• 打開LINE聊天室\n• 點擊「+」→「語音訊息」\n• 錄製並發送\n\n**方式2：透過選單**\n• 輸入「語音轉文字」\n• 點擊「🎤 開始語音錄製」\n• 按照指示操作\n\n🔧 **支援功能：**\n✅ 語音轉文字\n✅ 糖尿病問題諮詢\n✅ 飲食建議查詢\n✅ 藥物使用指導\n\n❓ 有問題嗎？直接發送語音訊息試試看！"
            )
            line_bot_api.reply_message(event.reply_token, message)
            return

        elif postback_data == "action=start_voice_input":
            # 保持舊版本相容性
            message = TextSendMessage(
                text="🎤 請現在發送語音訊息給我！\n\n💡 使用方法：\n1. 點擊LINE聊天室下方的「+」按鈕\n2. 選擇「語音訊息」\n3. 按住錄音按鈕開始說話\n4. 放開按鈕完成錄音\n5. 發送語音訊息\n\n✨ 我會自動將您的語音轉換為文字，並提供相關的糖尿病建議！"
            )
            line_bot_api.reply_message(event.reply_token, message)
            return

        elif postback_data == "action=cancel_voice_input":
            message = TextSendMessage(text="❌ 已取消語音輸入功能")
            line_bot_api.reply_message(event.reply_token, message)
            return

        # 處理緊湊型 Flex Message 中的「詳細資訊」按鈕
        elif postback_data.startswith("detailed_nutrition:"):
            food_names = postback_data.split(":", 1)[1].split(",")

            # 檢查是否有對應的英文食物名稱
            found = False
            for key in global_data_store.keys():
                key_foods = key.split(",")
                # 如果中文名與英文名的數量一致，則嘗試匹配
                if len(key_foods) == len(food_names):
                    stored_chinese_names = global_data_store[key].get("food_chinese_names", [])
                    if all(name in stored_chinese_names for name in food_names):
                        # 找到匹配的英文鍵
                        food_key = key
                        found = True
                        break

            # 如果沒找到匹配的英文鍵，則直接使用中文名
            if not found:
                food_key = ",".join(food_names)

            # 生成詳細營養資訊 FlexMessage
            detailed_flex = generate_detailed_nutrition_flex(food_names, food_key)
            line_bot_api.reply_message(event.reply_token, detailed_flex)

        # 處理緊湊型 Flex Message 中的「糖尿病建議」按鈕
        elif postback_data.startswith("diabetes_advice:"):
            food_names = postback_data.split(":", 1)[1].split(",")
            food_list = "、".join(food_names)

            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text="🔍 正在為您分析糖尿病飲食建議，請稍候...")
            )
            query = f"針對糖尿病患者，食用{food_list}的建議和注意事項"
            related_context = search_related_content(retriever, query)
            response = generate_answer(query, related_context)

            # 使用簡單的文字回應
            cleaned_response = response.replace("*", "").replace("**", "")
            line_bot_api.push_message(user_id, TextSendMessage(text=f"🩺 糖尿病飲食建議：\n\n{cleaned_response}"))

        # 處理糖尿病建議 Flex Message 中的「查看完整建議」按鈕
        elif postback_data.startswith("full_advice:"):
            food_names = postback_data.split(":", 1)[1].split(",")
            food_list = "、".join(food_names)

            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text="🔍 正在為您查詢完整的糖尿病飲食建議...")
            )
            query = f"針對糖尿病患者，詳細說明食用{food_list}的完整建議、注意事項、食用方法和搭配建議"
            related_context = search_related_content(retriever, query)
            response = generate_answer(query, related_context)

            # 清理回應文字
            cleaned_response = response.replace("*", "").replace("**", "")
            line_bot_api.push_message(user_id, TextSendMessage(text=f"📋 完整糖尿病飲食建議：\n\n{cleaned_response}"))

        # 處理糖尿病建議 Flex Message 中的「相關食物建議」按鈕
        elif postback_data.startswith("related_foods:"):
            food_names = postback_data.split(":", 1)[1].split(",")
            food_list = "、".join(food_names)

            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="🔍 正在為您查詢相關食物建議..."))
            query = f"推薦與{food_list}類似且適合糖尿病患者的其他食物選擇"
            related_context = search_related_content(retriever, query)
            response = generate_answer(query, related_context)

            # 清理回應文字
            cleaned_response = response.replace("*", "").replace("**", "")
            line_bot_api.push_message(user_id, TextSendMessage(text=f"🔍 相關食物建議：\n\n{cleaned_response}"))

        # 處理「查看完整熱量來源分析」按鈕（保持向後兼容）
        elif postback_data.startswith("detailed_calorie_source:"):
            food_names = postback_data.split(":", 1)[1].split(",")

            # 檢查是否有對應的英文食物名稱
            found = False
            for key in global_data_store.keys():
                key_foods = key.split(",")
                # 如果中文名與英文名的數量一致，則嘗試匹配
                if len(key_foods) == len(food_names):
                    stored_chinese_names = global_data_store[key].get("food_chinese_names", [])
                    if all(name in stored_chinese_names for name in food_names):
                        # 找到匹配的英文鍵
                        food_key = key
                        found = True
                        break

            # 如果沒找到匹配的英文鍵，則直接使用中文名
            if not found:
                food_key = ",".join(food_names)

            detailed_analysis = generate_detailed_nutrition_flex(food_names, food_key)
            line_bot_api.reply_message(event.reply_token, detailed_analysis)
            return

        # 血糖記錄相關的 Postback 處理
        elif postback_data == "action=select_date":
            selected_date = event.postback.params.get("date")
            if not selected_date:
                message = TextSendMessage(text="❌ 請選擇一個日期！")
            else:
                message = create_blood_sugar_message(user_id, selected_date)
            line_bot_api.reply_message(event.reply_token, message)

        elif postback_data == "action=add_blood_sugar":
            user_states[user_id] = "waiting_for_bloodsugar"
            print(f"🩸 設置用戶 {user_id} 為血糖輸入狀態")
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="請輸入血糖值"))

        elif postback_data == "action=edit_blood_sugar":
            tz = pytz.timezone("Asia/Taipei")
            today = datetime.now(tz).strftime("%Y-%m-%d")
            message = show_records_for_edit(user_id, today)
            line_bot_api.reply_message(event.reply_token, message)

        elif postback_data == "action=delete_blood_sugar":
            tz = pytz.timezone("Asia/Taipei")
            today = datetime.now(tz).strftime("%Y-%m-%d")
            message = show_records_for_delete(user_id, today)
            line_bot_api.reply_message(event.reply_token, message)

        elif postback_data.startswith("action=edit_record"):
            import re

            index = int(re.search(r"index=(\d+)", postback_data).group(1))
            tz = pytz.timezone("Asia/Taipei")
            today = datetime.now(tz).strftime("%Y-%m-%d")
            user_states[user_id] = {"state": "editing_bloodsugar", "date": today, "index": index}
            print(f"🩸 設置用戶 {user_id} 為血糖修改狀態: 日期={today}, 索引={index}")
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="請輸入新的血糖值"))

        elif postback_data.startswith("action=delete_record"):
            import re

            index = int(re.search(r"index=(\d+)", postback_data).group(1))
            tz = pytz.timezone("Asia/Taipei")
            today = datetime.now(tz).strftime("%Y-%m-%d")
            if BLOOD_SUGAR_AVAILABLE:
                response_text = delete_blood_sugar(user_id, today, index)
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=response_text))
            else:
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 血糖記錄功能暫時不可用"))

        elif postback_data == "action=report_today":
            if BLOOD_SUGAR_AVAILABLE:
                try:
                    result = handle_blood_sugar_report(user_id, "today")
                    if isinstance(result, list):  # 多個訊息
                        line_bot_api.reply_message(event.reply_token, result)
                    else:
                        line_bot_api.reply_message(event.reply_token, result)
                except Exception as e:
                    line_bot_api.reply_message(
                        event.reply_token, TextSendMessage(text=f"❌ 無法生成今日報表：{str(e)}")
                    )
            else:
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 血糖報表功能暫時不可用"))

        elif postback_data == "action=report_last_week":
            if BLOOD_SUGAR_AVAILABLE:
                try:
                    result = handle_blood_sugar_report(user_id, "week")
                    if isinstance(result, list):  # 多個訊息
                        line_bot_api.reply_message(event.reply_token, result)
                    else:
                        line_bot_api.reply_message(event.reply_token, result)
                except Exception as e:
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"❌ 無法生成週報表：{str(e)}"))
            else:
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 血糖報表功能暫時不可用"))

        elif postback_data == "action=report_trend_week":
            if BLOOD_SUGAR_AVAILABLE:
                try:
                    # 先發送處理中訊息
                    line_bot_api.reply_message(
                        event.reply_token, TextSendMessage(text="📊 正在生成週趨勢分析報表，請稍候...")
                    )

                    result = handle_blood_sugar_report(user_id, "trend_week")
                    if isinstance(result, list):  # 多個訊息
                        for message in result:
                            line_bot_api.push_message(user_id, message)
                    else:
                        line_bot_api.push_message(user_id, result)
                except Exception as e:
                    line_bot_api.push_message(user_id, TextSendMessage(text=f"❌ 無法生成趨勢分析：{str(e)}"))
            else:
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 血糖報表功能暫時不可用"))

        # 處理日期選擇的報表
        elif postback_data.startswith("action=report_select_date"):
            if BLOOD_SUGAR_AVAILABLE:
                # 從 postback 資料中取得選擇的日期
                date_str = event.postback.params.get("date")  # LINE SDK 會將日期放在 params 中
                if date_str:
                    try:
                        result = handle_blood_sugar_report(user_id, "date", date_str)
                        if isinstance(result, list):  # 多個訊息
                            line_bot_api.reply_message(event.reply_token, result)
                        else:
                            line_bot_api.reply_message(event.reply_token, result)
                    except Exception as e:
                        line_bot_api.reply_message(
                            event.reply_token, TextSendMessage(text=f"❌ 無法生成{date_str}報表：{str(e)}")
                        )
                else:
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 日期選擇錯誤"))
            else:
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 血糖報表功能暫時不可用"))

        # 處理血糖趨勢日期選擇 - 調用 blood_sugar.py 的報表功能
        elif postback_data.startswith("action=blood_sugar_trend_date"):
            if BLOOD_SUGAR_AVAILABLE:
                # 從 postback 資料中取得選擇的日期
                date_str = event.postback.params.get("date")
                if date_str:
                    try:
                        # 先回覆處理中訊息
                        line_bot_api.reply_message(
                            event.reply_token, 
                            TextSendMessage(text=f"📊 正在生成 {date_str} 的血糖報表，請稍候...")
                        )
                        
                        # 調用 blood_sugar.py 的報表功能
                        result = handle_blood_sugar_report(user_id, "date", date_str)
                        if isinstance(result, list):  # 多個訊息
                            for message in result:
                                line_bot_api.push_message(user_id, message)
                        else:
                            line_bot_api.push_message(user_id, result)
                    except Exception as e:
                        line_bot_api.push_message(
                            user_id, TextSendMessage(text=f"❌ 無法生成{date_str}報表：{str(e)}")
                        )
                else:
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 日期選擇錯誤"))
            else:
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 血糖報表功能暫時不可用"))

        # 🍎 飲食建議相關處理
        elif postback_data == "action=low_gi_foods":
            # 先發送處理中訊息
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text="🔍 正在為您查詢低GI食物資訊，請稍候...")
            )
            # 異步處理並發送結果
            query = "推薦適合糖尿病患者的低GI食物"
            related_context = search_related_content(retriever, query)
            response = generate_answer(query, related_context)
            line_bot_api.push_message(user_id, TextSendMessage(text=f"🥗 低GI食物推薦：\n\n{response}"))

        elif postback_data == "action=diabetes_friendly_meals":
            # 查詢血糖友善餐點建議
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text="🔍 正在為您查詢血糖友善餐點，請稍候...")
            )
            query = "推薦適合糖尿病患者的血糖友善餐點"
            related_context = search_related_content(retriever, query)
            response = generate_answer(query, related_context)
            line_bot_api.push_message(user_id, TextSendMessage(text=f"🍽️ 血糖友善餐點：\n\n{response}"))

        elif postback_data == "action=foods_to_avoid":
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text="🔍 正在為您查詢應避免的食物，請稍候...")
            )
            query = "糖尿病患者應該避免的食物"
            related_context = search_related_content(retriever, query)
            response = generate_answer(query, related_context)
            line_bot_api.push_message(user_id, TextSendMessage(text=f"🚫 應避免的食物：\n\n{response}"))

        elif postback_data == "action=meal_timing":
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text="🔍 正在為您查詢進食時間建議，請稍候...")
            )
            query = "糖尿病患者的進食時間建議"
            related_context = search_related_content(retriever, query)
            response = generate_answer(query, related_context)
            line_bot_api.push_message(user_id, TextSendMessage(text=f"⏰ 進食時間建議：\n\n{response}"))

        # 💊 用藥建議相關處理
        elif postback_data == "action=medication_timing":
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text="🔍 正在為您查詢用藥時間資訊，請稍候...")
            )
            query = "糖尿病藥物的最佳服用時間"
            related_context = search_related_content(retriever, query)
            response = generate_answer(query, related_context)
            line_bot_api.push_message(user_id, TextSendMessage(text=f"💊 用藥時間：\n\n{response}"))

        elif postback_data == "action=meal_medication":
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text="🔍 正在為您查詢餐前餐後用藥建議，請稍候...")
            )
            query = "糖尿病藥物餐前餐後服用的建議"
            related_context = search_related_content(retriever, query)
            response = generate_answer(query, related_context)
            line_bot_api.push_message(user_id, TextSendMessage(text=f"🍽️ 餐前餐後用藥：\n\n{response}"))

        elif postback_data == "action=medication_precautions":
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text="🔍 正在為您查詢用藥注意事項，請稍候...")
            )
            query = "糖尿病用藥的注意事項和副作用"
            related_context = search_related_content(retriever, query)
            response = generate_answer(query, related_context)
            line_bot_api.push_message(user_id, TextSendMessage(text=f"⚠️ 用藥注意事項：\n\n{response}"))

        elif postback_data == "action=medication_blood_sugar":
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text="🔍 正在為您查詢藥物與血糖關係，請稍候...")
            )
            query = "糖尿病藥物對血糖的影響"
            related_context = search_related_content(retriever, query)
            response = generate_answer(query, related_context)
            line_bot_api.push_message(user_id, TextSendMessage(text=f"🩸 藥物與血糖：\n\n{response}"))

        # 🍽️ 血糖友善餐點詳細處理（來自 Carousel 按鈕）
        elif postback_data == "action=breakfast_details":
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text="🔍 正在為您查詢詳細早餐建議，請稍候...")
            )
            query = "糖尿病患者適合的早餐食物和搭配建議"
            related_context = search_related_content(retriever, query)
            response = generate_answer(query, related_context)
            line_bot_api.push_message(user_id, TextSendMessage(text=f"🌅 詳細早餐建議：\n\n{response}"))

        elif postback_data == "action=lunch_details":
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text="🔍 正在為您查詢詳細午餐建議，請稍候...")
            )
            query = "糖尿病患者適合的午餐食物和營養搭配"
            related_context = search_related_content(retriever, query)
            response = generate_answer(query, related_context)
            line_bot_api.push_message(user_id, TextSendMessage(text=f"🌞 詳細午餐建議：\n\n{response}"))

        elif postback_data == "action=dinner_details":
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text="🔍 正在為您查詢詳細晚餐建議，請稍候...")
            )
            query = "糖尿病患者適合的晚餐食物和控制血糖的建議"
            related_context = search_related_content(retriever, query)
            response = generate_answer(query, related_context)
            line_bot_api.push_message(user_id, TextSendMessage(text=f"🌙 詳細晚餐建議：\n\n{response}"))

        elif postback_data == "action=snack_details":
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text="🔍 正在為您查詢詳細點心建議，請稍候...")
            )
            query = "糖尿病患者適合的健康點心和零食選擇"
            related_context = search_related_content(retriever, query)
            response = generate_answer(query, related_context)
            line_bot_api.push_message(user_id, TextSendMessage(text=f"🍎 詳細點心建議：\n\n{response}"))

        # 餐點建議選項
        elif postback_data == "action=breakfast_meals":
            query = "推薦適合糖尿病患者的早餐選擇"
            related_context = search_related_content(retriever, query)
            response = generate_answer(query, related_context)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"🌅 早餐建議：\n\n{response}"))

        elif postback_data == "action=lunch_meals":
            query = "推薦適合糖尿病患者的午餐選擇"
            related_context = search_related_content(retriever, query)
            response = generate_answer(query, related_context)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"🌞 午餐建議：\n\n{response}"))

        elif postback_data == "action=dinner_meals":
            query = "推薦適合糖尿病患者的晚餐選擇"
            related_context = search_related_content(retriever, query)
            response = generate_answer(query, related_context)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"🌙 晚餐建議：\n\n{response}"))

        elif postback_data == "action=snack_meals":
            query = "推薦適合糖尿病患者的健康點心"
            related_context = search_related_content(retriever, query)
            response = generate_answer(query, related_context)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"🍎 點心建議：\n\n{response}"))

        elif postback_data == "action=full_meal_plan":
            # 查詢完整餐單建議
            query = "推薦適合糖尿病患者的完整一日餐點規劃"
            related_context = search_related_content(retriever, query)
            response = generate_answer(query, related_context)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"📋 完整餐單規劃：\n\n{response}"))

        # 🌟 生活習慣相關處理
        elif postback_data == "action=exercise_advice":
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="🔍 正在為您查詢運動建議，請稍候..."))
            query = "適合糖尿病患者的運動建議"
            related_context = search_related_content(retriever, query)
            response = generate_answer(query, related_context)
            line_bot_api.push_message(user_id, TextSendMessage(text=f"🏃‍♂️ 運動建議：\n\n{response}"))

        elif postback_data == "action=sleep_advice":
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="🔍 正在為您查詢睡眠建議，請稍候..."))
            query = "睡眠品質對糖尿病血糖的影響"
            related_context = search_related_content(retriever, query)
            response = generate_answer(query, related_context)
            line_bot_api.push_message(user_id, TextSendMessage(text=f"😴 睡眠品質：\n\n{response}"))

        elif postback_data == "action=stress_management":
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text="🔍 正在為您查詢壓力管理方法，請稍候...")
            )
            query = "糖尿病患者的壓力管理方法"
            related_context = search_related_content(retriever, query)
            response = generate_answer(query, related_context)
            line_bot_api.push_message(user_id, TextSendMessage(text=f"😰 壓力管理：\n\n{response}"))

        elif postback_data == "action=quit_smoking_drinking":
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text="🔍 正在為您查詢戒菸戒酒資訊，請稍候...")
            )
            query = "糖尿病患者戒菸戒酒的重要性和方法"
            related_context = search_related_content(retriever, query)
            response = generate_answer(query, related_context)
            line_bot_api.push_message(user_id, TextSendMessage(text=f"🚭 戒菸戒酒：\n\n{response}"))

        # 🔍 熱量分析相關處理
        elif postback_data == "action=manual_input":
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text="請輸入您想查詢的食物名稱，我會為您分析營養成分！")
            )

        elif postback_data == "action=cancel_analysis":
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 已取消熱量分析"))

        # 🎤 語音轉文字功能處理
        elif postback_data == "action=voice_to_text":
            message = create_voice_input_message()
            line_bot_api.reply_message(event.reply_token, message)

        # 📈 血糖趨勢分析相關處理 - 已改為日期選擇模式
        # elif postback_data.startswith("action=blood_sugar_trend"):
        #     # 舊的天數選擇功能已移除，改為日期選擇
        #     pass

        # elif postback_data == "action=show_trend_menu":
        #     # 舊的趨勢選單已移除，改為日期選擇
        #     pass

        else:
            # 如果沒有匹配的處理，使用大語言模型進行一般諮詢
            query = f"糖尿病相關問題：{postback_data}"
            related_context = search_related_content(retriever, query)
            response = generate_answer("請提供糖尿病管理的一般建議", related_context)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=response))

    except LineBotApiError as e:
        print(f"❌ Failed to reply message: {str(e)}")
    except Exception as e:
        print(f"❌ Error in handle_postback: {str(e)}")


def create_diet_advice_message():
    """創建飲食建議選單訊息"""
    try:
        quick_reply = QuickReply(
            items=[
                QuickReplyButton(action=PostbackAction(label="🥗 低GI食物推薦", data="action=low_gi_foods")),
                QuickReplyButton(action=PostbackAction(label="🍽️ 血糖友善餐點", data="action=diabetes_friendly_meals")),
                QuickReplyButton(action=PostbackAction(label="🚫 應避免的食物", data="action=foods_to_avoid")),
                QuickReplyButton(action=PostbackAction(label="⏰ 進食時間建議", data="action=meal_timing")),
            ]
        )
        return TextSendMessage(text="🍎 糖尿病飲食建議\n\n請選擇您想了解的飲食資訊：", quick_reply=quick_reply)
    except Exception as e:
        print(f"❌ Error in create_diet_advice_message: {str(e)}")
        return TextSendMessage(text="❌ 無法顯示飲食建議選單，請稍後再試")


def create_medication_advice_message():
    """創建用藥建議選單訊息"""
    try:
        quick_reply = QuickReply(
            items=[
                QuickReplyButton(action=PostbackAction(label="💊 用藥時間", data="action=medication_timing")),
                QuickReplyButton(action=PostbackAction(label="🍽️ 餐前餐後用藥", data="action=meal_medication")),
                QuickReplyButton(action=PostbackAction(label="⚠️ 用藥注意事項", data="action=medication_precautions")),
                QuickReplyButton(action=PostbackAction(label="🩸 藥物與血糖", data="action=medication_blood_sugar")),
            ]
        )
        return TextSendMessage(text="💊 糖尿病用藥須知\n\n請選擇您想了解的用藥資訊：", quick_reply=quick_reply)
    except Exception as e:
        print(f"❌ Error in create_medication_advice_message: {str(e)}")
        return TextSendMessage(text="❌ 無法顯示用藥建議選單，請稍後再試")


def create_lifestyle_advice_message():
    """創建生活習慣建議選單訊息"""
    try:
        quick_reply = QuickReply(
            items=[
                QuickReplyButton(action=PostbackAction(label="🏃‍♂️ 運動建議", data="action=exercise_advice")),
                QuickReplyButton(action=PostbackAction(label="😴 睡眠品質", data="action=sleep_advice")),
                QuickReplyButton(action=PostbackAction(label="😰 壓力管理", data="action=stress_management")),
                QuickReplyButton(action=PostbackAction(label="🚭 戒菸戒酒", data="action=quit_smoking_drinking")),
            ]
        )
        return TextSendMessage(text="🌟 生活習慣與血糖管理\n\n請選擇您想了解的生活習慣資訊：", quick_reply=quick_reply)
    except Exception as e:
        print(f"❌ Error in create_lifestyle_advice_message: {str(e)}")
        return TextSendMessage(text="❌ 無法顯示生活習慣建議選單，請稍後再試")


def create_calorie_analysis_message():
    """創建熱量分析選單訊息"""
    try:
        quick_reply = QuickReply(
            items=[
                QuickReplyButton(action=CameraAction(label="拍照")),
                QuickReplyButton(action=CameraRollAction(label="選擇相簿")),
                QuickReplyButton(action=PostbackAction(label="❌ 取消", data="action=cancel_analysis")),
            ]
        )
        return TextSendMessage(
            text="🔍 食物熱量分析\n\n請選擇您想使用的分析方式：\n\n📷 拍照 - 立即拍攝食物照片\n🖼️ 選擇相簿 - 從手機相簿選擇照片\n📝 手動輸入 - 輸入食物名稱查詢",
            quick_reply=quick_reply,
        )
    except Exception as e:
        print(f"❌ Error in create_calorie_analysis_message: {str(e)}")
        return TextSendMessage(text="❌ 無法顯示熱量分析選單，請稍後再試")


def create_voice_input_message():
    """創建語音輸入訊息，提供語音按鈕"""
    try:
        # 創建帶有語音輸入按鈕的Quick Reply
        quick_reply = QuickReply(
            items=[
                QuickReplyButton(action=PostbackAction(label="🎤 開始語音錄製", data="action=start_voice_recording")),
                QuickReplyButton(action=PostbackAction(label="📱 使用說明", data="action=voice_tutorial")),
                QuickReplyButton(action=PostbackAction(label="❌ 取消", data="action=cancel_voice_input")),
            ]
        )

        return TextSendMessage(
            text="🎤 語音轉文字功能\n\n選擇下方選項開始使用語音功能，或直接發送語音訊息給我！\n\n✨ 我會自動轉換您的語音為文字並提供糖尿病建議",
            quick_reply=quick_reply,
        )
    except Exception as e:
        print(f"❌ Error in create_voice_input_message: {str(e)}")
        return TextSendMessage(text="❌ 無法顯示語音輸入介面，請稍後再試")


def process_audio_with_gemini(audio_path):
    """
    使用 Gemini AI 處理音頻文件，將語音轉換為文字

    Args:
        audio_path: 音頻文件路徑

    Returns:
        轉換後的文字內容
    """
    try:
        # 讀取音頻文件
        with open(audio_path, "rb") as audio_file:
            audio_data = audio_file.read()
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")

        # 使用 Gemini AI 進行語音轉文字
        audio_prompt = """請將這段語音轉換為繁體中文文字。如果語音不清楚或無法識別，請回應「無法識別語音內容」。"""

        # 構建包含音頻的請求
        audio_response = model.generate_content([{"mime_type": "audio/mpeg", "data": audio_base64}, audio_prompt])

        if not audio_response or not hasattr(audio_response, "text"):
            return "❌ 語音轉換失敗，請重新嘗試"

        converted_text = audio_response.text.strip()

        # 檢查是否成功轉換
        if "無法識別" in converted_text or len(converted_text) < 2:
            return "❌ 無法識別語音內容，請嘗試在更安靜的環境中重新錄音，或說話更清晰一些"

        return converted_text

    except Exception as e:
        print(f"❌ Audio processing error: {str(e)}")
        return f"❌ 語音處理錯誤：{str(e)}"


def handle_blood_sugar_trend_analysis(user_id, days):
    """處理血糖趨勢分析並生成回應"""
    try:
        if not BLOOD_SUGAR_AVAILABLE:
            return TextSendMessage(text="❌ 血糖功能不可用，請檢查 Firebase 設定")

        # 生成血糖趨勢圖表
        result = generate_blood_sugar_trend_chart(user_id, days)

        if isinstance(result, str):  # 錯誤訊息
            return TextSendMessage(text=result)

        # 成功獲得分析結果
        chart_url = result["url"]
        analysis = result["analysis"]

        # 只發送一張大圖，不使用 Flex Message
        return ImageSendMessage(original_content_url=chart_url, preview_image_url=chart_url)

    except Exception as e:
        print(f"❌ Error in handle_blood_sugar_trend_analysis: {str(e)}")
        return TextSendMessage(text=f"❌ 血糖趨勢分析失敗：{str(e)}")


def create_blood_sugar_trend_analysis_flex(chart_url, analysis, days):
    """創建血糖趨勢分析的 Flex Message"""

    # 根據趨勢狀態設定顏色
    if analysis["trend_status"] == "stable":
        status_color = "#4caf50"
        status_bg = "#e8f5e9"
    elif analysis["trend_status"] == "rising":
        status_color = "#f44336"
        status_bg = "#ffebee"
    else:  # falling
        status_color = "#2196f3"
        status_bg = "#e3f2fd"

    # 根據健康狀態設定建議顏色
    if analysis["health_status"] == "normal":
        health_color = "#4caf50"
    elif analysis["health_status"] == "high":
        health_color = "#ff9800"
    else:  # low
        health_color = "#ff5722"

    flex_content = {
        "type": "bubble",
        "size": "giga",
        "header": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "text",
                    "text": f"{days}天血糖趨勢分析 {analysis['trend_emoji']}",
                    "weight": "bold",
                    "size": "xl",
                    "color": "#1976d2",
                    "align": "center",
                    "wrap": True,
                }
            ],
            "backgroundColor": "#f0f8ff",
            "paddingAll": "20px",
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "spacing": "md",
            "paddingAll": "20px",
            "contents": [
                # 圖表圖片
                {
                    "type": "image",
                    "url": chart_url,
                    "size": "full",
                    "aspectMode": "fit",
                    "aspectRatio": "16:10",
                    "margin": "none",
                },
                # 趨勢狀態
                {
                    "type": "box",
                    "layout": "vertical",
                    "contents": [
                        {
                            "type": "text",
                            "text": analysis["trend_message"],
                            "weight": "bold",
                            "size": "lg",
                            "color": status_color,
                            "align": "center",
                        }
                    ],
                    "backgroundColor": status_bg,
                    "cornerRadius": "10px",
                    "paddingAll": "15px",
                    "margin": "lg",
                },
                # 統計資訊
                {
                    "type": "box",
                    "layout": "vertical",
                    "spacing": "sm",
                    "margin": "lg",
                    "contents": [
                        {"type": "text", "text": "📊 統計資訊", "weight": "bold", "size": "md", "color": "#333333"},
                        {
                            "type": "box",
                            "layout": "horizontal",
                            "contents": [
                                {"type": "text", "text": "平均血糖", "size": "sm", "color": "#666666", "flex": 1},
                                {
                                    "type": "text",
                                    "text": f"{analysis['avg_all']} mg/dL",
                                    "size": "sm",
                                    "weight": "bold",
                                    "color": health_color,
                                    "align": "end",
                                },
                            ],
                        },
                        {
                            "type": "box",
                            "layout": "horizontal",
                            "contents": [
                                {"type": "text", "text": "最高 / 最低", "size": "sm", "color": "#666666", "flex": 1},
                                {
                                    "type": "text",
                                    "text": f"{analysis['max_value']} / {analysis['min_value']}",
                                    "size": "sm",
                                    "weight": "bold",
                                    "color": "#333333",
                                    "align": "end",
                                },
                            ],
                        },
                        {
                            "type": "box",
                            "layout": "horizontal",
                            "contents": [
                                {"type": "text", "text": "記錄筆數", "size": "sm", "color": "#666666", "flex": 1},
                                {
                                    "type": "text",
                                    "text": f"{analysis['data_count']} 筆",
                                    "size": "sm",
                                    "weight": "bold",
                                    "color": "#333333",
                                    "align": "end",
                                },
                            ],
                        },
                    ],
                },
                # 建議區塊
                {
                    "type": "box",
                    "layout": "vertical",
                    "spacing": "sm",
                    "margin": "lg",
                    "contents": [
                        {"type": "text", "text": "💡 專業建議", "weight": "bold", "size": "md", "color": "#333333"},
                        {
                            "type": "text",
                            "text": analysis["suggestion"],
                            "size": "sm",
                            "color": "#666666",
                            "wrap": True,
                            "margin": "sm",
                        },
                    ],
                    "backgroundColor": "#f9f9f9",
                    "cornerRadius": "8px",
                    "paddingAll": "15px",
                },
            ],
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "paddingAll": "20px",
            "contents": [
                {
                    "type": "button",
                    "style": "primary",
                    "height": "sm",
                    "action": {"type": "postback", "label": "📝 記錄新的血糖值", "data": "action=record_blood_sugar"},
                    "color": "#4caf50",
                },
                {
                    "type": "button",
                    "style": "secondary",
                    "height": "sm",
                    "action": {"type": "postback", "label": "📊 查看其他時間範圍", "data": "action=show_trend_menu"},
                },
            ],
        },
    }

    return FlexSendMessage(alt_text=f"{days}天血糖趨勢分析結果", contents=flex_content)


if __name__ == "__main__":
    print("🚀 啟動整合版糖尿病飲食分析與血糖記錄 LINE Bot")
    print("📊 功能包含：")
    print("   - 食物圖片營養分析")
    print("   - 糖尿病營養諮詢")
    print("   - 🎤 語音轉文字功能 ✅")
    print("   - 血糖記錄管理" + (" ✅" if BLOOD_SUGAR_AVAILABLE else " ❌ (模組未找到)"))
    print("   - 血糖報表生成" + (" ✅" if BLOOD_SUGAR_AVAILABLE else " ❌ (模組未找到)"))

    # 啟動 ngrok (僅在直接執行時)
    try:
        public_url = ngrok.connect(port).public_url
        print(f' * ngrok tunnel "{public_url}" -> "http://127.0.0.1:{port}" ')
    except Exception as e:
        print(f"⚠️ ngrok 啟動失敗: {str(e)}")
        print("🔧 Bot 仍可以在本地運行，但需要手動設置外部連接")

    app.run(host="0.0.0.0", port=port)
