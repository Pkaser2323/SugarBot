import os
import json
import io
import re
from typing import List, Tuple
from dotenv import load_dotenv
from flask import Flask, request
from datetime import datetime
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageMessage,
    FlexSendMessage, PostbackEvent, PostbackAction,
    QuickReply, QuickReplyButton, MessageAction,
    FollowEvent, UnfollowEvent
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

# 設置 tokenizers 並行處理
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免 fork 後的死鎖問題

# Configure Gemini AI
genai.configure(api_key=API_KEY)

generation_config = {
    "temperature": 0.2,
    "max_output_tokens": 512,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel("gemini-2.5-flash-lite", generation_config=generation_config)

# 模型配置
EMBED_MODEL_NAME = "DMetaSoul/sbert-chinese-general-v2"
SAS_MODEL_DIR = os.path.join(os.path.dirname(__file__), "sas_model")

# 全局變數
sas_model = None
SAS_PARAMS = {
    "temperature": 2.0,
    "high_threshold": 0.6,
    "low_threshold": 0.3
}
is_model_ready = False
model_init_error = None

def initialize_sas_model():
    """初始化 SAS 模型（非阻塞）"""
    global sas_model, SAS_PARAMS, is_model_ready, model_init_error
    
    try:
        print("⏳ 正在從 Hugging Face 加載 SAS 模型...")
        sas_model = CrossEncoder("Pkaser2323/SAS_Model", device="cpu")
        print("✅ 從 Hugging Face 加載 SAS 模型成功！")
        
        # 嘗試從 Hugging Face 加載參數
        try:
            from huggingface_hub import hf_hub_download
            params_path = hf_hub_download(
                repo_id="Pkaser2323/SAS_Model",
                filename="best_params.json"
            )
            with open(params_path, "r", encoding="utf-8") as f:
                SAS_PARAMS = json.load(f)
            print("✅ 從 Hugging Face 加載參數成功！")
        except Exception as e:
            print(f"⚠️ 無法從 Hugging Face 加載參數: {str(e)}")
            print("⏳ 嘗試從本地加載參數...")
            try:
                with open(os.path.join(SAS_MODEL_DIR, "best_params.json"), "r", encoding="utf-8") as f:
                    SAS_PARAMS = json.load(f)
                    print("✅ 從本地加載參數成功！")
            except Exception as e2:
                print(f"⚠️ 無法從本地加載參數: {str(e2)}")
                # 使用預設參數
                print("✅ 使用預設參數")

    except Exception as e:
        print(f"⚠️ 無法從 Hugging Face 加載模型: {str(e)}")
        print("⏳ 嘗試從本地加載模型...")
        try:
            sas_model = CrossEncoder(SAS_MODEL_DIR)
            sas_model.model = sas_model.model.to("cpu")
            print("✅ 從本地加載模型成功！")
            
            # 嘗試從本地加載參數
            try:
                with open(os.path.join(SAS_MODEL_DIR, "best_params.json"), "r", encoding="utf-8") as f:
                    SAS_PARAMS = json.load(f)
                print("✅ 從本地加載參數成功！")
            except Exception as e2:
                print(f"⚠️ 無法從本地加載參數: {str(e2)}")
                # 使用預設參數
                print("✅ 使用預設參數")
        except Exception as e2:
            print(f"❌ 本地模型加載也失敗: {str(e2)}")
            print("⚠️ 將以降級模式運行（不使用 SAS 模型）")
            model_init_error = str(e2)
            return

    is_model_ready = True


def predict_pos_prob(
    model,
    questions: List[str],
    answers: List[str],
    temperature: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """預測正類機率"""
    global sas_model, is_model_ready
    
    # 空輸入檢查
    if not questions or not answers or len(questions) != len(answers):
        return np.array([]), np.array([])
    
    # 延遲載入：首次使用時才初始化 SAS 模型
    if not is_model_ready and not model_init_error:
        print("⏳ 首次評分，正在初始化 SAS 模型...")
        try:
            initialize_sas_model()
        except Exception as e:
            print(f"❌ SAS 模型初始化失敗: {str(e)}")
            return np.ones(len(questions)) * 0.7, np.ones(len(questions)) * 0.7
    
    # 模型就緒檢查
    if not is_model_ready or model is None:
        print("⚠️ SAS 模型未就緒，返回預設分數")
        return np.ones(len(questions)) * 0.7, np.ones(len(questions)) * 0.7
    
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

    # 添加重試機制
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.1, "max_output_tokens": 300}
            )
            
            if not response or not hasattr(response, "text"):
                retry_count += 1
                print(f"⚠️ 生成子問題失敗 (嘗試 {retry_count}/{max_retries}): 回應無效")
                continue
                
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
            
            # 如果成功生成至少一個子問題，就返回結果
            if subqs:
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
            
            retry_count += 1
            print(f"⚠️ 生成子問題失敗 (嘗試 {retry_count}/{max_retries}): 未生成有效子問題")
            
        except Exception as e:
            retry_count += 1
            print(f"⚠️ 生成子問題失敗 (嘗試 {retry_count}/{max_retries}): {str(e)}")
            if retry_count == max_retries:
                print("❌ 達到最大重試次數，返回原問題")
                return [question]
            
    # 如果所有重試都失敗，返回原問題
    return [question]

def initialize_vector_db():
    """檢查向量資料庫文件是否存在，如果不存在則從 CSV 創建"""
    import pandas as pd
    
    # 1. 設置所有路徑（一次設定，不再更改）
    base_dir = os.path.dirname(__file__)
    db_dir = "/tmp/vector_DB" if os.environ.get("RENDER") else os.path.join(base_dir, "vector_DB")
    db_path = os.path.join(db_dir, "diabetes_comprehensive_db")
    csv_path = os.path.join(base_dir, "datacsv", "a_topic_analyzed_processed.csv")
    
    print(f"向量資料庫路徑: {db_path}")
    print(f"CSV 文件路徑: {csv_path}")
    
    try:
        # 2. 確保目錄存在
        os.makedirs(db_dir, exist_ok=True)
        
        # 3. 檢查向量資料庫文件是否存在且完整
        if os.path.exists(db_path) and os.path.exists(os.path.join(db_path, "index.faiss")):
            print("✅ 向量資料庫文件已存在")
            return db_path
        
        # 4. 從 CSV 創建新的向量資料庫
        print("⚙️ 開始創建新的向量資料庫...")
        
        # 檢查並讀取 CSV
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
        if df.empty:
            raise ValueError("CSV 文件中沒有有效的問答對")
        
        # 準備文本
        texts = []
        for _, row in df.iterrows():
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
        
        # 驗證保存是否成功
        if not os.path.exists(os.path.join(db_path, "index.faiss")):
            raise FileNotFoundError("向量資料庫保存失敗或保存的文件是空的")
            
        print(f"✅ 向量資料庫成功創建並保存至：{db_path}")
        return db_path
        
    except Exception as e:
        print(f"❌ 初始化向量資料庫時發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        raise  # 向上傳遞錯誤，讓應用程式知道初始化失敗

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

def search_related_content(query):
    """檢索相關文本
    
    Args:
        query: 查詢文本
        
    Returns:
        Tuple[str, List]: (合併後的文本, 文檔列表)
    """
    global vector_db
    
    try:
        # 延遲載入：首次使用時才初始化向量資料庫
        if not vector_db:
            print("⏳ 首次查詢，正在初始化向量資料庫...")
            db_path = initialize_vector_db()  # 只檢查文件或創建文件
            
            # 創建嵌入模型
            model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBED_MODEL_NAME,
                model_kwargs=model_kwargs
            )
            
            # 載入向量資料庫
            vector_db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
            print("✅ 向量資料庫初始化完成")
        
        # 執行檢索
        docs = vector_db.invoke(query)
        return "\n---\n".join([doc.page_content for doc in docs]), docs
        
    except Exception as e:
        print(f"❌ 檢索過程發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        return "", []

def generate_answer(query: str, docs=None):
    """生成回答，使用 Fast/Slow path 機制"""
    try:
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
你是一位充滿熱情、幽默又專業的糖尿病護理師，
請根據以下資訊，用自然、有感染力的口吻回答病患問題。

請想像你正在和病患聊天，
語氣要像在現場講話一樣溫暖、有活力、讓人放鬆。

相關資訊：
{evidence}

病患提問：
{query}

回答要求：
1. 使用繁體中文，以熱情親切的語氣回答
2. 回答限制100字以內
3. 內容要清晰易懂
4. 只根據提供的資訊回答
5. 不要使用任何特殊符號或標記（如*號）
6. 適當分段以提高可讀性

請提供您的專業建議：
"""
                # 添加重試機制
                max_retries = 3
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        response = model.generate_content(template)
                        if response and hasattr(response, 'text'):
                            return response.text
                        break
                    except Exception as e:
                        print(f"⚠️ 生成回答時發生錯誤 (嘗試 {retry_count + 1}/{max_retries}): {str(e)}")
                        retry_count += 1
                        if retry_count == max_retries:
                            # 如果重試全部失敗，返回檢索到的相關文本作為回答
                            return f"根據資料庫內容：{evidence[:100]}..."  # 截取前100字
                
                return "不好意思，我不清楚這個問題，建議您諮詢專業醫師。"
        
        # 如果 Fast path 失敗，嘗試 Slow path
        print("🐢 Slow path: 拆解子問題...")
        subqs = generate_subqueries(query)
        print(f"✓ 生成 {len(subqs)} 個子問題")
        
        # 為每個子問題檢索並評估
        all_evidence = []
        low_thr = SAS_PARAMS.get("low_threshold", 0.3)
        
        try:
            # 檢查向量資料庫是否初始化
            if not vector_db:
                print("⚠️ 向量資料庫尚未初始化")
                return "抱歉，系統暫時無法處理您的問題，請稍後再試。"

            # 為每個子問題檢索並評估
            for sq in subqs:
                try:
                    print(f"檢索子問題: {sq}")
                    # 檢索相關文本
                    sq_docs = vector_db.invoke(sq)
                    if not sq_docs:
                        print("未找到相關文本")
                        continue
                    
                    print(f"找到 {len(sq_docs)} 個相關文本")
                    
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
                        selected_docs = [sq_docs[i].page_content for i in top_indices]
                        all_evidence.extend(selected_docs)
                        print(f"添加 {len(selected_docs)} 個高分段落")
                    else:
                        print("沒有段落通過低門檻")
                        
                except Exception as e:
                    print(f"⚠️ 處理子問題時發生錯誤: {str(e)}")
                    continue  # 繼續處理下一個子問題
                    
        except Exception as e:
            print(f"❌ 檢索過程發生錯誤: {str(e)}")
            return "抱歉，系統暫時無法處理您的問題，請稍後再試。"
        
        # 如果沒有找到任何有效證據
        if not all_evidence:
            return "這個問題需要更多專業資訊才能完整回答，建議您諮詢主治醫師。"
        
        # 使用 GPT 整合所有證據生成回答
        evidence_text = "\n".join(all_evidence)
        template = f"""
您是一位熱情且專業的糖尿病護理師，請根據以下資訊回答病患問題：

相關資訊：
{evidence_text}

病患提問：
{query}

回答要求：
1.不要使用「您好」「別擔心」「請放心」等制式開場白。
2. 使用繁體中文，以專業但親切的語氣回答
3. 回答限制100字以內
4. 語氣讓人感覺被理解、有希望，內容清楚又實用
5. 只根據提供的資訊回答
6. 不要使用任何特殊符號或標記（如*號）
7. 適當分段以提高可讀性

請以貼心又充滿能量的方式回答病患：：
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
        
    except Exception as e:
        print(f"❌ 生成回答時發生錯誤: {str(e)}")
        return "抱歉，系統暫時無法處理您的問題，請稍後再試。"

# 設置 HuggingFace 快取目錄（在 Render 上使用 /tmp）
if os.environ.get("RENDER"):
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"
    os.environ["HF_HOME"] = "/tmp/huggingface"
    print(f"HuggingFace 快取目錄設為: {os.environ['TRANSFORMERS_CACHE']}")

# 全局變數初始化
app = Flask(__name__)
port = 5000
line_bot_api = None
handler = None
vector_db = None
initialized = False

def initialize_app():
    """初始化應用程式（只在第一次啟動時執行）"""
    global line_bot_api, handler, initialized
    
    if initialized:
        return
    
    try:
        print("⚙️ 開始初始化應用程式...")
        
        # LINE Bot setup
        print("⏳ 初始化 LINE Bot API...")
        global line_bot_api
        line_bot_api = LineBotApi(LINE_ACCESS_TOKEN)
        global handler
        handler = WebhookHandler(LINE_SECRET)
        print("✅ LINE Bot API 初始化成功")
        
        # 檢查必要文件是否存在
        print("⏳ 檢查必要文件...")
        db_dir = "/tmp/vector_DB" if os.environ.get("RENDER") else os.path.join(os.path.dirname(__file__), "vector_DB")
        db_path = os.path.join(db_dir, "diabetes_comprehensive_db")
        if not os.path.exists(db_path) or not os.path.exists(os.path.join(db_path, "index.faiss")):
            print("⚠️ 向量資料庫文件不存在，將在首次查詢時創建")
        else:
            print("✅ 向量資料庫文件檢查完成")
        
        # 標記初始化完成
        initialized = True
        print("✅ 基礎應用程式初始化完成")
        
    except Exception as e:
        print(f"❌ 應用程式初始化失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# 在應用程式啟動時初始化
initialize_app()

# 在另一個線程中預載 SAS 模型
import threading
def preload_sas_model():
    """在背景線程中預載 SAS 模型"""
    try:
        print("⏳ 在背景預載 SAS 模型...")
        import requests
        response = requests.get("http://localhost:5000/init")
        if response.status_code == 200:
            print("✅ SAS 模型預載請求已發送")
        else:
            print(f"⚠️ SAS 模型預載請求失敗: {response.status_code}")
    except Exception as e:
        print(f"⚠️ 無法預載 SAS 模型: {str(e)}")

# 啟動背景預載
if not os.environ.get("RENDER"):  # 本地開發環境
    threading.Thread(target=preload_sas_model, daemon=True).start()

# 根路徑
@app.route("/", methods=["GET"])
def root():
    """根路徑處理"""
    global initialized
    if not initialized:
        try:
            initialize_app()
            return "糖尿病諮詢 LINE Bot 服務初始化完成", 200
        except:
            return "糖尿病諮詢 LINE Bot 服務初始化失敗", 500
    
    # 在回應中包含模型狀態
    status = {
        "service": "running",
        "sas_model": "ready" if is_model_ready else "not_ready",
        "error": model_init_error if model_init_error else None
    }
    return json.dumps(status, ensure_ascii=False), 200

# 健康檢查路由
@app.route("/health", methods=["GET"])
def health_check():
    """健康檢查端點，回報服務狀態"""
    status = {
        "status": "healthy",
        "model_ready": is_model_ready,
        "error": model_init_error if model_init_error else None
    }
    return json.dumps(status), 200

# 初始化路由
@app.route("/init", methods=["GET"])
def init_models():
    """初始化 SAS 模型（在應用程式啟動後立即調用）"""
    global is_model_ready, model_init_error
    
    try:
        if not is_model_ready and not model_init_error:
            print("⏳ 預先初始化 SAS 模型...")
            initialize_sas_model()
            if is_model_ready:
                print("✅ SAS 模型初始化成功，已準備好進行評分")
            else:
                print("⚠️ SAS 模型初始化未完成")
    except Exception as e:
        print(f"❌ SAS 模型初始化失敗: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return json.dumps({
        "status": "success" if is_model_ready else "initializing",
        "model_ready": is_model_ready,
        "error": model_init_error if model_init_error else None
    }), 200

# LINE Bot webhook
@app.route("/", methods=['POST'])  # 改為根路徑
@app.route("/callback", methods=['POST'])  # 保留 /callback 以向後兼容
def callback():
    """Webhook 入口點，只做驗簽和分派"""
    body = request.get_data(as_text=True)
    signature = request.headers.get('X-Line-Signature', '')
    
    try:
        handler.handle(body, signature)
        return "OK", 200
    except InvalidSignatureError:
        print("無效的簽名")
        return "Invalid signature", 400
    except Exception as e:
        print(f"處理 Webhook 時發生錯誤: {str(e)}")
        print(f"收到內容: {body}")
        return "Error", 500
    
# 設置基礎路徑
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 用戶數據文件路徑（使用專案目錄的 data 資料夾確保所有 worker 共享）
USER_DATA_FILE = os.path.join(DATA_DIR, "user_data.json")

# 全局數據存儲（必須在 load_user_data 之前定義）
global_data_store = {
    "processed_messages": set(),  # 用於存儲已處理的消息ID
    "message_lock": False,        # 用於防止並發處理
    "food_analysis": {},         # 用於保存食物分析結果
    "user_data_cache": {},       # 用於緩存用戶數據
    "last_save_time": None,      # 上次保存時間
    "data_hash": None           # 用戶數據的雜湊值
}

def create_terms_flex_message():
    """創建專業的用戶條款 Flex Message"""
    return {
        "type": "flex",
        "altText": "糖小護服務條款",
        "contents": {
            "type": "bubble",
            "size": "mega",
            "header": {
                "type": "box",
                "layout": "vertical",
                "contents": [
                    {
                        "type": "text",
                        "text": "糖小護",
                        "weight": "bold",
                        "size": "xl",
                        "color": "#2E86AB",
                        "align": "center"
                    },
                    {
                        "type": "text",
                        "text": "服務條款",
                        "size": "md",
                        "color": "#5A9FD4",
                        "align": "center",
                        "margin": "sm"
                    }
                ],
                "backgroundColor": "#F0F8FF",
                "paddingAll": "20px",
                "cornerRadius": "10px"
            },
            "body": {
                "type": "box",
                "layout": "vertical",
                "contents": [
                    {
                        "type": "text",
                        "text": "一、資料蒐集範圍",
                        "weight": "bold",
                        "size": "sm",
                        "color": "#2E86AB",
                        "margin": "md"
                    },
                    {
                        "type": "text",
                        "text": "血糖數值記錄\n健康諮詢對話內容\n上傳的醫療相關圖片\n使用行為統計資料",
                        "size": "xs",
                        "color": "#666666",
                        "wrap": True,
                        "margin": "sm"
                    },
                    {
                        "type": "text",
                        "text": "二、使用目的",
                        "weight": "bold",
                        "size": "sm",
                        "color": "#2E86AB",
                        "margin": "lg"
                    },
                    {
                        "type": "text",
                        "text": "提供個人化健康建議\n生成專屬個人健康報表\n持續改善服務品質",
                        "size": "xs",
                        "color": "#666666",
                        "wrap": True,
                        "margin": "sm"
                    },
                    {
                        "type": "text",
                        "text": "三、隱私保護",
                        "weight": "bold",
                        "size": "sm",
                        "color": "#2E86AB",
                        "margin": "lg"
                    },
                    {
                        "type": "text",
                        "text": "您可隨時要求刪除個人資料\n全程遵守《個人資料保護法》及相關醫療資訊法規",
                        "size": "xs",
                        "color": "#666666",
                        "wrap": True,
                        "margin": "sm"
                    },
                    {
                        "type": "text",
                        "text": "四、同意與生效",
                        "weight": "bold",
                        "size": "sm",
                        "color": "#2E86AB",
                        "margin": "lg"
                    },
                    {
                        "type": "text",
                        "text": "繼續使用即表示您已閱讀並同意本服務條款。",
                        "size": "xs",
                        "color": "#666666",
                        "wrap": True,
                        "margin": "sm"
                    }
                ],
                "paddingAll": "20px",
                "spacing": "sm"
            },
            "footer": {
                "type": "box",
                "layout": "vertical",
                "contents": [
                    {
                        "type": "separator",
                        "margin": "md",
                        "color": "#E6F3FF"
                    },
                    {
                        "type": "box",
                        "layout": "horizontal",
                        "contents": [
                            {
                                "type": "button",
                                "style": "secondary",
                                "height": "sm",
                                "action": {
                                    "type": "message",
                                    "label": "暫不同意",
                                    "text": "不同意"
                                },
                                "color": "#CCCCCC",
                                "flex": 1
                            },
                            {
                                "type": "button",
                                "style": "primary",
                                "height": "sm",
                                "action": {
                                    "type": "message",
                                    "label": "同意並開始使用",
                                    "text": "同意"
                                },
                                "color": "#2E86AB",
                                "flex": 2
                            }
                        ],
                        "spacing": "sm",
                        "margin": "md"
                    }
                ],
                "paddingAll": "20px"
            }
        }
    }

def load_user_data():
    """載入用戶數據"""
    global global_data_store
    
    try:
        # 如果緩存中有數據且距離上次保存時間不超過5秒，直接返回緩存
        if global_data_store["user_data_cache"] and global_data_store["last_save_time"]:
            time_diff = (datetime.now() - global_data_store["last_save_time"]).total_seconds()
            if time_diff < 5:
                print("⚡ 使用緩存的用戶數據")
                return global_data_store["user_data_cache"].copy()
        
        # 檢查文件是否存在
        if not os.path.exists(USER_DATA_FILE):
            print(f"⚠️ 用戶數據文件不存在，將創建新文件: {USER_DATA_FILE}")
            # 確保目錄存在
            os.makedirs(os.path.dirname(USER_DATA_FILE), exist_ok=True)
            # 創建空的用戶數據文件
            with open(USER_DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
            return {}
            
        # 檢查文件權限
        if not os.access(USER_DATA_FILE, os.R_OK):
            print(f"⚠️ 無法讀取用戶數據文件（權限問題）: {USER_DATA_FILE}")
            return {}
            
        # 讀取文件
            with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
        # 驗證數據格式
        if not isinstance(data, dict):
            print(f"⚠️ 用戶數據格式無效: {type(data)}")
            return {}
            
        print(f"✅ 成功載入用戶數據，共 {len(data)} 位用戶")
        
        # 檢查並修復可能的無效狀態
        for user_id, user_data in data.items():
            if not isinstance(user_data, dict):
                print(f"⚠️ 用戶 {user_id} 數據格式無效，重置")
                data[user_id] = {"status": "pending"}
                continue
                
            if "status" not in user_data:
                print(f"⚠️ 用戶 {user_id} 缺少狀態資訊，設為 pending")
                user_data["status"] = "pending"
            elif user_data["status"] not in ["pending", "awaiting_button_response", "awaiting_tutorial_choice", "tutorial_shown", "detailed_tutorial", "agreed", "disagreed"]:
                print(f"⚠️ 用戶 {user_id} 狀態無效: {user_data['status']}，重置為 pending")
                user_data["status"] = "pending"
                
            print(f"用戶 {user_id} 狀態: {user_data['status']}")
        
        # 更新緩存（使用深拷貝）
        global_data_store["user_data_cache"] = json.loads(json.dumps(data))
        global_data_store["last_save_time"] = datetime.now()
        global_data_store["data_hash"] = calculate_data_hash(data)
            
        return data
        
    except json.JSONDecodeError as e:
        print(f"❌ 用戶數據文件格式錯誤: {e}")
        # 備份損壞的文件
        if os.path.exists(USER_DATA_FILE):
            backup_file = f"{USER_DATA_FILE}.bak.{int(time.time())}"
            try:
                import shutil
                shutil.copy2(USER_DATA_FILE, backup_file)
                print(f"✅ 已備份損壞的文件到: {backup_file}")
            except Exception as be:
                print(f"⚠️ 備份文件失敗: {be}")
        return {}
    except Exception as e:
        print(f"❌ 載入用戶數據失敗: {e}")
        import traceback
        traceback.print_exc()
    return {}

def calculate_data_hash(data):
    """計算數據的雜湊值（用於檢測變更）"""
    try:
        # 將數據轉換為規範化的 JSON 字符串（確保鍵的順序一致）
        json_str = json.dumps(data, sort_keys=True)
        # 計算 SHA-256 雜湊
        return hashlib.sha256(json_str.encode()).hexdigest()
    except Exception as e:
        print(f"⚠️ 計算數據雜湊時出錯: {e}")
        return None

def save_user_data(data):
    """保存用戶數據"""
    global USER_DATA_FILE, global_data_store
    
    # 計算當前數據的雜湊值
    current_hash = calculate_data_hash(data)
    cached_hash = global_data_store.get("data_hash")
    
    # 檢查是否需要保存
    if global_data_store["last_save_time"] and current_hash and cached_hash:
        time_diff = (datetime.now() - global_data_store["last_save_time"]).total_seconds()
        if time_diff < 5 and current_hash == cached_hash:
            print("⏳ 跳過保存：數據未變更或距離上次保存時間太短")
            return
    
    try:
        # 驗證數據
        if not isinstance(data, dict):
            raise ValueError(f"數據格式無效: {type(data)}")
            
        # 驗證每個用戶的數據格式
        for user_id, user_data in data.items():
            if not isinstance(user_data, dict):
                raise ValueError(f"用戶 {user_id} 數據格式無效: {type(user_data)}")
            if "status" not in user_data:
                raise ValueError(f"用戶 {user_id} 缺少狀態資訊")
                
        # 確保目錄存在
        directory = os.path.dirname(USER_DATA_FILE)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"✅ 創建目錄成功: {directory}")
            except Exception as e:
                print(f"⚠️ 無法創建目錄 {directory}: {e}")
                # 如果是 Render 環境，使用 /tmp
                if os.environ.get("RENDER"):
                    USER_DATA_FILE = "/tmp/user_data.json"
                    print(f"🔄 切換到 /tmp 目錄: {USER_DATA_FILE}")
        
        # 保存文件
        with open(USER_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"✅ 成功保存用戶數據，共 {len(data)} 位用戶")
        
        # 更新緩存（使用深拷貝）
        global_data_store["user_data_cache"] = json.loads(json.dumps(data))
        global_data_store["last_save_time"] = datetime.now()
        global_data_store["data_hash"] = current_hash
        
        # 驗證文件是否真的保存成功
        if not os.path.exists(USER_DATA_FILE):
            raise FileNotFoundError(f"文件未成功創建: {USER_DATA_FILE}")
            
        file_size = os.path.getsize(USER_DATA_FILE)
        if file_size == 0:
            raise ValueError(f"文件大小為0: {USER_DATA_FILE}")
            
        print(f"✅ 文件保存成功: {USER_DATA_FILE} ({file_size} bytes)")
        
    except Exception as e:
        print(f"❌ 保存用戶數據失敗: {e}")
        print(f"當前數據: {data}")
        import traceback
        traceback.print_exc()

def create_welcome_message():
    """創建歡迎訊息 - 條款同意完成"""
    return {
        "type": "flex",
        "altText": "歡迎使用糖小護",
        "contents": {
            "type": "bubble",
            "header": {
                "type": "box",
                "layout": "vertical",
                "contents": [
                    {
                        "type": "text",
                        "text": "條款同意完成",
                        "weight": "bold",
                        "size": "lg",
                        "color": "#FFFFFF",
                        "align": "center"
                    }
                ],
                "backgroundColor": "#2E86AB",
                "paddingAll": "15px"
            },
            "body": {
                "type": "box",
                "layout": "vertical",
                "contents": [
                    {
                        "type": "text",
                        "text": "接下來將為您介紹糖小護的功能...",
                        "size": "sm",
                        "color": "#666666",
                        "align": "center",
                        "wrap": True
                    }
                ],
                "paddingAll": "20px"
            }
        }
    }

def create_button_check_message():
    """創建第一階段按鈕確認訊息 - 詢問是否看到按鈕"""
    return TextSendMessage(
        text="嗨~你有沒有看到下面的按鈕呢？",
        quick_reply=QuickReply(
            items=[
                QuickReplyButton(
                    action=MessageAction(label="有", text="有")
                ),
                QuickReplyButton(
                    action=MessageAction(label="沒有", text="沒有")
                )
            ]
        )
    )

def create_tutorial_choice_message():
    """創建第二階段教學意願確認訊息"""
    return TextSendMessage(
        text="太棒了！那你想要了解更多的教學內容嗎？",
        quick_reply=QuickReply(
            items=[
                QuickReplyButton(
                    action=MessageAction(label="我要教學", text="我要教學")
                ),
                QuickReplyButton(
                    action=MessageAction(label="我不要教學", text="我不要教學")
                )
            ]
        )
    )

def create_skip_tutorial_message():
    """創建跳過教學的祝福訊息"""
    return TextSendMessage(
        text="好的，那祝你使用愉快！🍭\n\n如果之後想了解功能，隨時都可以詢問我喔～\n\n現在就開始記錄您的血糖數值或詢問健康問題吧！"
    )

def create_tutorial_carousel():
    """創建5頁Flex Carousel功能介紹訊息"""
    return {
        "type": "flex",
        "altText": "糖小護功能介紹",
        "contents": {
            "type": "carousel",
            "contents": [
                {
                    "type": "bubble",
                    "hero": {
                        "type": "image",
                        "url": "https://i.postimg.cc/7h9gjpYL/url.png",
                        "size": "full",
                        "aspectRatio": "20:13",
                        "aspectMode": "cover"
                    },
                    "body": {
                        "type": "box",
                        "layout": "vertical",
                        "contents": [
                            {
                                "type": "text",
                                "text": "歡迎使用糖小護",
                                "weight": "bold",
                                "size": "lg",
                                "color": "#2E86AB",
                                "align": "center"
                            },
                            {
                                "type": "text",
                                "text": "您的專屬健康管理助手",
                                "size": "md",
                                "color": "#666666",
                                "align": "center",
                                "wrap": True,
                                "margin": "sm"
                            },
                            {
                                "type": "separator",
                                "margin": "lg",
                                "color": "#E6F3FF"
                            },
                            {
                                "type": "text",
                                "text": "👉 往右滑動查看功能介紹",
                                "size": "sm",
                                "color": "#5A9FD4",
                                "align": "center",
                                "margin": "lg",
                                "weight": "bold"
                            }
                        ],
                        "paddingAll": "20px"
                    }
                },
                {
                    "type": "bubble",
                    "hero": {
                        "type": "image",
                        "url": "https://i.postimg.cc/GtdF7cry/url2.png",
                        "size": "full",
                        "aspectRatio": "20:13",
                        "aspectMode": "cover"
                    },
                    "body": {
                        "type": "box",
                        "layout": "vertical",
                        "contents": [
                            {
                                "type": "text",
                                "text": "問與答",
                                "weight": "bold",
                                "size": "lg",
                                "color": "#2E86AB",
                                "align": "center"
                            },
                            {
                                "type": "text",
                                "text": "• 專業糖尿病知識問答\n• RAG 檢索增強生成\n• 24小時智能諮詢",
                                "size": "sm",
                                "color": "#666666",
                                "wrap": True,
                                "margin": "md"
                            }
                        ],
                        "paddingAll": "20px"
                    },
                    "footer": {
                        "type": "box",
                        "layout": "vertical",
                        "contents": [
                            {
                                "type": "button",
                                "style": "primary",
                                "height": "sm",
                                "action": {
                                    "type": "message",
                                    "label": "查看教學",
                                    "text": "問答教學"
                                },
                                "color": "#2E86AB"
                            }
                        ],
                        "paddingAll": "20px"
                    }
                },
                {
                    "type": "bubble",
                    "hero": {
                        "type": "image",
                        "url": "https://i.postimg.cc/x8nvx0Yk/url3.png",
                        "size": "full",
                        "aspectRatio": "20:13",
                        "aspectMode": "cover"
                    },
                    "body": {
                        "type": "box",
                        "layout": "vertical",
                        "contents": [
                            {
                                "type": "text",
                                "text": "語音轉文字",
                                "weight": "bold",
                                "size": "lg",
                                "color": "#2E86AB",
                                "align": "center"
                            },
                            {
                                "type": "text",
                                "text": "• 支援國語、台語辨識\n• LIFF 網頁錄音介面\n• 即時語音轉文字",
                                "size": "sm",
                                "color": "#666666",
                                "wrap": True,
                                "margin": "md"
                            }
                        ],
                        "paddingAll": "20px"
                    },
                    "footer": {
                        "type": "box",
                        "layout": "vertical",
                        "contents": [
                            {
                                "type": "button",
                                "style": "primary",
                                "height": "sm",
                                "action": {
                                    "type": "message",
                                    "label": "查看教學",
                                    "text": "語音教學"
                                },
                                "color": "#2E86AB"
                            }
                        ],
                        "paddingAll": "20px"
                    }
                },
                {
                    "type": "bubble",
                    "hero": {
                        "type": "image",
                        "url": "https://i.postimg.cc/d3w2Hqvk/url4.png",
                        "size": "full",
                        "aspectRatio": "20:13",
                        "aspectMode": "cover"
                    },
                    "body": {
                        "type": "box",
                        "layout": "vertical",
                        "contents": [
                            {
                                "type": "text",
                                "text": "血糖管理室",
                                "weight": "bold",
                                "size": "lg",
                                "color": "#2E86AB",
                                "align": "center"
                            },
                            {
                                "type": "text",
                                "text": "• 血糖數值記錄追蹤\n• Firebase 雲端儲存\n• 個人化報表圖表",
                                "size": "sm",
                                "color": "#666666",
                                "wrap": True,
                                "margin": "md"
                            }
                        ],
                        "paddingAll": "20px"
                    },
                    "footer": {
                        "type": "box",
                        "layout": "vertical",
                        "contents": [
                            {
                                "type": "button",
                                "style": "primary",
                                "height": "sm",
                                "action": {
                                    "type": "message",
                                    "label": "查看教學",
                                    "text": "血糖教學"
                                },
                                "color": "#2E86AB"
                            }
                        ],
                        "paddingAll": "20px"
                    }
                },
                {
                    "type": "bubble",
                    "hero": {
                        "type": "image",
                        "url": "https://i.postimg.cc/KjfnCdv7/url5.png",
                        "size": "full",
                        "aspectRatio": "20:13",
                        "aspectMode": "cover"
                    },
                    "body": {
                        "type": "box",
                        "layout": "vertical",
                        "contents": [
                            {
                                "type": "text",
                                "text": "影像辨識",
                                "weight": "bold",
                                "size": "lg",
                                "color": "#2E86AB",
                                "align": "center"
                            },
                            {
                                "type": "text",
                                "text": "• Gemini AI影像分析\n• 醫療相關圖片辨識\n• 智能健康建議",
                                "size": "sm",
                                "color": "#666666",
                                "wrap": True,
                                "margin": "md"
                            }
                        ],
                        "paddingAll": "20px"
                    },
                    "footer": {
                        "type": "box",
                        "layout": "vertical",
                        "contents": [
                            {
                                "type": "button",
                                "style": "primary",
                                "height": "sm",
                                "action": {
                                    "type": "message",
                                    "label": "查看教學",
                                    "text": "影像教學"
                                },
                                "color": "#2E86AB"
                            }
                        ],
                        "paddingAll": "20px"
                    }
                }
            ]
        }
    }

def create_qa_tutorial_carousel():
    """創建問答功能詳細教學 Carousel"""
    return {
        "type": "flex",
        "altText": "問答功能教學",
        "contents": {
            "type": "carousel",
            "contents": [
                {
                    "type": "bubble",
                    "hero": {
                        "type": "image",
                        "url": "https://your-image-host.com/qa-step1.jpg",
                        "size": "full",
                        "aspectRatio": "20:13",
                        "aspectMode": "cover"
                    },
                    "body": {
                        "type": "box",
                        "layout": "vertical",
                        "contents": [
                            {
                                "type": "text",
                                "text": "第1步：開始提問",
                                "weight": "bold",
                                "size": "lg",
                                "color": "#2E86AB",
                                "align": "center"
                            },
                            {
                                "type": "text",
                                "text": "直接在聊天室輸入您的健康問題，例如：\n\n• 血糖高怎麼辦？\n• 糖尿病可以吃什麼？\n• 運動對血糖的影響",
                                "size": "sm",
                                "color": "#666666",
                                "wrap": True,
                                "margin": "md"
                            }
                        ],
                        "paddingAll": "20px"
                    }
                },
                {
                    "type": "bubble",
                    "hero": {
                        "type": "image",
                        "url": "https://your-image-host.com/qa-step2.jpg",
                        "size": "full",
                        "aspectRatio": "20:13",
                        "aspectMode": "cover"
                    },
                    "body": {
                        "type": "box",
                        "layout": "vertical",
                        "contents": [
                            {
                                "type": "text",
                                "text": "第2步：AI分析回答",
                                "weight": "bold",
                                "size": "lg",
                                "color": "#2E86AB",
                                "align": "center"
                            },
                            {
                                "type": "text",
                                "text": "糖小護會透過RAG系統：\n\n• 搜尋專業知識庫\n• 分析您的問題\n• 提供準確的健康建議\n• 給出相關的參考資料",
                                "size": "sm",
                                "color": "#666666",
                                "wrap": True,
                                "margin": "md"
                            }
                        ],
                        "paddingAll": "20px"
                    }
                }
            ]
        }
    }

def create_voice_tutorial_carousel():
    """創建語音轉文字詳細教學 Carousel"""
    return {
        "type": "flex",
        "altText": "語音轉文字教學",
        "contents": {
            "type": "carousel",
            "contents": [
                {
                    "type": "bubble",
                    "hero": {
                        "type": "image",
                        "url": "https://i.postimg.cc/56xSYYbr/voice1.png",
                        "size": "full",
                        "aspectRatio": "20:13",
                        "aspectMode": "cover"
                    },
                    "body": {
                        "type": "box",
                        "layout": "vertical",
                        "contents": [
                            {
                                "type": "text",
                                "text": "第1步：點擊語音按鈕",
                                "weight": "bold",
                                "size": "lg",
                                "color": "#2E86AB",
                                "align": "center"
                            },
                            {
                                "type": "text",
                                "text": "在聊天室下方的功能按鈕中，找到並點擊：\n\n🎤 語音轉文字\n\n點擊後會自動跳轉到錄音網頁",
                                "size": "sm",
                                "color": "#666666",
                                "wrap": True,
                                "margin": "md"
                            }
                        ],
                        "paddingAll": "20px"
                    }
                },
                {
                    "type": "bubble",
                    "hero": {
                        "type": "image",
                        "url": "https://i.postimg.cc/1f9rnnsY/voice2.png",
                        "size": "full",
                        "aspectRatio": "20:13",
                        "aspectMode": "cover"
                    },
                    "body": {
                        "type": "box",
                        "layout": "vertical",
                        "contents": [
                            {
                                "type": "text",
                                "text": "第2步：選擇語言",
                                "weight": "bold",
                                "size": "lg",
                                "color": "#2E86AB",
                                "align": "center"
                            },
                            {
                                "type": "text",
                                "text": "在錄音頁面選擇您要使用的語言：\n\n🇹🇼 國語\n🇹🇼 台語\n\n選擇完成後準備開始錄音",
                                "size": "sm",
                                "color": "#666666",
                                "wrap": True,
                                "margin": "md"
                            }
                        ],
                        "paddingAll": "20px"
                    }
                },
                {
                    "type": "bubble",
                    "hero": {
                        "type": "image",
                        "url": "https://i.postimg.cc/svV4QQsH/voice3.png",
                        "size": "full",
                        "aspectRatio": "20:13",
                        "aspectMode": "cover"
                    },
                    "body": {
                        "type": "box",
                        "layout": "vertical",
                        "contents": [
                            {
                                "type": "text",
                                "text": "第3步：開始錄音",
                                "weight": "bold",
                                "size": "lg",
                                "color": "#2E86AB",
                                "align": "center"
                            },
                            {
                                "type": "text",
                                "text": "點擊錄音按鈕開始說話：\n\n• 清楚說出您的問題\n• 錄音完成後點擊停止\n• 系統會自動轉換成文字\n• 文字會直接發送到聊天室",
                                "size": "sm",
                                "color": "#666666",
                                "wrap": True,
                                "margin": "md"
                            }
                        ],
                        "paddingAll": "20px"
                    }
                }
            ]
        }
    }

def create_blood_sugar_tutorial_carousel():
    """創建血糖管理詳細教學 Carousel"""
    return {
        "type": "flex",
        "altText": "血糖管理室教學",
        "contents": {
            "type": "carousel",
            "contents": [
                {
                    "type": "bubble",
                    "hero": {
                        "type": "image",
                        "url": "https://your-image-host.com/blood-step1.jpg",
                        "size": "full",
                        "aspectRatio": "20:13",
                        "aspectMode": "cover"
                    },
                    "body": {
                        "type": "box",
                        "layout": "vertical",
                        "contents": [
                            {
                                "type": "text",
                                "text": "第1步：記錄血糖數值",
                                "weight": "bold",
                                "size": "lg",
                                "color": "#2E86AB",
                                "align": "center"
                            },
                            {
                                "type": "text",
                                "text": "直接輸入血糖數值即可記錄：\n\n• 直接輸入數字：120\n• 加上單位：150mg/dL\n• 加上說明：早餐後血糖 140",
                                "size": "sm",
                                "color": "#666666",
                                "wrap": True,
                                "margin": "md"
                            }
                        ],
                        "paddingAll": "20px"
                    }
                },
                {
                    "type": "bubble",
                    "hero": {
                        "type": "image",
                        "url": "https://your-image-host.com/blood-step2.jpg",
                        "size": "full",
                        "aspectRatio": "20:13",
                        "aspectMode": "cover"
                    },
                    "body": {
                        "type": "box",
                        "layout": "vertical",
                        "contents": [
                            {
                                "type": "text",
                                "text": "第2步：查看歷史記錄",
                                "weight": "bold",
                                "size": "lg",
                                "color": "#2E86AB",
                                "align": "center"
                            },
                            {
                                "type": "text",
                                "text": "輸入關鍵字查看記錄：\n\n• 輸入「報表」\n• 輸入「歷史」\n• 輸入「記錄」\n\n系統會顯示您的血糖趨勢",
                                "size": "sm",
                                "color": "#666666",
                                "wrap": True,
                                "margin": "md"
                            }
                        ],
                        "paddingAll": "20px"
                    }
                },
                {
                    "type": "bubble",
                    "hero": {
                        "type": "image",
                        "url": "https://your-image-host.com/blood-step3.jpg",
                        "size": "full",
                        "aspectRatio": "20:13",
                        "aspectMode": "cover"
                    },
                    "body": {
                        "type": "box",
                        "layout": "vertical",
                        "contents": [
                            {
                                "type": "text",
                                "text": "第3步：生成個人報表",
                                "weight": "bold",
                                "size": "lg",
                                "color": "#2E86AB",
                                "align": "center"
                            },
                            {
                                "type": "text",
                                "text": "系統會自動生成：\n\n• 血糖趨勢圖表\n• 每日平均數值\n• 健康狀態評估\n• 個人化建議",
                                "size": "sm",
                                "color": "#666666",
                                "wrap": True,
                                "margin": "md"
                            }
                        ],
                        "paddingAll": "20px"
                    }
                }
            ]
        }
    }

def create_image_tutorial_carousel():
    """創建影像辨識詳細教學 Carousel"""
    return {
        "type": "flex",
        "altText": "影像辨識教學",
        "contents": {
            "type": "carousel",
            "contents": [
                {
                    "type": "bubble",
                    "hero": {
                        "type": "image",
                        "url": "https://your-image-host.com/image-step1.jpg",
                        "size": "full",
                        "aspectRatio": "20:13",
                        "aspectMode": "cover"
                    },
                    "body": {
                        "type": "box",
                        "layout": "vertical",
                        "contents": [
                            {
                                "type": "text",
                                "text": "第1步：拍攝清楚照片",
                                "weight": "bold",
                                "size": "lg",
                                "color": "#2E86AB",
                                "align": "center"
                            },
                            {
                                "type": "text",
                                "text": "拍攝以下類型的圖片：\n\n• 血糖儀螢幕讀數\n• 藥品包裝或標籤\n• 食物營養標示\n• 醫療報告數據",
                                "size": "sm",
                                "color": "#666666",
                                "wrap": True,
                                "margin": "md"
                            }
                        ],
                        "paddingAll": "20px"
                    }
                },
                {
                    "type": "bubble",
                    "hero": {
                        "type": "image",
                        "url": "https://your-image-host.com/image-step2.jpg",
                        "size": "full",
                        "aspectRatio": "20:13",
                        "aspectMode": "cover"
                    },
                    "body": {
                        "type": "box",
                        "layout": "vertical",
                        "contents": [
                            {
                                "type": "text",
                                "text": "第2步：發送圖片",
                                "weight": "bold",
                                "size": "lg",
                                "color": "#2E86AB",
                                "align": "center"
                            },
                            {
                                "type": "text",
                                "text": "直接在聊天室發送圖片：\n\n• 點擊相機圖示\n• 選擇拍照或從相簿選取\n• 確認圖片清晰可見\n• 發送給糖小護",
                                "size": "sm",
                                "color": "#666666",
                                "wrap": True,
                                "margin": "md"
                            }
                        ],
                        "paddingAll": "20px"
                    }
                },
                {
                    "type": "bubble",
                    "hero": {
                        "type": "image",
                        "url": "https://your-image-host.com/image-step3.jpg",
                        "size": "full",
                        "aspectRatio": "20:13",
                        "aspectMode": "cover"
                    },
                    "body": {
                        "type": "box",
                        "layout": "vertical",
                        "contents": [
                            {
                                "type": "text",
                                "text": "第3步：AI智能分析",
                                "weight": "bold",
                                "size": "lg",
                                "color": "#2E86AB",
                                "align": "center"
                            },
                            {
                                "type": "text",
                                "text": "Gemini AI會自動分析：\n\n• 識別圖片中的文字和數據\n• 理解醫療相關內容\n• 提供專業健康建議\n• 回答相關問題",
                                "size": "sm",
                                "color": "#666666",
                                "wrap": True,
                                "margin": "md"
                            }
                        ],
                        "paddingAll": "20px"
                    }
                }
            ]
        }
    }

# 載入用戶同意狀態
user_consent = load_user_data()

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
1. 你是一位充滿熱情與關懷的專業營養師，請根據以下食物的營養資訊進行分析：
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
    """計算熱量來源佔比並評估糖分含量"""
    total_carb_calories = 0
    total_protein_calories = 0
    total_fat_calories = 0
    total_sugar_calories = 0
    total_calories = 0
    total_sugar_grams = 0

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
        total_sugar_grams += sugar

    # 計算糖分佔總熱量的百分比
    sugar_percent = (total_sugar_calories / total_calories * 100) if total_calories > 0 else 0

    # 評估糖分含量
    if total_sugar_grams > 25 or sugar_percent > 10:
        sugar_label = "高糖 (需注意，可能超過建議攝取)"
    elif total_sugar_grams > 10 or sugar_percent > 5:
        sugar_label = "中糖 (適量攝取)"
    else:
        sugar_label = "低糖"

    return {
        "carbs_calories": round(total_carb_calories, 0),
        "protein_calories": round(total_protein_calories, 0),
        "fat_calories": round(total_fat_calories, 0),
        "sugar_calories": round(total_sugar_calories, 0),
        "total_calories": round(total_calories, 0),
        "total_sugar_grams": round(total_sugar_grams, 1),
        "sugar_percent": round(sugar_percent, 1),
        "sugar_label": sugar_label,
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

# 處理加好友事件
@handler.add(FollowEvent)
def handle_follow(event):
    """處理加好友事件"""
    user_id = event.source.user_id
    
    try:
        # 檢查用戶是否已有狀態記錄
        if user_id in user_consent and isinstance(user_consent[user_id], dict):
            current_status = user_consent[user_id].get("status")
            if current_status:
                print(f"用戶 {user_id} 已有狀態記錄: {current_status}，跳過條款發送")
                # 如果用戶之前有狀態，保持原狀
                return
                
        # 新用戶或狀態無效 → 發送專業的條款頁面
        print(f"新用戶 {user_id} 加入或狀態無效，發送條款")
        flex_message = create_terms_flex_message()
        line_bot_api.reply_message(event.reply_token, FlexSendMessage(
            alt_text=flex_message["altText"],
            contents=flex_message["contents"]
        ))
        
        # 初始化或重置用戶狀態
        user_consent[user_id] = {
            "status": "pending",
            "first_contact": datetime.now().isoformat(),
            "blood_sugar_records": []
        }
        save_user_data(user_consent)
        
    except Exception as e:
        print(f"❌ 處理加好友事件時發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        # 即使發生錯誤，也確保用戶能收到回應
        try:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="歡迎使用糖小護！如果您看不到服務條款，請輸入「重新開始」。")
            )
        except:
            pass

# 處理訊息事件
@handler.add(MessageEvent)
def handle_message(event):
    """處理所有類型的訊息事件"""
    # 檢查消息是否已經處理過
    if event.message.id in global_data_store["processed_messages"]:
        print(f"⚠️ 跳過重複消息: {event.message.id}")
        return

    try:
    # 檢查是否有其他消息正在處理中
        if global_data_store.get("message_lock", False):
            print("⚠️ 另一個消息正在處理中，稍後重試")
            return

        # 設置消息鎖
        global_data_store["message_lock"] = True
        
        user_id = event.source.user_id
        
        # 處理不同類型的消息
        if isinstance(event.message, TextMessage):
            msg = event.message.text.strip()
            print(f"收到: {msg}")
            
            # 檢查是否已經同意
            if user_id not in user_consent:
                # 新用戶 → 發送專業的條款頁面
                flex_message = create_terms_flex_message()
                line_bot_api.reply_message(event.reply_token, FlexSendMessage(
                    alt_text=flex_message["altText"],
                    contents=flex_message["contents"]
                ))
                user_consent[user_id] = {
                    "status": "pending",
                    "first_contact": datetime.now().isoformat(),
                    "blood_sugar_records": []
                }
                save_user_data(user_consent)
                return
            
            # 根據用戶狀態處理消息
            status = user_consent[user_id].get("status", "pending")
            
            if status == "pending":
                # 等待用戶同意條款
                if msg == "同意":
                    welcome_message = create_welcome_message()
                    button_check_message = create_button_check_message()
                    line_bot_api.reply_message(event.reply_token, [
                        FlexSendMessage(
                            alt_text=welcome_message["altText"],
                            contents=welcome_message["contents"]
                        ),
                        button_check_message
                    ])
                    user_consent[user_id]["status"] = "awaiting_button_response"
                    user_consent[user_id]["agreed_time"] = datetime.now().isoformat()
                    save_user_data(user_consent)
                    return
                elif msg == "不同意":
                    reply = "感謝您的回覆。如果您改變心意，歡迎隨時重新開始對話。\n\n為了保護您的隱私，我們將不會保存任何資料。"
                    user_consent[user_id]["status"] = "disagreed"
                    user_consent[user_id]["disagreed_time"] = datetime.now().isoformat()
                    save_user_data(user_consent)
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
                    return
                else:
                    reply = "請點選條款頁面中的「同意並開始使用」或「暫不同意」按鈕，或直接回覆「同意」或「不同意」。"
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
                    return
                    
            elif status == "awaiting_button_response":
                # 處理按鈕確認回應
                if msg == "有":
                    tutorial_choice_message = create_tutorial_choice_message()
                    line_bot_api.reply_message(event.reply_token, tutorial_choice_message)
                    user_consent[user_id]["status"] = "awaiting_tutorial_choice"
                    save_user_data(user_consent)
                    return
                elif msg == "沒有":
                    reply = "沒關係！我們來說明一下：\n\n在我的訊息下方，您會看到一些按鈕，這些按鈕可以幫助您快速選擇回應。\n\n如果您現在看到了，請回覆「有」；如果還是沒看到，請回覆「沒有」。"
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
                    return
                else:
                    reply = "請回覆「有」或「沒有」，讓我知道您是否看到下面的按鈕。"
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
                    return
                    
            elif status == "awaiting_tutorial_choice":
                # 處理教學選擇回應
                if msg == "我要教學":
                    tutorial_carousel = create_tutorial_carousel()
                    line_bot_api.reply_message(event.reply_token, FlexSendMessage(
                        alt_text=tutorial_carousel["altText"],
                        contents=tutorial_carousel["contents"]
                    ))
                    user_consent[user_id]["status"] = "tutorial_shown"
                    save_user_data(user_consent)
                    return
                elif msg == "我不要教學":
                    skip_message = create_skip_tutorial_message()
                    line_bot_api.reply_message(event.reply_token, skip_message)
                    user_consent[user_id]["status"] = "agreed"
                    save_user_data(user_consent)
                    return
                else:
                    reply = "請回覆「我要教學」或「我不要教學」，讓我知道您的選擇。"
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
                    return
                    
            elif status == "tutorial_shown":
                # 處理教學相關回應
                if msg in ["問答教學", "語音教學", "血糖教學", "影像教學"]:
                    tutorial_carousels = {
                        "問答教學": create_qa_tutorial_carousel(),
                        "語音教學": create_voice_tutorial_carousel(),
                        "血糖教學": create_blood_sugar_tutorial_carousel(),
                        "影像教學": create_image_tutorial_carousel()
                    }
                    selected_carousel = tutorial_carousels[msg]
                    line_bot_api.reply_message(event.reply_token, FlexSendMessage(
                        alt_text=selected_carousel["altText"],
                        contents=selected_carousel["contents"]
                    ))
                    user_consent[user_id]["status"] = "detailed_tutorial"
                    save_user_data(user_consent)
                    return
                else:
                    user_consent[user_id]["status"] = "agreed"
                    save_user_data(user_consent)
                    
            elif status == "detailed_tutorial":
                # 用戶看完詳細教學，進入正常使用狀態
                user_consent[user_id]["status"] = "agreed"
                save_user_data(user_consent)
                
            # 處理教學相關指令（全域可觸發，只要已同意就可以使用）
            if status in ["agreed", "tutorial_shown", "detailed_tutorial"]:
                # 教學相關指令
                if msg == "教學" or msg == "功能介紹":
                    tutorial_carousel = create_tutorial_carousel()
                    line_bot_api.reply_message(event.reply_token, FlexSendMessage(
                        alt_text=tutorial_carousel["altText"],
                        contents=tutorial_carousel["contents"]
                    ))
                    return
                elif msg in ["問答教學", "語音教學", "血糖教學", "影像教學"]:
                    tutorial_carousels = {
                        "問答教學": create_qa_tutorial_carousel(),
                        "語音教學": create_voice_tutorial_carousel(),
                        "血糖教學": create_blood_sugar_tutorial_carousel(),
                        "影像教學": create_image_tutorial_carousel()
                    }
                    selected_carousel = tutorial_carousels[msg]
                    line_bot_api.reply_message(event.reply_token, FlexSendMessage(
                        alt_text=selected_carousel["altText"],
                        contents=selected_carousel["contents"]
                    ))
                    return
                
            # 處理一般功能（已同意用戶）
            if status == "agreed":
                # 使用 RAG 生成回答
                print(f"💬 處理一般文字訊息: {msg}")
                _, docs = search_related_content(msg)
                response = generate_answer(msg, docs)
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=response))
                return
                
            elif status == "disagreed":
                if msg == "重新開始":
                    del user_consent[user_id]
                    save_user_data(user_consent)
                    flex_message = create_terms_flex_message()
                    line_bot_api.reply_message(event.reply_token, FlexSendMessage(
                        alt_text=flex_message["altText"],
                        contents=flex_message["contents"]
                    ))
                else:
                    reply = "由於您尚未同意服務條款，目前無法使用糖小護的功能。\n\n如果您想重新開始，請輸入「重新開始」。"
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
                return
            
        elif isinstance(event.message, ImageMessage):
            # 處理圖片訊息
            print("✅ 收到圖片訊息")
            if user_consent[user_id].get("status") == "agreed":
                handle_image_message(event)
            else:
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text="請先同意服務條款才能使用圖片分析功能。")
                )
            
    except LineBotApiError as e:
        print(f"❌ LINE API 錯誤: {str(e)}")
        try:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="⚠️ 系統暫時無法處理您的訊息，請稍後再試。")
            )
        except:
            pass
    except Exception as e:
        print(f"❌ 處理訊息時發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        try:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="⚠️ 系統發生錯誤，請稍後再試。")
            )
        except:
            pass
    finally:
        # 標記消息為已處理
        global_data_store["processed_messages"].add(event.message.id)
        
        # 如果已處理消息數量超過1000，清理舊的記錄
        if len(global_data_store["processed_messages"]) > 1000:
            global_data_store["processed_messages"] = set(list(global_data_store["processed_messages"])[-1000:])
            
        # 釋放消息鎖
        global_data_store["message_lock"] = False

def handle_image_message(event):
    """處理圖片訊息"""
    temp_dir = "temp_images"
    image_path = None
        
    try:
        # 創建臨時文件夾（如果不存在）
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        # 從 LINE 獲取圖片內容
        message_content = line_bot_api.get_message_content(event.message.id)
        
        # 保存圖片到臨時文件
        image_path = os.path.join(temp_dir, f"{event.message.id}.jpg")
        with open(image_path, "wb") as f:
            for chunk in message_content.iter_content():
                f.write(chunk)
        
            # 分析圖片並獲取 Flex Message
            flex_message = analyze_food_image(image_path)
            
            # 回覆 Flex Message
            line_bot_api.reply_message(
                event.reply_token,
                flex_message
            )
    
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
    finally:
        # 清理臨時文件
        if image_path and os.path.exists(image_path):
            try:
                    os.remove(image_path)
            except Exception as e:
                print(f"⚠️ 無法刪除臨時文件: {str(e)}")

if __name__ == "__main__":
    print("啟動糖尿病諮詢 LINE Bot")
    print("功能包含：")
    print("   - RAG 檢索")
    print("   - SAS 答案評估")
    print("   - Fast/Slow path 機制")
    print("   - 食物圖片分析")
    
    # 從環境變數獲取端口號，如果沒有則使用預設值
    port = int(os.environ.get('PORT', 5000))
    
    # 啟動 Flask 應用（生產環境中建議關閉 debug 模式）
    app.run(host="0.0.0.0", port=port, debug=False)