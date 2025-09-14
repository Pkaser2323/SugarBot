import requests
import logging
import os
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 設定 FatSecret API 金鑰（從環境變數讀取）
CLIENT_ID = os.getenv("FATSECRET_CLIENT_ID")
CLIENT_SECRET = os.getenv("FATSECRET_CLIENT_SECRET")

# 檢查 API 金鑰是否存在
if not CLIENT_ID or not CLIENT_SECRET:
    print("⚠️ 警告：未設定 FATSECRET_CLIENT_ID 或 FATSECRET_CLIENT_SECRET 環境變數")
    print("請在 .env 檔案中設定這些變數")

# OAuth 2.0 Token 請求 URL
TOKEN_URL = "https://oauth.fatsecret.com/connect/token"
FATSECRET_SEARCH_URL = "https://platform.fatsecret.com/rest/server.api"

def get_fatsecret_token():
    """
    取得 FatSecret 的 OAuth 2.0 Access Token，使用 'premier' scope
    """
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "scope": "premier"
    }
    response = requests.post(TOKEN_URL, headers=headers, data=data, auth=(CLIENT_ID, CLIENT_SECRET))

    if response.status_code == 200:
        token_info = response.json()
        access_token = token_info.get("access_token")
        print("✅ 成功獲取 Access Token")
        return access_token
    else:
        logging.error(f"❌ 獲取 Access Token 失敗: {response.text}")
        return None


def search_food_with_fatsecret(food_query):
    """
    搜尋 FatSecret API `foods.search.v3` 來查找 **最相關** 的食物資訊
    """
    access_token = get_fatsecret_token()
    if not access_token:
        return {"error": "❌ 無法取得 FatSecret API 授權"}

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    params = {
        "method": "foods.search.v3",
        "search_expression": food_query,
        "format": "json",
        "max_results": 10  # 增加 max_results 以便篩選
    }

    response = requests.get(FATSECRET_SEARCH_URL, headers=headers, params=params)

    if response.status_code != 200:
        logging.error(f"❌ FatSecret API 解析失敗: {response.text}")
        return {"error": f"FatSecret API 解析失敗: {response.text}"}

    food_data = response.json()
    logging.info("🍽️ FatSecret 解析結果: " + str(food_data))

    # 取得所有搜尋結果
    foods_list = food_data.get("foods_search", {}).get("results", {}).get("food", [])

    if not foods_list:
        return {"error": f"❌ FatSecret 沒有找到 {food_query} 的食物資訊"}

    # **1️⃣ 優先篩選出 `Generic`（不包含品牌的）**
    generic_foods = [food for food in foods_list if food.get("food_type") == "Generic"]

    if not generic_foods:
        logging.warning(f"⚠️ 只找到品牌食品，可能不是最佳匹配：{food_query}")
        generic_foods = foods_list  # 如果沒有 Generic，就用所有食物

    # **2️⃣ 優先選擇名稱完全匹配 `food_query` 的**
    exact_match = next((food for food in generic_foods if food["food_name"].lower() == food_query.lower()), None)
    if exact_match:
        selected_food = exact_match
    else:
        # **3️⃣ 選擇名稱最短、最相關的**
        selected_food = min(generic_foods, key=lambda food: len(food["food_name"]))

    food_name = selected_food.get("food_name", "未知")
    food_id = selected_food.get("food_id")

    if not food_id:
        return {"error": f"❌ 找不到 {food_name} 的 food_id"}

    # 取得該食物的詳細營養資訊
    return get_food_details(food_id, food_name, access_token)  # **回傳字典**


def get_food_details(food_id, food_name, access_token):
    """
    使用 FatSecret API `food.get.v2` 來獲取指定食物 ID 的詳細營養資訊
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    params = {
        "method": "food.get.v2",
        "food_id": food_id,
        "format": "json"
    }

    response = requests.get(FATSECRET_SEARCH_URL, headers=headers, params=params)

    if response.status_code != 200:
        logging.error(f"❌ 無法獲取 {food_name} 的營養資訊: {response.text}")
        return {"error": f"無法獲取 {food_name} 的營養資訊"}  # **回傳字典，而不是字串**

    food_data = response.json()
    logging.info(f"📊 {food_name} 的詳細資訊: {food_data}")

    # 解析食物營養資訊
    food_info = food_data.get("food", {})
    servings = food_info.get("servings", {}).get("serving", [])

    if isinstance(servings, list) and len(servings) > 0:  # 如果有多個 serving，取第一個
        serving = servings[0]
    elif isinstance(servings, dict):  # 如果只有一個 serving，直接使用
        serving = servings
    else:
        return {"error": f"{food_name} 沒有可用的營養數據"}  # **回傳字典**

    # 取得營養成分
    nutrition_data = {
        "food_name": food_name,
        "serving_size": serving.get("serving_description", "未知"),
        "calories": serving.get("calories", "N/A"),
        "carbohydrate": serving.get("carbohydrate", "N/A"),
        "protein": serving.get("protein", "N/A"),
        "fat": serving.get("fat", "N/A"),
        "sugar": serving.get("sugar", "N/A"),
        "fiber": serving.get("fiber", "N/A"),
        "sodium": serving.get("sodium", "N/A"),
    }

    return nutrition_data  # **回傳字典，而不是字串**


# 測試函式 (搜尋 "apple")
#print(search_food_with_fatsecret("apple"))
