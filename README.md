# 糖尿病諮詢 LINE Bot

## 主要功能

1. 問答系統
   - 使用 RAG (Retrieval-Augmented Generation) 技術
   - Fast/Slow path 機制提升回答準確度
   - SAS 答案評估確保回答品質

2. 食物圖片分析
   - 使用 Gemini Vision 識別食物
   - 透過 FatSecret API 獲取營養資訊
   - 提供熱量來源分析和糖尿病飲食建議
   - 美觀的 Flex Message 卡片呈現

## 安裝與設置

1. 安裝依賴套件：
```bash
pip install -r requirements.txt
```

2. 設置環境變數（創建 .env 文件）：
```
GOOGLE_API_KEY=your_gemini_api_key
LINE_ACCESS_TOKEN=your_line_access_token
LINE_SECRET=your_line_secret

在 ../FatSecret 創建.env
FATSECRET_CLIENT_ID = your_client_id
FATSECRET_CLIENT_SECRET = your_client_secret
```

3. 準備必要的模型和資料：
   - 確保 `sas_model` 目錄中有必要的模型文件
   - 準備 `datacsv/a_topic_analyzed_processed.csv` 用於建立向量資料庫

## 核心函數說明

### 食物圖片分析流程

1. `analyze_food_image(image_path)`
   - 主要的圖片分析入口函數
   - 使用 Gemini Vision 識別食物
   - 調用 FatSecret API 獲取營養資訊
   - 生成 Flex Message 回應

2. `translate_to_chinese(english_text)`
   - 將英文食物名稱翻譯成繁體中文
   - 使用 Gemini AI 進行精確翻譯

3. `analyze_nutrition_for_flex(nutrition_data)`
   - 分析營養數據
   - 生成優點、風險和建議
   - 返回 JSON 格式的分析結果

4. `calculate_calorie_sources(nutrition_data_list)`
   - 計算食物的熱量來源佔比
   - 包含碳水、蛋白質、脂肪、糖分等
   - 處理估算值的情況

### 問答系統核心函數

1. `generate_answer(query, docs)`
   - 使用 Fast/Slow path 機制生成回答
   - 整合檢索結果和 AI 生成
   - 確保回答品質和相關性

2. `generate_subqueries(question)`
   - 將複雜問題拆解為子問題
   - 提升回答的完整性

3. `predict_pos_prob(model, questions, answers)`
   - SAS 模型評估答案品質
   - 計算答案的相關度分數

4. `search_related_content(retriever, query)`
   - 在向量資料庫中搜索相關內容
   - 返回最相關的文檔

### LINE Bot 處理函數

1. `handle_message(event)`
   - 處理文字訊息
   - 調用問答系統生成回應

2. `handle_image(event)`
   - 處理圖片訊息
   - 調用食物分析系統

## Flex Message 客製化

如果要修改 Flex Message 的顯示格式，需要編輯以下文件：

1. `flexMessage.py`
   - `generate_flex_message()`: 基本 Flex Message 模板
   - `generate_carousel_flex()`: 輪播式 Flex Message
   - `generate_calorie_source_flex_message()`: 熱量來源分析模板

主要修改點：
- 卡片樣式和顏色
- 內容排版和字體
- 按鈕和互動元素
- 圖表和視覺化效果

## 食物辨識完整流程

1. 圖片接收和處理：
   - `handle_image()` 接收 LINE 圖片
   - 保存為臨時文件

2. 食物識別：
   - `analyze_food_image()` 使用 Gemini Vision
   - 提取英文食物名稱

3. 營養資訊獲取：
   - `search_food_with_fatsecret()` 查詢 FatSecret API
   - 獲取詳細營養成分

4. 資料處理和分析：
   - `translate_to_chinese()` 翻譯食物名稱
   - `calculate_calorie_sources()` 計算熱量佔比
   - `analyze_nutrition_for_flex()` 生成營養建議

5. 結果呈現：
   - `generate_calorie_source_flex_message()` 生成美觀的卡片
   - LINE Bot 發送回應


## 注意事項

- 請確保所有環境變數都已正確設置
- 首次運行時會自動建立向量資料庫，可能需要一些時間
- 建議在生產環境中關閉 debug 模式
- 請定期檢查 API 金鑰的有效性




3. 確認模型檔案位置：
- `sas_model/model.safetensors`：SAS 評估模型（約 1GB）
- `sas_model/config.json`：模型配置
- `sas_model/vocab.txt`：詞彙表

### 下載 SAS 模型

您需要從 Hugging Face 下載 SAS 模型檔案：

1. 訪問模型頁面：[SAS_Model on Hugging Face](https://huggingface.co/Pkaser2323/SAS_Model)
2. 下載以下檔案並放入 `sas_model/` 目錄：
   - `model.safetensors`
   - `config.json`
   - `vocab.txt`
   - `best_params.json`

或使用 Hugging Face CLI 下載：
```bash
# 安裝 Hugging Face CLI
pip install -U huggingface_hub

# 下載模型檔案
huggingface-cli download Pkaser2323/SAS_Model --local-dir ./sas_model/
```

## 模型檔案說明

SAS 模型檔案結構：
```
sas_model/
├── model.safetensors    # 主要模型檔案（使用 Git LFS 管理）
├── config.json          # 模型配置
├── vocab.txt           # 詞彙表
└── best_params.json    # 最佳參數設定
```

"# SugarBot_" 
