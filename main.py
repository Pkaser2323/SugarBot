from flask import Flask, request
from linebot import LineBotApi, WebhookHandler
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, PostbackEvent,
    QuickReply, QuickReplyButton, DatetimePickerAction, PostbackAction
)
from linebot.exceptions import InvalidSignatureError, LineBotApiError
import os
from datetime import datetime
from linebot.models import ImageSendMessage
import pytz
import blood_sugar
import hmac
import hashlib
import base64

# 設定 Flask 伺服器
app = Flask(__name__)

# 你的 LINE Bot 權杖 (從環境變數讀取)
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

# 檢查環境變數是否正確載入
print(f"✅ Loaded LINE_CHANNEL_ACCESS_TOKEN: {LINE_CHANNEL_ACCESS_TOKEN}")
print(f"✅ Loaded LINE_CHANNEL_SECRET: {LINE_CHANNEL_SECRET}")

if not LINE_CHANNEL_ACCESS_TOKEN:
    raise ValueError("❌ 環境變數 LINE_CHANNEL_ACCESS_TOKEN 未正確設定！")
if not LINE_CHANNEL_SECRET:
    raise ValueError("❌ 環境變數 LINE_CHANNEL_SECRET 未正確設定！")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# 儲存使用者狀態（判斷是否要記錄血糖）
user_states = {}

# 測試 LINE API 連線（啟動時發送一條測試訊息）
try:
    print("✅ Testing LINE API connection by sending a test message")
    line_bot_api.push_message(
        "U4743c3f8d1cfa0a7e6571e10fb2cf5d",  # 你的 user_id
        TextSendMessage(text="這是一條測試訊息，確認 LINE API 是否正常")
    )
    print("✅ Test message sent successfully")
except LineBotApiError as e:
    print(f"❌ Failed to send test message: {str(e)}")

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    print(f"✅ Server time: {datetime.now(pytz.timezone('Asia/Taipei')).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✅ Received request with signature: {signature}")
    print(f"✅ Request body: {body}")

    # 手動計算簽名
    hash = hmac.new(LINE_CHANNEL_SECRET.encode('utf-8'), body.encode('utf-8'), hashlib.sha256).digest()
    calculated_signature = base64.b64encode(hash).decode('utf-8')
    print(f"✅ Calculated signature: {calculated_signature}")

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

#-----------------------訊息欄格式-------------------------------
def create_blood_sugar_message(user_id, date_str):
    try:
        # 查詢指定日期的血糖紀錄
        print(f"✅ Querying blood sugar for user {user_id} on date {date_str}")
        records = blood_sugar.get_blood_sugar_by_date(user_id, date_str)
        print(f"✅ Retrieved records: {records}")

        # 準備訊息內容
        message_text = f"今日血糖紀錄\n({date_str})\n"
        
        if isinstance(records, str):  # 如果返回錯誤訊息
            message_text += records
        elif records:  # 如果有紀錄
            for record in records:
                message_text += f"🔹 {record['time']} - {record['value']} mg/dL\n"
        else: 
            message_text += "尚無血糖紀錄！\n"

        # 最後一行加入「選擇日期」、「新增」、「修改」、「刪除」按鈕
        quick_reply = QuickReply(items=[
            QuickReplyButton(
                action=DatetimePickerAction(
                    label="選擇日期",
                    data="action=select_date",
                    mode="date",
                    initial=date_str,
                    max=datetime.now(pytz.timezone("Asia/Taipei")).strftime("%Y-%m-%d"),
                    min="2020-01-01"
                )
            ),
            QuickReplyButton(
                action=PostbackAction(
                    label="新增",
                    data="action=add_blood_sugar"
                )
            ),
            QuickReplyButton(
                action=PostbackAction(
                    label="修改",
                    data="action=edit_blood_sugar"
                )
            ),
            QuickReplyButton(
                action=PostbackAction(
                    label="刪除",
                    data="action=delete_blood_sugar"
                )
            )
        ])

        return TextSendMessage(text=message_text, quick_reply=quick_reply)
    except Exception as e:
        print(f"❌ Error in create_blood_sugar_message: {str(e)}")
        return TextSendMessage(text=f"❌ 無法顯示血糖紀錄，錯誤：{str(e)}")

#------------------------------個人報表相關----------------------------------------

def create_report_menu_message():
    try:
        quick_reply = QuickReply(items=[
            QuickReplyButton(
                action=PostbackAction(
                    label="今天",
                    data="action=report_today"
                )
            ),
            QuickReplyButton(
                action=PostbackAction(
                    label="最近一週",
                    data="action=report_last_week"
                )
            ),
            QuickReplyButton(
                action=DatetimePickerAction(
                    label="查看更多日期",
                    data="action=report_select_date",
                    mode="date",
                    initial=datetime.now(pytz.timezone("Asia/Taipei")).strftime("%Y-%m-%d"),
                    max=datetime.now(pytz.timezone("Asia/Taipei")).strftime("%Y-%m-%d"),
                    min="2020-01-01"
                )
            )
        ])
        return TextSendMessage(text="請選擇要查看的報表時間範圍：", quick_reply=quick_reply)
    except Exception as e:
        print(f"❌ Error in create_report_menu_message: {str(e)}")
        return TextSendMessage(text=f"❌ 無法顯示報表選單，錯誤：{str(e)}")





def show_records_for_edit(user_id, date_str):
    try:
        print(f"✅ Showing records for edit for user {user_id} on date {date_str}")
        records = blood_sugar.get_blood_sugar_by_date(user_id, date_str)

        message_text = f"請選擇要修改的血糖紀錄\n({date_str})\n"
        
        if isinstance(records, str):
            message_text += records
            return TextSendMessage(text=message_text)
        elif not records:
            message_text += "尚無血糖紀錄！\n"
            return TextSendMessage(text=message_text)

        # 將每筆紀錄轉為按鈕
        quick_reply_items = []
        for idx, record in enumerate(records):
            button_label = f"{record['time']} - {record['value']} mg/dL"
            quick_reply_items.append(
                QuickReplyButton(
                    action=PostbackAction(
                        label=button_label[:20],  # LINE 按鈕標籤最多 20 字元
                        data=f"action=edit_record&index={idx}"
                    )
                )
            )

        return TextSendMessage(text=message_text, quick_reply=QuickReply(items=quick_reply_items))
    except Exception as e:
        print(f"❌ Error in show_records_for_edit: {str(e)}")
        return TextSendMessage(text=f"❌ 無法顯示血糖紀錄，錯誤：{str(e)}")







def show_records_for_delete(user_id, date_str):
    try:
        print(f"✅ Showing records for delete for user {user_id} on date {date_str}")
        records = blood_sugar.get_blood_sugar_by_date(user_id, date_str)

        message_text = f"請選擇要刪除的血糖紀錄\n({date_str})\n"
        
        if isinstance(records, str):
            message_text += records
            return TextSendMessage(text=message_text)
        elif not records:
            message_text += "尚無血糖紀錄！\n"
            return TextSendMessage(text=message_text)

        # 將每筆紀錄轉為按鈕
        quick_reply_items = []
        for idx, record in enumerate(records):
            button_label = f"{record['time']} - {record['value']} mg/dL"
            quick_reply_items.append(
                QuickReplyButton(
                    action=PostbackAction(
                        label=button_label[:20],  # LINE 按鈕標籤最多 20 字元
                        data=f"action=delete_record&index={idx}"
                    )
                )
            )

        return TextSendMessage(text=message_text, quick_reply=QuickReply(items=quick_reply_items))
    except Exception as e:
        print(f"❌ Error in show_records_for_delete: {str(e)}")
        return TextSendMessage(text=f"❌ 無法顯示血糖紀錄，錯誤：{str(e)}")




@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    print(f"✅ 收到訊息：{event.message.text}")  # Debug 訊息
    user_id = event.source.user_id
    message_text = event.message.text.strip()

    # 1️⃣ 使用者輸入「血糖紀錄」，預設顯示今天的紀錄
    if message_text == "血糖紀錄":
        tz = pytz.timezone("Asia/Taipei")
        today = datetime.now(tz).strftime("%Y-%m-%d")  # 取得今天的日期
        print(f"✅ Generating blood sugar message for date: {today}")
        message = create_blood_sugar_message(user_id, today)
        try:
            print(f"✅ Attempting to reply with token: {event.reply_token}")
            line_bot_api.reply_message(event.reply_token, message)
            print("✅ Reply message sent successfully")
        except LineBotApiError as e:
            print(f"❌ Failed to reply message: {str(e)}")
        return
    # 新增：使用者輸入「語音轉文字」時，回傳 LIFF 語音網頁連結
    if message_text == "語音轉文字":
        liff_url = "https://liff.line.me/2007818922-R4NG21no"
        try:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"請點擊進行語音輸入：{liff_url}"))
        except LineBotApiError as e:
            print(f"❌ Failed to reply message: {str(e)}")
        return
    # 2️⃣ 使用者輸入「個人報表」，顯示報表選單
    if message_text == "個人報表":
        print(f"✅ Generating report menu for user {user_id}")
        message = create_report_menu_message()
        try:
            print(f"✅ Attempting to reply with token: {event.reply_token}")
            line_bot_api.reply_message(event.reply_token, message)
            print("✅ Reply message sent successfully")
        except LineBotApiError as e:
            print(f"❌ Failed to reply message: {str(e)}")
        return
    
    # 2️⃣ 如果使用者正在等待輸入血糖值，則記錄血糖
    if user_states.get(user_id) == "waiting_for_bloodsugar":
        try:
            blood_sugar_value = int(message_text)  # 確保是數字
            print(f"✅ Recording blood sugar value: {blood_sugar_value}")
            response_text = blood_sugar.record_blood_sugar(user_id, blood_sugar_value)
            if response_text.startswith("✅"):
                # 取得今日日期
                tz = pytz.timezone("Asia/Taipei")
                today = datetime.now(tz).strftime("%Y-%m-%d")
                # 生成今日紀錄訊息
                today_records_message = create_blood_sugar_message(user_id, today)
                # 在訊息前加上「已記錄！」和分隔線
                final_message = TextSendMessage(
                    text=f"已記錄！\n-------------\n{today_records_message.text}",
                    quick_reply=today_records_message.quick_reply
                )
                user_states[user_id] = None  # 清除狀態
                try:
                    print(f"✅ Attempting to reply with token: {event.reply_token}")
                    line_bot_api.reply_message(event.reply_token, final_message)
                    print("✅ Reply message sent successfully")
                except LineBotApiError as e:
                    print(f"❌ Failed to reply message: {str(e)}")
            else:
                # 記錄失敗，回傳錯誤訊息
                try:
                    print(f"✅ Attempting to reply with token: {event.reply_token}")
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=response_text))
                    print("✅ Reply message sent successfully")
                except LineBotApiError as e:
                    print(f"❌ Failed to reply message: {str(e)}")
        except ValueError:
            try:
                print(f"✅ Attempting to reply with token: {event.reply_token}")
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 請輸入有效的數字！"))
                print("✅ Reply message sent successfully")
            except LineBotApiError as e:
                print(f"❌ Failed to reply message: {str(e)}")
        return
    



#修改

    if user_states.get(user_id, {}).get("state") == "editing_bloodsugar":
        try:
            new_value = int(message_text)  # 確保是數字
            state = user_states[user_id]
            date_str = state["date"]
            record_index = state["index"]
            print(f"✅ Updating blood sugar for user {user_id} on {date_str}, index {record_index}")
            response_text = blood_sugar.update_blood_sugar(user_id, date_str, record_index, new_value)
            
            # 檢查是否更新成功
            if response_text.startswith("✅"):
                # 生成今日紀錄訊息
                today_records_message = create_blood_sugar_message(user_id, date_str)
                # 在訊息前加上「已修改！」和分隔線
                final_message = TextSendMessage(
                    text=f"已修改！\n-------------\n{today_records_message.text}",
                    quick_reply=today_records_message.quick_reply
                )
                user_states[user_id] = None  # 清除狀態
                try:
                    print(f"✅ Attempting to reply with token: {event.reply_token}")
                    line_bot_api.reply_message(event.reply_token, final_message)
                    print("✅ Reply message sent successfully")
                except LineBotApiError as e:
                    print(f"❌ Failed to reply message: {str(e)}")
            else:
                # 更新失敗，回傳錯誤訊息
                try:
                    print(f"✅ Attempting to reply with token: {event.reply_token}")
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=response_text))
                    print("✅ Reply message sent successfully")
                except LineBotApiError as e:
                    print(f"❌ Failed to reply message: {str(e)}")
        except ValueError:
            try:
                print(f"✅ Attempting to reply with token: {event.reply_token}")
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 請輸入有效的數字！"))
                print("✅ Reply message sent successfully")
            except LineBotApiError as e:
                print(f"❌ Failed to reply message: {str(e)}")
        return



    # 3️⃣ 預設回應，提示使用者可以做什麼
    response_text = "📋 請選擇操作：\n- 輸入「血糖紀錄」查看紀錄"
    try:
        print(f"✅ Attempting to reply with token: {event.reply_token}")
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=response_text))
        print("✅ Reply message sent successfully")
    except LineBotApiError as e:
        print(f"❌ Failed to reply message: {str(e)}")

# 處理 Postback 事件（按鈕點擊）
@handler.add(PostbackEvent)
def handle_postback(event):
    user_id = event.source.user_id
    postback_data = event.postback.data
    print(f"✅ Handling postback: {postback_data}")

    # 1️⃣ 使用者點擊「選擇日期」
    if postback_data == "action=select_date":
        selected_date = event.postback.params.get("date")  # 安全地取得日期
        if not selected_date:
            print("❌ No date selected in postback params")
            try:
                print(f"✅ Attempting to reply with token: {event.reply_token}")
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 請選擇一個日期！"))
                print("✅ Reply message sent successfully")
            except LineBotApiError as e:
                print(f"❌ Failed to reply message: {str(e)}")
            return

        print(f"✅ Selected date: {selected_date}")
        message = create_blood_sugar_message(user_id, selected_date)
        try:
            print(f"✅ Attempting to reply with token: {event.reply_token}")
            line_bot_api.reply_message(event.reply_token, message)
            print("✅ Reply message sent successfully")
        except LineBotApiError as e:
            print(f"❌ Failed to reply message: {str(e)}")
        return

    # 2️⃣ 使用者點擊「新增」按鈕
    if postback_data == "action=add_blood_sugar":
        print(f"✅ User {user_id} clicked 'add_blood_sugar'")
        user_states[user_id] = "waiting_for_bloodsugar"
        try:
            print(f"✅ Attempting to reply with token: {event.reply_token}")
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="請輸入血糖"))
            print("✅ Reply message sent successfully")
        except LineBotApiError as e:
            print(f"❌ Failed to reply message: {str(e)}")
        return

    # 3️⃣ 使用者點擊「修改」或「刪除」按鈕（目前只顯示畫面，功能未實作）
    if postback_data in ["action=edit_blood_sugar", "action=delete_blood_sugar"]:
        action = "修改" if postback_data == "action=edit_blood_sugar" else "刪除"
        print(f"✅ User {user_id} clicked '{action}'")
        
        if action == "修改":
            # 只能修改今日紀錄
            tz = pytz.timezone("Asia/Taipei")
            today = datetime.now(tz).strftime("%Y-%m-%d")
            message = show_records_for_edit(user_id, today)
            try:
                print(f"✅ Attempting to reply with token: {event.reply_token}")
                line_bot_api.reply_message(event.reply_token, message)
                print("✅ Reply message sent successfully")
            except LineBotApiError as e:
                print(f"❌ Failed to reply message: {str(e)}")
            return
        else:  # 刪除
        # 只能刪除今日紀錄
            tz = pytz.timezone("Asia/Taipei")
            today = datetime.now(tz).strftime("%Y-%m-%d")
            message = show_records_for_delete(user_id, today)
            try:
                print(f"✅ Attempting to reply with token: {event.reply_token}")
                line_bot_api.reply_message(event.reply_token, message)
                print("✅ Reply message sent successfully")
            except LineBotApiError as e:
                print(f"❌ Failed to reply message: {str(e)}")
            return
    


    if postback_data.startswith("action=edit_record"):
        import re
        index = int(re.search(r"index=(\d+)", postback_data).group(1))
        print(f"✅ User {user_id} selected record index {index} to edit")
        
        # 儲存使用者狀態，記錄正在修改哪筆紀錄
        tz = pytz.timezone("Asia/Taipei")
        today = datetime.now(tz).strftime("%Y-%m-%d")
        user_states[user_id] = {"state": "editing_bloodsugar", "date": today, "index": index}
        
        try:
            print(f"✅ Attempting to reply with token: {event.reply_token}")
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="請輸入新的血糖值"))
            print("✅ Reply message sent successfully")
        except LineBotApiError as e:
            print(f"❌ Failed to reply message: {str(e)}")
        return


    if postback_data.startswith("action=delete_record"):
        import re
        index = int(re.search(r"index=(\d+)", postback_data).group(1))
        print(f"✅ User {user_id} selected record index {index} to delete")
        
        # 執行刪除
        tz = pytz.timezone("Asia/Taipei")
        today = datetime.now(tz).strftime("%Y-%m-%d")
        response_text = blood_sugar.delete_blood_sugar(user_id, today, index)
        
        try:
            print(f"✅ Attempting to reply with token: {event.reply_token}")
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=response_text))
            print("✅ Reply message sent successfully")
        except LineBotApiError as e:
            print(f"❌ Failed to reply message: {str(e)}")
        return


#------------------------------------------------
    if postback_data == "action=report_today":
        print(f"✅ User {user_id} clicked 'report_today'")
        tz = pytz.timezone("Asia/Taipei")
        today = datetime.now(tz).strftime("%Y-%m-%d")
        try:
            records = blood_sugar.get_blood_sugar_by_date(user_id, today)
            print(f"✅ Retrieved records for {today}: {records}")
            
            if isinstance(records, str):  # 查詢錯誤
                message = TextSendMessage(text=records)
            elif not records:  # 無紀錄
                message = TextSendMessage(text="今天還沒有記錄血糖喔!")
            else:  # 有紀錄，generate_blood_sugar_chart已包含上傳並返回URL
                image_url = blood_sugar.generate_blood_sugar_chart(user_id, records, period="today")
                if isinstance(image_url, str) and image_url.startswith("❌"):
                    message = TextSendMessage(text=image_url)
                else:
                    message = ImageSendMessage(original_content_url=image_url, preview_image_url=image_url)
            
            print(f"✅ Attempting to reply with token: {event.reply_token}")
            line_bot_api.reply_message(event.reply_token, message)
            print("✅ Reply message sent successfully")
        except Exception as e:
            print(f"❌ Error in report_today: {str(e)}")
            try:
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"❌ 無法生成報表，錯誤：{str(e)}"))
            except LineBotApiError as e:
                print(f"❌ Failed to reply message: {str(e)}")
        return





# ✅ 健康檢查路由，確保 UptimeRobot 可以 Ping Render
@app.route("/health", methods=["GET"])
def health_check():
    return "OK", 200  # 讓 UptimeRobot 知道伺服器正常運行

# ✅ 確保 Flask 伺服器正確啟動
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
