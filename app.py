from flask import Flask, request
import json
import os
from datetime import datetime
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageSendMessage,
    FlexSendMessage, PostbackEvent, PostbackAction,
    QuickReply, QuickReplyButton, MessageAction
)

app = Flask(__name__)

# 用戶數據文件路徑
USER_DATA_FILE = "user_data.json"

def load_user_data():
    """載入用戶數據"""
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_user_data(data):
    """保存用戶數據"""
    try:
        with open(USER_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存用戶數據失敗: {e}")

# 載入用戶同意狀態
user_consent = load_user_data()

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
                        "url": "https://your-image-host.com/voice-step1.jpg",
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
                        "url": "https://your-image-host.com/voice-step2.jpg",
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
                        "url": "https://your-image-host.com/voice-step3.jpg",
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

def create_skip_tutorial_message():
    """創建跳過教學的祝福訊息"""
    return TextSendMessage(
        text="好的，那祝你使用愉快！🍭\n\n如果之後想了解功能，隨時都可以詢問我喔～\n\n現在就開始記錄您的血糖數值或詢問健康問題吧！"
    )

def create_main_welcome_message():
    """創建主要歡迎訊息 - 替代原本LINE後台設定"""
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
                        "text": "歡迎使用糖小護",
                        "weight": "bold",
                        "size": "xl",
                        "color": "#FFFFFF",
                        "align": "center"
                    },
                    {
                        "type": "text",
                        "text": "您的專屬健康管理助手",
                        "size": "md",
                        "color": "#E6F3FF",
                        "align": "center",
                        "margin": "sm"
                    }
                ],
                "backgroundColor": "#2E86AB",
                "paddingAll": "20px"
            },
            "body": {
                "type": "box",
                "layout": "vertical",
                "contents": [
                    {
                        "type": "text",
                        "text": "主要功能",
                        "weight": "bold",
                        "size": "md",
                        "color": "#2E86AB",
                        "margin": "md"
                    },
                    {
                        "type": "separator",
                        "margin": "md",
                        "color": "#E6F3FF"
                    },
                    {
                        "type": "text",
                        "text": "血糖管理室\n健康諮詢服務\n個人化報表\n數據分析追蹤",
                        "size": "sm",
                        "color": "#666666",
                        "wrap": True,
                        "margin": "lg"
                    },
                    {
                        "type": "separator",
                        "margin": "lg",
                        "color": "#E6F3FF"
                    },
                    {
                        "type": "text",
                        "text": "使用方式",
                        "weight": "bold",
                        "size": "md",
                        "color": "#2E86AB",
                        "margin": "lg"
                    },
                    {
                        "type": "text",
                        "text": "• 直接輸入血糖數值進行記錄\n• 輸入「報表」查看歷史數據\n• 輸入健康問題獲得建議",
                        "size": "sm",
                        "color": "#666666",
                        "wrap": True,
                        "margin": "sm"
                    },
                    {
                        "type": "text",
                        "text": "現在就開始記錄您的第一筆血糖數值吧！",
                        "size": "sm",
                        "color": "#5A9FD4",
                        "wrap": True,
                        "margin": "lg",
                        "align": "center",
                        "weight": "bold"
                    }
                ],
                "paddingAll": "20px"
            }
        }
    }

@app.route("/callback", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)
    try:
        json_data = json.loads(body)
        access_token = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
        secret = os.environ.get('LINE_CHANNEL_SECRET')
        
        if not access_token or not secret:
            print("錯誤: LINE_CHANNEL_ACCESS_TOKEN 或 LINE_CHANNEL_SECRET 環境變數未設定")
            return
        line_bot_api = LineBotApi(access_token)
        handler = WebhookHandler(secret)
        signature = request.headers['X-Line-Signature']
        handler.handle(body, signature)

        event = json_data['events'][0]
        event_type = event['type']
        user_id = event['source']['userId']  # 使用者 ID
        
        # 某些事件沒有 replyToken (如 unfollow)
        tk = event.get('replyToken')
        if not tk:
            print(f"事件類型 {event_type} 沒有 replyToken，忽略")
            return "OK"
        
        # 處理加好友事件
        if event_type == 'follow':
            # 新用戶加入 → 發送專業的條款頁面
            flex_message = create_terms_flex_message()
            line_bot_api.reply_message(tk, FlexSendMessage(
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

        elif event_type == 'message':
            msg_type = event['message']['type']
            if msg_type == 'text':
                msg = event['message']['text']
                print(f"收到: {msg}")

                # 檢查是否已經同意
                if user_id not in user_consent:
                    # 新用戶 → 發送專業的條款頁面
                    flex_message = create_terms_flex_message()
                    line_bot_api.reply_message(tk, FlexSendMessage(
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

                elif user_consent[user_id].get("status") == "pending":
                    # 等待用戶回覆
                    if msg == "同意":
                        # 發送條款完成訊息 + 直接發送按鈕確認訊息
                        welcome_message = create_welcome_message()
                        button_check_message = create_button_check_message()
                        
                        # 發送兩條訊息：條款完成 + 按鈕確認
                        line_bot_api.reply_message(tk, [
                            FlexSendMessage(
                                alt_text=welcome_message["altText"],
                                contents=welcome_message["contents"]
                            ),
                            button_check_message
                        ])
                        
                        user_consent[user_id]["status"] = "awaiting_button_response"  # 直接設為等待按鈕回應
                        user_consent[user_id]["agreed_time"] = datetime.now().isoformat()
                        save_user_data(user_consent)
                        return
                    elif msg == "不同意":
                        reply = "感謝您的回覆。如果您改變心意，歡迎隨時重新開始對話。\n\n為了保護您的隱私，我們將不會保存任何資料。"
                        user_consent[user_id]["status"] = "disagreed"
                        user_consent[user_id]["disagreed_time"] = datetime.now().isoformat()
                        save_user_data(user_consent)
                    else:
                        reply = "請點選條款頁面中的「同意並開始使用」或「暫不同意」按鈕，或直接回覆「同意」或「不同意」。"

                elif user_consent[user_id].get("status") == "awaiting_button_response":
                    # 處理按鈕確認回應
                    if msg == "有":
                        # 用戶看到按鈕了，詢問是否要教學
                        tutorial_choice_message = create_tutorial_choice_message()
                        line_bot_api.reply_message(tk, tutorial_choice_message)
                        user_consent[user_id]["status"] = "awaiting_tutorial_choice"
                        save_user_data(user_consent)
                        return
                    elif msg == "沒有":
                        # 用戶沒看到按鈕，提供說明
                        reply = "沒關係！我們來說明一下：\n\n在我的訊息下方，您會看到一些按鈕，這些按鈕可以幫助您快速選擇回應。\n\n如果您現在看到了，請回覆「有」；如果還是沒看到，請回覆「沒有」。"
                    else:
                        reply = "請回覆「有」或「沒有」，讓我知道您是否看到下面的按鈕。"

                elif user_consent[user_id].get("status") == "awaiting_tutorial_choice":
                    # 處理教學選擇回應
                    if msg == "我要教學":
                        # 發送5頁功能介紹carousel
                        tutorial_carousel = create_tutorial_carousel()
                        line_bot_api.reply_message(tk, FlexSendMessage(
                            alt_text=tutorial_carousel["altText"],
                            contents=tutorial_carousel["contents"]
                        ))
                        user_consent[user_id]["status"] = "tutorial_shown"
                        save_user_data(user_consent)
                        return
                    elif msg == "我不要教學":
                        # 發送跳過教學祝福訊息
                        skip_message = create_skip_tutorial_message()
                        line_bot_api.reply_message(tk, skip_message)
                        user_consent[user_id]["status"] = "agreed"  # 直接進入正常使用狀態
                        save_user_data(user_consent)
                        return
                    else:
                        reply = "請回覆「我要教學」或「我不要教學」，讓我知道您的選擇。"

                elif user_consent[user_id].get("status") == "tutorial_shown":
                    # 教學已顯示，處理教學相關回應或進入正常功能
                    if msg in ["問答教學", "語音教學", "血糖教學", "影像教學"]:
                        # 根據不同的教學選擇發送對應的詳細教學Carousel
                        tutorial_carousels = {
                            "問答教學": create_qa_tutorial_carousel(),
                            "語音教學": create_voice_tutorial_carousel(),
                            "血糖教學": create_blood_sugar_tutorial_carousel(),
                            "影像教學": create_image_tutorial_carousel()
                        }
                        
                        selected_carousel = tutorial_carousels[msg]
                        
                        # 發送詳細教學Carousel
                        line_bot_api.reply_message(tk, FlexSendMessage(
                            alt_text=selected_carousel["altText"],
                            contents=selected_carousel["contents"]
                        ))
                        
                        user_consent[user_id]["status"] = "detailed_tutorial"  # 設為詳細教學狀態
                        save_user_data(user_consent)
                        return
                    else:
                        # 其他訊息，更新狀態並繼續處理正常功能邏輯
                        user_consent[user_id]["status"] = "agreed"
                        save_user_data(user_consent)
                
                elif user_consent[user_id].get("status") == "detailed_tutorial":
                    # 用戶看完詳細教學，任何訊息都進入正常使用狀態
                    user_consent[user_id]["status"] = "agreed"
                    save_user_data(user_consent)
                    # 繼續處理正常功能邏輯

                else:
                    # 已經有狀態了
                    if user_consent[user_id].get("status") == "agreed":
                        # 用戶已完成引導，準備接收RAG功能
                        if msg == "教學" or msg == "功能介紹":
                            # 重新顯示功能介紹carousel
                            tutorial_carousel = create_tutorial_carousel()
                            line_bot_api.reply_message(tk, FlexSendMessage(
                                alt_text=tutorial_carousel["altText"],
                                contents=tutorial_carousel["contents"]
                            ))
                            user_consent[user_id]["status"] = "tutorial_shown"
                            save_user_data(user_consent)
                            return
                        else:
                            # 其他訊息 - 準備接收RAG功能
                            reply = f"💬 您好！我是糖小護，您的專屬健康管理助手。\n\n🔧 RAG智能問答系統整合中，敬請期待！\n\n如需重新查看功能介紹，請輸入「教學」。"
                    elif user_consent[user_id].get("status") == "disagreed":
                        reply = "由於您尚未同意服務條款，目前無法使用糖小護的功能。\n\n如果您想重新開始，請輸入「重新開始」。"
                        if msg == "重新開始":
                            del user_consent[user_id]
                            save_user_data(user_consent)
                            flex_message = create_terms_flex_message()
                            line_bot_api.reply_message(tk, FlexSendMessage(
                                alt_text=flex_message["altText"],
                                contents=flex_message["contents"]
                            ))
                            return
            else:
                reply = "💬 糖小護收到您的訊息！\n\n🔧 多媒體功能整合中，敬請期待！"
        
        else:
            # 其他事件類型 (unfollow, postback 等)
            return "OK"

        print("回覆:", reply)
        line_bot_api.reply_message(tk, TextSendMessage(reply))

    except Exception as e:
        print("錯誤:", e)
        print("收到內容:", body)
    
    return "OK"

if __name__ == "__main__":
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)