import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timedelta
import pytz
<<<<<<< HEAD
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
import matplotlib
import os
import uuid
from google.cloud import storage
from firebase_admin import storage as firebase_storage


# 確保 Firebase 不會重複初始化
if not firebase_admin._apps:
    cred = credentials.Certificate("./blood-sugar.json")
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'blood-sugar-57c0e.firebasestorage.app'
    })

# 連接 Firestore
db = firestore.client()

=======

# 設置 matplotlib 後端（必須在導入 pyplot 之前）
import matplotlib
matplotlib.use('Agg')  # 使用非交互式後端，避免 tkinter 衝突

import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
import os
import uuid
from google.cloud import storage

# 設置 matplotlib 中文字體
def setup_chinese_font():
    """設置 matplotlib 中文字體顯示"""
    try:
        # 嘗試載入 Noto Sans CJK TC 字體
        font_paths = [
            "./NotoSansCJKtc-Regular.otf",
            os.path.join(os.path.dirname(__file__), "NotoSansCJKtc-Regular.otf"),
            os.path.join(os.path.dirname(__file__), "fonts", "NotoSansCJKtc-Regular.otf"),
        ]
        
        font_loaded = False
        for font_path in font_paths:
            if os.path.exists(font_path):
                fontManager.addfont(font_path)
                matplotlib.rcParams['font.family'] = 'Noto Sans CJK TC'
                font_loaded = True
                print(f"✅ 載入字體：{font_path}")
                break
        
        if not font_loaded:
            # 使用系統字體
            if os.name == 'nt':  # Windows
                matplotlib.rcParams['font.family'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
            else:  # Linux/Mac  
                matplotlib.rcParams['font.family'] = ['Noto Sans CJK TC', 'AR PL UMing CN', 'WenQuanYi Micro Hei', 'DejaVu Sans']
            print("✅ 使用系統預設中文字體")
            
        # 解決負號顯示問題
        matplotlib.rcParams['axes.unicode_minus'] = False
        
    except Exception as e:
        print(f"⚠️ 字體設置失敗：{str(e)}")
        # 最後備用方案
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft JhengHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False

# 在模組載入時設置字體
setup_chinese_font()

# LINE Bot 相關導入
try:
    from linebot.models import (
        TextSendMessage,
        QuickReply,
        QuickReplyButton,
        DatetimePickerAction,
        PostbackAction,
        ImageSendMessage,
    )
    LINEBOT_AVAILABLE = True
except ImportError:
    print("❌ LINE Bot SDK 未安裝，血糖訊息功能將受限")
    LINEBOT_AVAILABLE = False
    # 定義空的類別以避免錯誤
    class TextSendMessage:
        def __init__(self, text): pass
    class QuickReply:
        def __init__(self, items): pass
    class QuickReplyButton:
        def __init__(self, action): pass
    class DatetimePickerAction:
        def __init__(self, **kwargs): pass
    class PostbackAction:
        def __init__(self, **kwargs): pass
    class ImageSendMessage:
        def __init__(self, **kwargs): pass

# 設置 Firebase 可用狀態
FIREBASE_AVAILABLE = False

# Firebase 初始化
try:
    # 確保 Firebase 不會重複初始化
    if not firebase_admin._apps:
        # 嘗試不同的憑證路徑
        credential_paths = [
            "/etc/secrets/blood-sugar.json",  # Linux/Docker 環境
            "blood-sugar.json",  # 當前目錄
            os.path.join(os.path.dirname(__file__), "blood-sugar.json"),  # 相對於當前文件
            os.path.join(os.path.dirname(__file__), "..", "blood-sugar.json"),  # 上一層目錄
            os.path.expanduser("~/blood-sugar.json"),  # 用戶家目錄
        ]
        
        cred = None
        used_path = None
        for path in credential_paths:
            if os.path.exists(path):
                cred = credentials.Certificate(path)
                used_path = path
                break
        
        if cred is None:
            print("❌ Firebase 憑證文件未找到，血糖功能將不可用")
            print(f"   請將 blood-sugar.json 放在以下位置之一：")
            for path in credential_paths:
                print(f"   - {path}")
            db = None
        else:
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            FIREBASE_AVAILABLE = True
            print(f"✅ Firebase 初始化成功，使用憑證：{used_path}")
    else:
        db = firestore.client()
        FIREBASE_AVAILABLE = True
        print("✅ Firebase 已初始化")

except Exception as e:
    print(f"❌ Firebase 初始化失敗: {str(e)}")
    db = None
    FIREBASE_AVAILABLE = False
>>>>>>> 28b3d67 (message)

#Create
def record_blood_sugar(user_id, blood_sugar):
    """
    記錄使用者的血糖數據到 Firebase Firestore（使用子集合）。
    
    :param user_id: LINE 使用者 ID
    :param blood_sugar: 使用者輸入的血糖數值
    :return: 記錄成功訊息
    """
<<<<<<< HEAD
=======
    if not FIREBASE_AVAILABLE or db is None:
        return "❌ 血糖記錄功能暫時不可用"
        
>>>>>>> 28b3d67 (message)
    try:
        
        tz = pytz.timezone("Asia/Taipei")
        now = datetime.now(tz)
        date_str = now.strftime("%Y-%m-%d")  # 取得日期（YYYY-MM-DD）
        time_str = now.strftime("%H:%M")  # 取得時間（HH:MM）

        document_id = f"{date_str}_{user_id}"

        # 取得該使用者當日的紀錄
        doc_ref = db.collection("BloodSugarRecord").document(document_id)
        doc = doc_ref.get()

        if doc.exists:
            # 如果當天的文件已存在，則追加新的血糖數值
            doc_data = doc.to_dict()
            blood_sugar_records = doc_data.get("bloodsugar_records", [])
        else:
            # 如果當天文件不存在，則初始化
            blood_sugar_records = []

        # 加入新的血糖紀錄
        blood_sugar_records.append({
            "time": time_str,
            "value": blood_sugar
        })

        # 更新 Firestore
        doc_ref.set({
            "user_id": user_id,
            "date": date_str,
            "bloodsugar_records": blood_sugar_records,
            "last_updated": now.strftime("%Y-%m-%d %H:%M:%S")  # 紀錄最後更新時間
        }, merge=True)

        return f"✅ 已記錄血糖：{blood_sugar} mg/dL（{time_str}）"

    except Exception as e:
        return f"❌ 記錄失敗，錯誤：{str(e)}"


#Read 
def get_blood_sugar_by_date(user_id, date_str):
<<<<<<< HEAD
=======
    if not FIREBASE_AVAILABLE or db is None:
        return "❌ 血糖記錄功能暫時不可用"
        
>>>>>>> 28b3d67 (message)
    try:
        document_id = f"{date_str}_{user_id}"
        doc_ref = db.collection("BloodSugarRecord").document(document_id)
        doc = doc_ref.get()
        print(f"✅ Queried document {document_id}, exists: {doc.exists}")
        if doc.exists:
            records = doc.to_dict().get("bloodsugar_records", [])
            print(f"✅ Retrieved records: {records}")
            return records
        return []
    except Exception as e:
        print(f"❌ Error in get_blood_sugar_by_date: {str(e)}")
        return f"❌ 查詢失敗，錯誤：{str(e)}"

def get_latest_blood_sugar(user_id):
    """
    查詢某使用者的最新血糖記錄。
    
    :param user_id: LINE 使用者 ID
    :return: 最新的血糖記錄（dict），若無資料則返回 None
    """
<<<<<<< HEAD
    try:
        docs = (db.collection("BloodSugarRecord")
                .where("user_id", "==", user_id)
                .order_by("last_updated", direction=firestore.Query.DESCENDING)
                .limit(1)
                .stream())
        for doc in docs:
            records = doc.to_dict().get("bloodsugar_records", [])
            if records:
                return records[-1]
=======
    if not FIREBASE_AVAILABLE or db is None:
        return "❌ 血糖記錄功能暫時不可用"
        
    try:
        # 避免複合索引問題，先查詢所有該用戶的記錄，然後在客戶端排序
        docs = (db.collection("BloodSugarRecord")
                .where("user_id", "==", user_id)
                .stream())
        
        # 收集所有記錄並按日期排序
        all_records = []
        for doc in docs:
            doc_data = doc.to_dict()
            all_records.append(doc_data)
        
        if not all_records:
            return None
        
        # 按 last_updated 降序排序，找到最新的記錄
        all_records.sort(key=lambda x: x.get("last_updated", ""), reverse=True)
        
        # 獲取最新記錄中的最後一筆血糖值
        latest_record = all_records[0]
        records = latest_record.get("bloodsugar_records", [])
        
        if records:
            return records[-1]  # 返回當天的最後一筆記錄
        
>>>>>>> 28b3d67 (message)
        return None
    except Exception as e:
        return f"❌ 查詢失敗，錯誤：{str(e)}"

def get_blood_sugar_history(user_id, limit=7):
    """
    查詢某使用者的歷史血糖記錄（預設最近 7 天）。
    
    :param user_id: LINE 使用者 ID
    :param limit: 返回的天數上限，預設為 7
    :return: 歷史記錄的字典，key 為日期，value 為該日記錄列表
    """
<<<<<<< HEAD
    try:
        docs = (db.collection("BloodSugarRecord")
                .where("user_id", "==", user_id)
                .order_by("date", direction=firestore.Query.DESCENDING)
                .limit(limit)
                .stream())
        history = {}
        for doc in docs:
            doc_data = doc.to_dict()
            history[doc_data["date"]] = doc_data.get("bloodsugar_records", [])
=======
    if not FIREBASE_AVAILABLE or db is None:
        return "❌ 血糖記錄功能暫時不可用"
        
    try:
        # 避免複合索引問題，先查詢所有該用戶的記錄，然後在客戶端排序
        docs = (db.collection("BloodSugarRecord")
                .where("user_id", "==", user_id)
                .stream())
        
        # 收集所有記錄並按日期排序
        all_records = []
        for doc in docs:
            doc_data = doc.to_dict()
            all_records.append(doc_data)
        
        # 按日期降序排序（最新的在前）
        all_records.sort(key=lambda x: x.get("date", ""), reverse=True)
        
        # 取前 limit 筆記錄
        limited_records = all_records[:limit]
        
        # 轉換為歷史記錄字典
        history = {}
        for record in limited_records:
            history[record["date"]] = record.get("bloodsugar_records", [])
        
>>>>>>> 28b3d67 (message)
        return history
    except Exception as e:
        return f"❌ 查詢失敗，錯誤：{str(e)}"
    


def update_blood_sugar(user_id, date_str, record_index, new_value):
    """
    更新某使用者某日期的某筆血糖紀錄。

    :param user_id: LINE 使用者 ID
    :param date_str: 日期（格式：YYYY-MM-DD）
    :param record_index: 要更新的紀錄索引（第幾筆）
    :param new_value: 新的血糖值
    :return: 更新結果訊息
    """
<<<<<<< HEAD
=======
    if not FIREBASE_AVAILABLE or db is None:
        return "❌ 血糖記錄功能暫時不可用"
        
>>>>>>> 28b3d67 (message)
    try:
        document_id = f"{date_str}_{user_id}"
        doc_ref = db.collection("BloodSugarRecord").document(document_id)
        doc = doc_ref.get()

        if not doc.exists:
            return "❌ 找不到該日期的紀錄！"

        doc_data = doc.to_dict()
        blood_sugar_records = doc_data.get("bloodsugar_records", [])

        if record_index < 0 or record_index >= len(blood_sugar_records):
            return "❌ 無效的紀錄索引！"

        # 更新指定索引的血糖值
        old_value = blood_sugar_records[record_index]["value"]
        blood_sugar_records[record_index]["value"] = new_value

        # 更新 Firestore
        tz = pytz.timezone("Asia/Taipei")
        now = datetime.now(tz)
        doc_ref.set({
            "user_id": user_id,
            "date": date_str,
            "bloodsugar_records": blood_sugar_records,
            "last_updated": now.strftime("%Y-%m-%d %H:%M:%S")
        }, merge=True)

        return f"✅ 已將 {blood_sugar_records[record_index]['time']} 的血糖值從 {old_value} 修改為 {new_value} mg/dL"

    except Exception as e:
        return f"❌ 修改失敗，錯誤：{str(e)}"
    


def delete_blood_sugar(user_id, date_str, record_index):
    """
    刪除某使用者某日期的某筆血糖紀錄。
    
    :param user_id: LINE 使用者 ID
    :param date_str: 日期（格式：YYYY-MM-DD）
    :param record_index: 要刪除的紀錄索引（第幾筆）
    :return: 刪除結果訊息
    """
<<<<<<< HEAD
=======
    if not FIREBASE_AVAILABLE or db is None:
        return "❌ 血糖記錄功能暫時不可用"
        
>>>>>>> 28b3d67 (message)
    try:
        document_id = f"{date_str}_{user_id}"
        doc_ref = db.collection("BloodSugarRecord").document(document_id)
        doc = doc_ref.get()

        if not doc.exists:
            return "❌ 找不到該日期的紀錄！"

        doc_data = doc.to_dict()
        blood_sugar_records = doc_data.get("bloodsugar_records", [])

        if record_index < 0 or record_index >= len(blood_sugar_records):
            return "❌ 無效的紀錄索引！"

        # 刪除指定索引的血糖紀錄
        deleted_record = blood_sugar_records.pop(record_index)

        # 如果沒有剩餘紀錄，刪除整個文件；否則更新文件
        if not blood_sugar_records:
            doc_ref.delete()
            return f"✅ 已刪除 {deleted_record['time']} 的血糖紀錄 ({deleted_record['value']} mg/dL)，該日期無其他紀錄。"
        else:
            tz = pytz.timezone("Asia/Taipei")
            now = datetime.now(tz)
            doc_ref.set({
                "user_id": user_id,
                "date": date_str,
                "bloodsugar_records": blood_sugar_records,
                "last_updated": now.strftime("%Y-%m-%d %H:%M:%S")
            }, merge=True)
            return f"✅ 已刪除 {deleted_record['time']} 的血糖紀錄 ({deleted_record['value']} mg/dL)"
    except Exception as e:
        return f"❌ 刪除失敗，錯誤：{str(e)}"
    
def generate_blood_sugar_chart(user_id, records, period="today"):
<<<<<<< HEAD
=======
    if not FIREBASE_AVAILABLE or db is None:
        return "❌ 血糖記錄功能暫時不可用"
        
>>>>>>> 28b3d67 (message)
    if not records or isinstance(records, str):
        return "❌ 沒有紀錄，無法生成圖表"

    try:
<<<<<<< HEAD
        times = [record['time'] for record in records]
        values = [record['value'] for record in records]
        max_val = max(values)
        min_val = min(values)
        avg = sum(values) / len(values)

        # 專業醫療級配色和判斷
        if max_val > 140:
            status_text = "血糖偏高"
            status_color = '#D32F2F'  # 深紅色，專業警示色
            status_bg = '#FFEBEE'
            status_icon = '⚠'
        elif min_val < 70:
            status_text = "血糖偏低" 
            status_color = '#F57C00'  # 橘色警示
            status_bg = '#FFF3E0'
            status_icon = '⚠'
        else:
            status_text = "血糖正常"
            status_color = '#2E7D32'  # 深綠色
            status_bg = '#E8F5E9'
            status_icon = '✓'

        filename = f"bloodsugar_{user_id}_{uuid.uuid4().hex[:8]}.png"
        file_path = os.path.join("/tmp", filename)
        fontManager.addfont("./NotoSansCJKtc-Regular.otf")
        plt.rcParams['font.family'] = 'Noto Sans CJK TC'

        # 使用橫式長方形圖表尺寸（縮小以避免裁切），提升可讀性
        fig, ax = plt.subplots(figsize=(16, 9))

        # 控制整體版面，預留上方標題與狀態條、底部統計徽章區，並增大左邊距避免 Y 軸標題被裁切
        plt.subplots_adjust(left=0.16, right=0.96, top=0.83, bottom=0.28)

        # 純白背景，更專業乾淨
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')

        # 動態Y軸範圍，確保有足夠空間顯示
        y_min = max(0, min_val - 30)
        y_max = max_val + 50
        ax.set_ylim(y_min, y_max)

        # --- 主標題，更大字體（放在最上方，不與狀態條重疊） --- #
        plt.suptitle("血糖監測報告", fontsize=44, fontweight='bold', y=0.982, color='#1A1A1A')

        # --- 狀態指示條（再往下移動，避免覆蓋標題） --- #
        plt.figtext(
            0.5, 0.84,
            f"{status_icon} {status_text}",
            fontsize=30, fontweight='bold', color=status_color,
            ha='center', va='center',
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=status_bg,
                edgecolor=status_color,
                linewidth=3,
                alpha=0.95
=======
        # 保持簡潔的圖表生成邏輯（基於您好看的版本）
        times = [record['time'] for record in records]
        values = [record['value'] for record in records]
        max_idx = values.index(max(values))
        min_idx = values.index(min(values))
        avg = sum(values) / len(values)

        # summary 條設計（根據期間動態調整）
        if period == "week":
            title_text = "最近一週血糖趨勢"
            prefix = "一週"
        else:
            title_text = "今日血糖趨勢"  
            prefix = "今日"

        if max(values) > 140:
            summary_txt = f"{prefix}血糖偏高"
            summary_color = '#e53935'
            summary_bg = '#ffebee'
        elif min(values) < 70:
            summary_txt = f"{prefix}血糖偏低"
            summary_color = '#fb8c00'
            summary_bg = '#fff8e1'
        else:
            summary_txt = f"{prefix}血糖正常"
            summary_color = '#388e3c'
            summary_bg = '#e8f5e9'

        filename = f"bloodsugar_{user_id}_{uuid.uuid4().hex[:8]}.png"
        
        # 在 Windows 環境下使用 temp 目錄
        if os.name == 'nt':  # Windows
            temp_dir = os.environ.get('TEMP', 'C:\\temp')
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            file_path = os.path.join(temp_dir, filename)
        else:  # Linux/Mac
            file_path = os.path.join("/tmp", filename)
            
        # 中文字體已在模組載入時設置，這裡只需確保設定生效
        plt.rcParams['axes.unicode_minus'] = False

        fig, ax = plt.subplots(figsize=(14, 9))
        ax.set_ylim(min(values) - 40, max(values) + 60)
        ax.set_facecolor('#f9fafc')
        plt.gcf().patch.set_facecolor('#f9fafc')

        # --- 主標題 --- #
        plt.suptitle(title_text, fontsize=44, fontweight='bold', y=0.98)

        # --- 結論 summary 條 --- #
        plt.figtext(
            0.5, 0.85,
            summary_txt,
            fontsize=32, fontweight='heavy', color=summary_color,
            ha='center', va='center',
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor=summary_bg,
                edgecolor=summary_color,
                linewidth=2,
                alpha=0.97
>>>>>>> 28b3d67 (message)
            ),
            zorder=30
        )

<<<<<<< HEAD
        # --- 主要數據線條，使用醫療級藍色 --- #
        medical_blue = '#1565C0'
        ax.plot(times, values, color=medical_blue, linewidth=6, zorder=3, 
                solid_capstyle='round', alpha=0.9)
        
        # 數據點設計 - 更大更清晰
        for i, (x, y) in enumerate(zip(times, values)):
            # 外圈白色邊框
            ax.scatter(x, y, s=1800, color='white', zorder=4, 
                      edgecolors=medical_blue, linewidths=4)
            # 內部數據點
            ax.scatter(x, y, s=1200, color=medical_blue, zorder=5)
            # 數值標籤 - 超大字體
            ax.text(x, y + 20, f"{y}", fontsize=44, fontweight='bold', 
                   ha='center', va='bottom', color=medical_blue, zorder=7)

        # --- 參考線 --- #
        # 低血糖線 (70)
        ax.axhline(70, color='#F57C00', linestyle='--', linewidth=3, alpha=0.8, zorder=2)
        ax.text(times[0], 72, '低血糖線 (70)', fontsize=20, color='#F57C00', 
                fontweight='bold', va='bottom')
        
        # 高血糖線 (140) 
        ax.axhline(140, color='#D32F2F', linestyle='--', linewidth=3, alpha=0.8, zorder=2)
        ax.text(times[0], 142, '高血糖線 (140)', fontsize=20, color='#D32F2F', 
                fontweight='bold', va='bottom')
        
        # 平均線
        ax.axhline(avg, color='#2E7D32', linestyle='-', linewidth=2, alpha=0.6, zorder=2)

        # --- 底部右側三色統計徽章（不遮擋圖表，長輩友善） --- #
        badge_font = 28
        # 最高（紅）
        plt.figtext(
            0.82, 0.10,
            f"最高\n{max_val}",
            fontsize=badge_font, color='#D32F2F', fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#FFEBEE', edgecolor='#D32F2F', linewidth=3, alpha=0.98)
        )
        # 平均（綠）
        plt.figtext(
            0.90, 0.10,
            f"平均\n{avg:.1f}",
            fontsize=badge_font, color='#2E7D32', fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=3, alpha=0.98)
        )
        # 最低（橘）
        plt.figtext(
            0.985, 0.10,
            f"最低\n{min_val}",
            fontsize=badge_font, color='#F57C00', fontweight='bold', ha='right', va='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#FFF3E0', edgecolor='#F57C00', linewidth=3, alpha=0.98)
        )

        # --- 軸標籤，超大字體 --- #
        plt.xlabel("時間", fontsize=36, labelpad=25, fontweight='bold', color='#424242')
        plt.ylabel("血糖值 (mg/dL)", fontsize=36, labelpad=25, fontweight='bold', color='#424242')
        
        # --- 網格線，更清晰 --- #
        plt.grid(True, linestyle="-", alpha=0.2, color='#BDBDBD', linewidth=1)
        
        # --- 刻度標籤，超大字體 --- #
        plt.xticks(fontsize=30, color='#424242', fontweight='500')
        plt.yticks(fontsize=30, color='#424242', fontweight='500')
        
        # 保持刻度與標籤清晰，同時確保不裁切（tight）
        plt.savefig(file_path, dpi=200, bbox_inches='tight', pad_inches=0.5, facecolor='white', edgecolor='none')
=======
        # --- 主圖內容（保持您的美觀樣式）--- #
        ax.plot(times, values, color="#90caf9", linewidth=8, zorder=2, solid_capstyle='round', alpha=0.85)
        ax.plot(times, values, color="#90caf9", linewidth=16, zorder=1, alpha=0.08)
        for i, (x, y) in enumerate(zip(times, values)):
            ax.scatter(x, y, s=1200, color='white', zorder=4, alpha=0.60, edgecolors='none')
            ax.scatter(x, y, s=800, color="#1e88e5", zorder=5, edgecolors="white", linewidths=7)
            ax.text(x, y + 15, f"{y}", fontsize=38, fontweight='bold', ha='center', va='bottom', color="#1e88e5", zorder=7)
        
        # 參考線
        ax.axhline(70, color="#fb8c00", linestyle="--", linewidth=2, alpha=0.8)
        ax.axhline(140, color="#e53935", linestyle="--", linewidth=2, alpha=0.8)
        ax.axhline(avg, color="#388e3c", linestyle="--", linewidth=3, alpha=0.7, zorder=3)
        
        # 底部統計資訊
        plt.figtext(0.19, 0.03, f"⬆ 最高 {values[max_idx]}", fontsize=26, color='#e53935', fontweight='bold', ha='left', va='center')
        plt.figtext(0.5, 0.03, f"平均 {avg:.1f}", fontsize=26, color='#388e3c', fontweight='bold', ha='center', va='center')
        plt.figtext(0.81, 0.03, f"⬇ 最低 {values[min_idx]}", fontsize=26, color='#fb8c00', fontweight='bold', ha='right', va='center')
        
        # 軸標籤和格式
        plt.xlabel("時間", fontsize=28, labelpad=20)
        plt.ylabel("血糖值 (mg/dL)", fontsize=28, labelpad=20)
        plt.grid(True, linestyle="--", alpha=0.13)
        
        # 簡化的標籤處理（保持美觀）
        if period == "week" and len(times) > 8:
            # 週圖表時，適當簡化標籤
            step = max(1, len(times) // 6)
            selected_indices = list(range(0, len(times), step))
            ax.set_xticks(selected_indices)
            ax.set_xticklabels([times[i] for i in selected_indices], rotation=45, ha='right', fontsize=18)
        else:
            # 保持原有的美觀樣式
            plt.xticks(fontsize=24)
            
        plt.yticks(fontsize=24)
        plt.tight_layout(rect=[0.06, 0.13, 0.96, 0.84])  # 預留 summary 條空間

        plt.savefig(file_path, dpi=150, bbox_inches='tight')
>>>>>>> 28b3d67 (message)
        plt.close()

        print(f"✅ 圖表已產生：{file_path}")

<<<<<<< HEAD
        # 上傳 Storage 並取得網址
        url = upload_and_get_url(file_path, user_id, period)
        print(f"✅ Storage 簽名網址：{url}")

        # 只有在上傳成功時才刪除本地檔案
        if url and not url.startswith("❌"):
            try:
                os.remove(file_path)
                print(f"✅ 已刪除暫存檔：{file_path}")
            except Exception as e:
                print(f"⚠️ 無法刪除暫存檔：{str(e)}")
        else:
            print(f"⚠️ 上傳失敗，保留暫存檔：{file_path}")

=======
        # 步驟2：上傳 Storage 並取得短網址
        url = upload_and_get_url(file_path, user_id, period)
        print(f"✅ Storage 簽名網址：{url}")

        # 步驟3：刪除本地檔案
        try:
            os.remove(file_path)
            print(f"✅ 已刪除暫存檔：{file_path}")
        except Exception as e:
            print(f"⚠️ 無法刪除暫存檔：{str(e)}")

        # 步驟4：直接回傳網址
>>>>>>> 28b3d67 (message)
        return url

    except Exception as e:
        print(f"❌ 產生圖表失敗：{str(e)}")
        return f"❌ 圖表生成錯誤：{str(e)}"

def upload_and_get_url(local_file, user_id, period="today"):
    """
    上傳檔案到 Firebase Storage，並取得 10 分鐘有效的簽名網址
    """
<<<<<<< HEAD
    try:
        # 使用 Firebase Admin SDK 來上傳到 Firebase Storage
        bucket = firebase_storage.bucket()
        
=======
    if not FIREBASE_AVAILABLE:
        print("❌ Firebase 不可用，無法上傳圖片")
        return "❌ Storage 不可用"
        
    try:
        # 尋找憑證文件
        credential_paths = [
            "/etc/secrets/blood-sugar.json",  # Linux/Docker 環境
            "blood-sugar.json",  # 當前目錄
            os.path.join(os.path.dirname(__file__), "blood-sugar.json"),  # 相對於當前文件
            os.path.join(os.path.dirname(__file__), "..", "blood-sugar.json"),  # 上一層目錄
            os.path.expanduser("~/blood-sugar.json"),  # 用戶家目錄
        ]
        
        credential_path = None
        for path in credential_paths:
            if os.path.exists(path):
                credential_path = path
                break
        
        if credential_path is None:
            print("❌ Firebase 憑證文件未找到，無法上傳圖片")
            return "❌ Storage 憑證未找到"
        
        # 這裡用你專案的 bucket 名稱
        bucket_name = "blood-sugar-57c0e.firebasestorage.app"  # 換成你的 bucket name
        storage_client = storage.Client.from_service_account_json(credential_path)
        bucket = storage_client.bucket(bucket_name)
>>>>>>> 28b3d67 (message)
        # 上傳目標路徑
        filename = os.path.basename(local_file)
        blob_path = f"charts/{user_id}/{period}/{filename}"
        blob = bucket.blob(blob_path)

<<<<<<< HEAD
        # 上傳檔案
        with open(local_file, 'rb') as f:
            blob.upload_from_file(f, content_type='image/png')

        # 產生公開下載網址
        blob.make_public()
        url = blob.public_url

        print(f"✅ 圖片已上傳到 Firebase Storage: {url}")
=======
        # 上傳
        blob.upload_from_filename(local_file)

        # 產生 10 分鐘有效的簽名網址
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=10),
            method="GET"
        )

        print(f"✅ 圖片已上傳到 Storage: {url}")
>>>>>>> 28b3d67 (message)
        return url

    except Exception as e:
        print(f"❌ 上傳 Storage 失敗：{str(e)}")
<<<<<<< HEAD
        return f"❌ Storage 上傳錯誤：{str(e)}"
=======
        return f"❌ Storage 上傳錯誤：{str(e)}"


# ==================== 血糖記錄介面相關函數 ====================

def create_blood_sugar_message(user_id, date_str):
    """創建血糖記錄訊息"""
    if not FIREBASE_AVAILABLE:
        return TextSendMessage(text="❌ 血糖記錄功能暫時不可用")

    try:
        # 查詢指定日期的血糖紀錄
        print(f"✅ Querying blood sugar for user {user_id} on date {date_str}")
        records = get_blood_sugar_by_date(user_id, date_str)
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
        quick_reply = QuickReply(
            items=[
                QuickReplyButton(
                    action=DatetimePickerAction(
                        label="選擇日期",
                        data="action=select_date",
                        mode="date",
                        initial=date_str,
                        max=datetime.now(pytz.timezone("Asia/Taipei")).strftime("%Y-%m-%d"),
                        min="2020-01-01",
                    )
                ),
                QuickReplyButton(action=PostbackAction(label="新增", data="action=add_blood_sugar")),
                QuickReplyButton(action=PostbackAction(label="修改", data="action=edit_blood_sugar")),
                QuickReplyButton(action=PostbackAction(label="刪除", data="action=delete_blood_sugar")),
            ]
        )

        return TextSendMessage(text=message_text, quick_reply=quick_reply)
    except Exception as e:
        print(f"❌ Error in create_blood_sugar_message: {str(e)}")
        return TextSendMessage(text=f"❌ 無法顯示血糖紀錄，錯誤：{str(e)}")


def create_report_menu_message():
    """創建報表選單訊息"""
    if not FIREBASE_AVAILABLE:
        return TextSendMessage(text="❌ 血糖報表功能暫時不可用")

    try:
        quick_reply = QuickReply(
            items=[
                QuickReplyButton(action=PostbackAction(label="今天", data="action=report_today")),
                QuickReplyButton(action=PostbackAction(label="最近一週", data="action=report_last_week")),
                QuickReplyButton(
                    action=DatetimePickerAction(
                        label="選擇日期",
                        data="action=report_select_date",
                        mode="date",
                        initial=datetime.now(pytz.timezone("Asia/Taipei")).strftime("%Y-%m-%d"),
                        max=datetime.now(pytz.timezone("Asia/Taipei")).strftime("%Y-%m-%d"),
                        min="2020-01-01",
                    )
                ),
            ]
        )
        return TextSendMessage(text="📊 請選擇要查看的血糖報表類型：", quick_reply=quick_reply)
    except Exception as e:
        print(f"❌ Error in create_report_menu_message: {str(e)}")
        return TextSendMessage(text=f"❌ 無法顯示報表選單，錯誤：{str(e)}")


def handle_blood_sugar_report(user_id, report_type, date_str=None):
    """處理血糖報表生成

    Args:
        user_id: 使用者ID
        report_type: 報表類型 ('today', 'date', 'week')
        date_str: 指定日期 (for report_type='date')

    Returns:
        LINE 訊息物件或訊息列表
    """
    if not FIREBASE_AVAILABLE:
        return TextSendMessage(text="❌ 血糖報表功能暫時不可用")

    try:
        tz = pytz.timezone("Asia/Taipei")

        if report_type == "today":
            # 今日報表
            today = datetime.now(tz).strftime("%Y-%m-%d")
            records = get_blood_sugar_by_date(user_id, today)

            if isinstance(records, str):  # 查詢錯誤
                return TextSendMessage(text=records)
            elif not records:  # 無紀錄
                return TextSendMessage(text="📊 今天還沒有記錄血糖喔！\n\n💡 快來記錄您的第一筆血糖數據吧！")
            else:  # 有紀錄，生成圖表
                image_url = generate_blood_sugar_chart(user_id, records, period="today")
                if image_url.startswith("❌"):
                    return TextSendMessage(text=image_url)
                else:
                    # 生成分析文字
                    analysis_text = generate_daily_analysis_text(records, today)
                    return [
                        TextSendMessage(text=analysis_text),
                        ImageSendMessage(original_content_url=image_url, preview_image_url=image_url),
                    ]

        elif report_type == "week":
            # 週報表 - 顯示最近7天的所有數據
            history = get_blood_sugar_history(user_id, limit=7)

            if isinstance(history, str):  # 錯誤訊息
                return TextSendMessage(text=history)
            elif not history:
                return TextSendMessage(text="📊 最近一週沒有血糖記錄\n\n💡 建議開始記錄血糖數據，以便追蹤健康狀況！")
            else:
                # 合併所有血糖記錄生成週圖表
                all_records = []
                for date_str, records in history.items():
                    if records:
                        # 為每個記錄添加日期信息，以便在圖表中顯示
                        for record in records:
                            all_records.append({
                                "time": f"{date_str} {record['time']}",
                                "value": record["value"],
                                "date": date_str,
                                "original_time": record["time"]
                            })
                
                if all_records:
                    # 按日期時間排序
                    all_records.sort(key=lambda x: f"{x['date']} {x['original_time']}")
                    
                    # 生成週圖表
                    image_url = generate_blood_sugar_chart(user_id, all_records, period="week")
                    if image_url.startswith("❌"):
                        return TextSendMessage(text=image_url)
                    else:
                        # 生成週分析文字
                        analysis_text = generate_weekly_analysis_text(history, 7)
                        return [
                            TextSendMessage(text=analysis_text),
                            ImageSendMessage(original_content_url=image_url, preview_image_url=image_url),
                        ]
                else:
                    # 如果沒有有效記錄，只返回文字分析
                    analysis_text = generate_weekly_analysis_text(history, 7)
                    return TextSendMessage(text=analysis_text)

        elif report_type == "date" and date_str:
            # 指定日期報表
            records = get_blood_sugar_by_date(user_id, date_str)

            if isinstance(records, str):  # 查詢錯誤
                return TextSendMessage(text=records)
            elif not records:  # 無紀錄
                return TextSendMessage(text=f"📊 {date_str} 沒有血糖記錄")
            else:  # 有紀錄，生成圖表
                image_url = generate_blood_sugar_chart(user_id, records, period=date_str)
                if image_url.startswith("❌"):
                    return TextSendMessage(text=image_url)
                else:
                    analysis_text = generate_daily_analysis_text(records, date_str)
                    return [
                        TextSendMessage(text=analysis_text),
                        ImageSendMessage(original_content_url=image_url, preview_image_url=image_url),
                    ]
        else:
            return TextSendMessage(text="❌ 不支援的報表類型")

    except Exception as e:
        print(f"❌ Error in handle_blood_sugar_report: {str(e)}")
        return TextSendMessage(text=f"❌ 報表生成失敗：{str(e)}")


def generate_daily_analysis_text(records, date_str):
    """生成每日血糖分析文字"""
    try:
        values = [record["value"] for record in records]
        times = [record["time"] for record in records]

        avg_value = sum(values) / len(values)
        max_value = max(values)
        min_value = min(values)
        max_time = times[values.index(max_value)]
        min_time = times[values.index(min_value)]

        # 健康狀態評估
        if avg_value > 140:
            health_status = "⚠️ 偏高"
            advice = "建議注意飲食控制，必要時諮詢醫師"
        elif avg_value < 80:
            health_status = "⚠️ 偏低"
            advice = "注意低血糖風險，建議適時補充糖分"
        else:
            health_status = "✅ 正常"
            advice = "血糖控制良好，請繼續保持！"

        # 血糖變化評估
        if max_value - min_value > 60:
            stability = "波動較大"
        elif max_value - min_value > 30:
            stability = "輕微波動"
        else:
            stability = "穩定"

        analysis_text = f"""📊 {date_str} 血糖分析報告

🎯 整體狀況：{health_status}
📈 平均血糖：{avg_value:.1f} mg/dL
📊 血糖範圍：{min_value} - {max_value} mg/dL
📉 波動狀況：{stability}

🔍 詳細數據：
• 最高血糖：{max_value} mg/dL ({max_time})
• 最低血糖：{min_value} mg/dL ({min_time})
• 測量次數：{len(records)} 次

💡 建議：{advice}"""

        return analysis_text

    except Exception as e:
        print(f"❌ Error generating daily analysis: {str(e)}")
        return f"📊 {date_str} 血糖報表\n\n記錄了 {len(records)} 筆血糖數據"


def generate_weekly_analysis_text(history, days):
    """生成週血糖分析文字"""
    try:
        total_records = 0
        all_values = []
        record_days = []
        
        for date_str, records in history.items():
            if records:
                record_days.append(date_str)
                total_records += len(records)
                all_values.extend([record["value"] for record in records])
        
        if not all_values:
            return f"📊 最近{days}天血糖趨勢分析\n\n暫無血糖記錄數據"
        
        avg_all = sum(all_values) / len(all_values)
        max_value = max(all_values)
        min_value = min(all_values)
        
        # 健康狀態評估
        if avg_all > 140:
            health_status = "⚠️ 偏高"
            suggestion = "整體血糖偏高，建議調整飲食並增加運動，必要時諮詢醫師"
        elif avg_all < 80:
            health_status = "⚠️ 偏低"
            suggestion = "整體血糖偏低，注意低血糖風險，建議適時補充糖分"
        else:
            health_status = "✅ 良好"
            suggestion = "血糖控制良好，請繼續保持現有的生活習慣"

        analysis_text = f"""📊 最近{days}天血糖趨勢分析

🎯 整體評估：{health_status}
📊 平均血糖：{avg_all:.1f} mg/dL
📉 血糖範圍：{min_value} - {max_value} mg/dL

📅 統計資料：
• 記錄天數：{len(record_days)} 天
• 測量次數：{total_records} 次

💡 健康建議：
{suggestion}"""

        return analysis_text

    except Exception as e:
        print(f"❌ Error generating weekly analysis: {str(e)}")
        return f"📊 最近{days}天血糖趨勢分析\n\n共記錄了血糖數據，請查看詳細記錄。"


def show_records_for_edit(user_id, date_str):
    """顯示可編輯的血糖記錄"""
    if not FIREBASE_AVAILABLE:
        return TextSendMessage(text="❌ 血糖記錄功能暫時不可用")

    try:
        print(f"✅ Showing records for edit for user {user_id} on date {date_str}")
        records = get_blood_sugar_by_date(user_id, date_str)

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
                        label=button_label[:20], data=f"action=edit_record&index={idx}"  # LINE 按鈕標籤最多 20 字元
                    )
                )
            )

        return TextSendMessage(text=message_text, quick_reply=QuickReply(items=quick_reply_items))
    except Exception as e:
        print(f"❌ Error in show_records_for_edit: {str(e)}")
        return TextSendMessage(text=f"❌ 無法顯示血糖紀錄，錯誤：{str(e)}")


def show_records_for_delete(user_id, date_str):
    """顯示可刪除的血糖記錄"""
    if not FIREBASE_AVAILABLE:
        return TextSendMessage(text="❌ 血糖記錄功能暫時不可用")

    try:
        print(f"✅ Showing records for delete for user {user_id} on date {date_str}")
        records = get_blood_sugar_by_date(user_id, date_str)

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
                        label=button_label[:20], data=f"action=delete_record&index={idx}"  # LINE 按鈕標籤最多 20 字元
                    )
                )
            )

        return TextSendMessage(text=message_text, quick_reply=QuickReply(items=quick_reply_items))
    except Exception as e:
        print(f"❌ Error in show_records_for_delete: {str(e)}")
        return TextSendMessage(text=f"❌ 無法顯示血糖紀錄，錯誤：{str(e)}")
>>>>>>> 28b3d67 (message)
