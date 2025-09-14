from linebot.models import FlexSendMessage
import json

def generate_carousel_flex(elements):
    """
    生成輪播式 Flex Message
    
    Args:
        elements: 多個 Flex Message 內容
    """
    # 創建 Carousel Container
    contents = {
        "type": "carousel",
        "contents": elements
    }
    
    return contents

def generate_flex_message(title, food_name, data_dict):
    """
    生成基本 Flex Message
    
    Args:
        title: 標題
        food_name: 食物名稱
        data_dict: 資料字典
    """
    # 創建六個部分
    sections = []
    
    # 優點部分
    if 'advantages' in data_dict and data_dict['advantages']:
        advantages_section = {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "text",
                    "text": "優點",
                    "weight": "bold",
                    "color": "#1DB446",
                    "size": "sm"
                }
            ]
        }
        
        for adv in data_dict['advantages']:
            advantages_section['contents'].append({
                "type": "text",
                "text": f"• {adv}",
                "wrap": True,
                "color": "#666666",
                "size": "sm",
                "margin": "md"
            })
        
        sections.append(advantages_section)
    
    # 可能風險部分
    if 'potential_risks' in data_dict and data_dict['potential_risks']:
        risks_section = {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "text",
                    "text": "可能風險",
                    "weight": "bold",
                    "color": "#FF6B6E",
                    "size": "sm"
                }
            ],
            "margin": "md"
        }
        
        for risk in data_dict['potential_risks']:
            risks_section['contents'].append({
                "type": "text",
                "text": f"• {risk}",
                "wrap": True,
                "color": "#666666",
                "size": "sm",
                "margin": "md"
            })
        
        sections.append(risks_section)
    
    # 建議部分
    if 'suggestions' in data_dict and data_dict['suggestions']:
        suggestions_section = {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "text",
                    "text": "建議",
                    "weight": "bold",
                    "color": "#17C950",
                    "size": "sm"
                }
            ],
            "margin": "md"
        }
        
        for suggestion in data_dict['suggestions']:
            suggestions_section['contents'].append({
                "type": "text",
                "text": f"• {suggestion}",
                "wrap": True,
                "color": "#666666",
                "size": "sm",
                "margin": "md"
            })
        
        sections.append(suggestions_section)
    
    # 創建 Flex Message 容器
    bubble = {
        "type": "bubble",
        "size": "mega",
        "header": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "text",
                    "text": title,
                    "weight": "bold",
                    "color": "#1DB446",
                    "size": "sm"
                },
                {
                    "type": "text",
                    "text": food_name,
                    "weight": "bold",
                    "size": "xxl",
                    "margin": "md"
                }
            ],
            "paddingAll": "20px"
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": sections,
            "paddingAll": "13px"
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "contents": [
                {
                    "type": "text",
                    "text": "數據來源：營養成分表＆臨床研究文獻",
                    "wrap": True,
                    "color": "#aaaaaa",
                    "size": "xs"
                }
            ],
            "paddingAll": "13px"
        }
    }
    
    return bubble

def generate_calorie_source_flex_message(food_names, nutrition_data):
    """
    生成熱量來源分析 Flex Message
    
    Args:
        food_names: 食物名稱列表
        nutrition_data: 營養數據，包含以下鍵：
            - total_calories: 總熱量 (float, 單位: 大卡)
            - carbs_calories: 碳水化合物熱量 (float, 單位: 大卡)
            - protein_calories: 蛋白質熱量 (float, 單位: 大卡)
            - fat_calories: 脂肪熱量 (float, 單位: 大卡)
            - sugar_calories: 糖分熱量 (float, 單位: 大卡)
            - is_estimated: 是否為估算值 (boolean)
            
    Returns:
        FlexSendMessage 物件
    """
    # 計算百分比
    total_calories = nutrition_data.get('total_calories', 0)
    carbs_calories = nutrition_data.get('carbs_calories', 0)
    protein_calories = nutrition_data.get('protein_calories', 0)
    fat_calories = nutrition_data.get('fat_calories', 0)
    sugar_calories = nutrition_data.get('sugar_calories', 0)
    is_estimated = nutrition_data.get('is_estimated', False)
    
    if total_calories > 0:
        carbs_percent = round(carbs_calories / total_calories * 100)
        protein_percent = round(protein_calories / total_calories * 100)
        fat_percent = round(fat_calories / total_calories * 100)
        sugar_percent = round(sugar_calories / total_calories * 100)
    else:
        carbs_percent = 0
        protein_percent = 0
        fat_percent = 0
        sugar_percent = 0
    
    # 創建食物清單字串
    if len(food_names) > 1:
        food_title = "、".join(food_names[:3])
        if len(food_names) > 3:
            food_title += f" 等{len(food_names)}種食物"
    else:
        food_title = food_names[0] if food_names else "食物"
    
    # 顏色設定，確保在手機上能清楚顯示
    color_settings = {
        "carbs": {"bg": "#4A90E2", "text": "#0066CC", "name": "碳水化合物"},
        "protein": {"bg": "#F5A623", "text": "#CC6600", "name": "蛋白質"},
        "fat": {"bg": "#7ED321", "text": "#336633", "name": "脂肪"},
        "sugar": {"bg": "#FF69B4", "text": "#FF1493", "name": "糖分"}
    }
    
    # 選擇前三大熱量來源
    nutrient_data = [
        {"name": color_settings["carbs"]["name"], "percent": carbs_percent, "calories": carbs_calories, 
         "color": color_settings["carbs"]["bg"], "text_color": color_settings["carbs"]["text"], "type": "carbs"},
         
        {"name": color_settings["protein"]["name"], "percent": protein_percent, "calories": protein_calories, 
         "color": color_settings["protein"]["bg"], "text_color": color_settings["protein"]["text"], "type": "protein"},
         
        {"name": color_settings["fat"]["name"], "percent": fat_percent, "calories": fat_calories, 
         "color": color_settings["fat"]["bg"], "text_color": color_settings["fat"]["text"], "type": "fat"},
         
        {"name": color_settings["sugar"]["name"], "percent": sugar_percent, "calories": sugar_calories, 
         "color": color_settings["sugar"]["bg"], "text_color": color_settings["sugar"]["text"], "type": "sugar"}
    ]
    
    # 按熱量百分比排序，選出前三名
    top_nutrients = sorted(nutrient_data, key=lambda x: x["percent"], reverse=True)[:3]
    
    # 視覺化總熱量計量表（最大值500卡）
    max_calories = 500
    calories_percent = min(round(total_calories / max_calories * 100), 100)
    
    # 設定熱量標題（如果是估算值，添加提示）
    calorie_title = f"總熱量: {total_calories} 大卡"
    if is_estimated:
        calorie_title += " (估算值)"
    
    # 創建象形圖表示總熱量（使用文字圖標代替圖片URL）
    flame_levels = min(5, max(1, round(calories_percent / 20)))
    
    # 使用文字表情符號表示火焰
    flame_emoji = "🔥"
    empty_flame = "⚪"
    flame_text = ""
    
    for i in range(5):
        if i < flame_levels:
            flame_text += flame_emoji
        else:
            flame_text += empty_flame
    
    # 添加熱量圖示到容器
    calorie_meter = {
        "type": "box",
        "layout": "vertical",
        "contents": [
            {
                "type": "text",
                "text": calorie_title,
                "weight": "bold",
                "size": "md",
                "align": "center",
                "margin": "md"
            },
            {
                "type": "text",
                "text": flame_text,
                "align": "center",
                "size": "xl",
                "margin": "md"
            },
            {
                "type": "box",
                "layout": "vertical",
                "contents": [
                    {
                        "type": "box",
                        "layout": "vertical",
                        "contents": [],
                        "width": f"{calories_percent}%",
                        "height": "12px",
                        "backgroundColor": "#FF6B6E"
                    }
                ],
                "backgroundColor": "#EEEEEE",
                "height": "12px",
                "margin": "md"
            },
            {
                "type": "box",
                "layout": "horizontal",
                "contents": [
                    {
                        "type": "text",
                        "text": "0",
                        "size": "xs",
                        "color": "#555555"
                    },
                    {
                        "type": "text",
                        "text": "250",
                        "size": "xs",
                        "color": "#555555",
                        "align": "center"
                    },
                    {
                        "type": "text",
                        "text": "500 大卡",
                        "size": "xs",
                        "color": "#555555",
                        "align": "end"
                    }
                ],
                "margin": "sm"
            }
        ],
        "margin": "lg"
    }
    
    # 創建熱量來源分析以條形圖表示
    bar_contents = []
    
    # 添加標題
    bar_contents.append({
        "type": "text",
        "text": "主要熱量來源分析",
        "weight": "bold",
        "size": "md",
        "margin": "md"
    })
    
    # 如果是估算值，添加提示信息
    if is_estimated:
        bar_contents.append({
            "type": "text",
            "text": "⚠️ 以下數據為AI估算值，僅供參考",
            "size": "xs",
            "color": "#FF6B6E",
            "margin": "sm"
        })
    
    # 為前三大熱量來源創建條形圖，確保最小寬度為5%以便在手機上顯示
    for nutrient in top_nutrients:
        # 確保條形圖至少有5%的寬度
        display_percent = max(5, nutrient["percent"])
        
        bar_contents.append({
            "type": "box",
            "layout": "horizontal",
            "contents": [
                {
                    "type": "text",
                    "text": nutrient["name"],
                    "size": "sm",
                    "color": nutrient["text_color"],
                    "weight": "bold",
                    "flex": 3
                },
                {
                    "type": "text",
                    "text": f"{nutrient['percent']}%",
                    "size": "sm",
                    "color": nutrient["text_color"],
                    "align": "end",
                    "weight": "bold",
                    "flex": 1
                }
            ],
            "margin": "md"
        })
        
        # 確保手機上能看到條形圖顏色
        bar_contents.append({
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "box",
                    "layout": "vertical",
                    "contents": [
                        {
                            "type": "text", 
                            "text": " ", 
                            "size": "xxs"
                        }
                    ],
                    "height": "15px",
                    "width": f"{display_percent}%",
                    "backgroundColor": nutrient["color"]
                }
            ],
            "backgroundColor": "#EEEEEE",
            "height": "15px",
            "margin": "sm",
            "cornerRadius": "sm"
        })
    
    bar_chart = {
        "type": "box",
        "layout": "vertical",
        "contents": bar_contents,
        "margin": "xl",
        "paddingAll": "sm",
        "backgroundColor": "#FFFFFF",
        "cornerRadius": "md"
    }
    
    # 獲取分析結果 (使用標準的碳水、蛋白質、脂肪分析)
    std_carbs = next((item for item in nutrient_data if item["type"] == "carbs"), {"percent": 0})["percent"]
    std_protein = next((item for item in nutrient_data if item["type"] == "protein"), {"percent": 0})["percent"]
    std_fat = next((item for item in nutrient_data if item["type"] == "fat"), {"percent": 0})["percent"]
    
    analysis = get_calorie_source_analysis(std_carbs, std_protein, std_fat)
    
    # 如果糖分佔比高，添加糖分的特別提醒
    if sugar_percent > 20:
        sugar_warning = "⚠️ 糖分含量較高，糖尿病患者應特別注意。建議控制食用量，或以低糖替代食物。"
        analysis = sugar_warning + "\n\n" + analysis
    
    # 如果是估算值，添加提示
    if is_estimated:
        analysis = "📝 本分析基於AI估算的營養數據，僅供參考，實際數值可能有所不同。\n\n" + analysis
    
    # 創建 Flex Message
    bubble = {
        "type": "bubble",
        "size": "mega",
        "header": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "text",
                    "text": "熱量來源分析",
                    "weight": "bold",
                    "color": "#1DB446",
                    "size": "sm"
                },
                {
                    "type": "text",
                    "text": food_title,
                    "weight": "bold",
                    "size": "xl",
                    "margin": "md",
                    "wrap": True
                }
            ],
            "paddingAll": "20px",
            "backgroundColor": "#F9F9F9"
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                calorie_meter,
                bar_chart,
                {
                    "type": "box",
                    "layout": "vertical",
                    "contents": [
                        {
                            "type": "text",
                            "text": "糖尿病飲食建議",
                            "weight": "bold",
                            "size": "md"
                        },
                        {
                            "type": "text",
                            "text": analysis,
                            "wrap": True,
                            "color": "#666666",
                            "size": "sm",
                            "margin": "md"
                        }
                    ],
                    "margin": "xl",
                    "paddingAll": "sm",
                    "backgroundColor": "#F9F9F9",
                    "cornerRadius": "md"
                }
            ],
            "paddingAll": "13px",
            "backgroundColor": "#FFFFFF"
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "contents": [
                {
                    "type": "button",
                    "style": "primary",
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": "查看詳細營養資訊",
                        "data": f"detailed_calorie_source:{','.join(food_names)}"
                    },
                    "color": "#1DB446"
                },
                {
                    "type": "text",
                    "text": "數據來源：FatSecret營養資料庫" + (" + AI估算" if is_estimated else ""),
                    "wrap": True,
                    "color": "#aaaaaa",
                    "size": "xs",
                    "margin": "md",
                    "align": "center"
                }
            ],
            "paddingAll": "13px",
            "backgroundColor": "#F9F9F9"
        },
        "styles": {
            "header": {
                "separator": True
            },
            "footer": {
                "separator": True
            }
        }
    }
    
    # 將 bubble 直接包裝為 FlexSendMessage 並返回
    return FlexSendMessage(alt_text=f"{food_title} 的熱量來源分析", contents=bubble)

def get_calorie_source_analysis(carbs_percent, protein_percent, fat_percent):
    """
    根據熱量來源分析給出針對糖尿病患者的簡潔建議
    
    Args:
        carbs_percent: 碳水化合物熱量百分比
        protein_percent: 蛋白質熱量百分比
        fat_percent: 脂肪熱量百分比
    """
    # 判斷主要營養素比例並給出簡潔建議
    if carbs_percent > 65:
        return "💡 碳水化合物占比較高，建議搭配蛋白質食用以穩定血糖。\n⚠️ 糖尿病患者應控制份量，避免血糖快速上升。"
    elif carbs_percent < 30:
        return "💡 碳水化合物含量較低，有助於血糖穩定。\n✅ 適合糖尿病患者，建議搭配適量蔬菜增加纖維。"
    elif protein_percent > 35:
        return "💡 蛋白質含量豐富，有助於延緩血糖上升。\n⚠️ 注意腎功能狀況，避免過量攝取。"
    elif fat_percent > 40:
        return "💡 脂肪含量較高，雖不直接影響血糖但熱量密度高。\n⚠️ 建議控制份量，選擇健康油脂如橄欖油、堅果。"
    else:
        return "💡 營養比例均衡，適合糖尿病患者食用。\n✅ 建議定時定量，搭配血糖監測更安心。"


