from linebot.models import FlexSendMessage

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
            "paddingAll": "10px"
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": sections,
            "paddingAll": "10px"
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
            "paddingAll": "10px"
        }
    }
    
    return bubble

def generate_calorie_source_flex_message(food_names, nutrition_data):
    """
    生成簡潔的熱量來源分析 Flex Message
    
    Args:
        food_names: 食物名稱列表
        nutrition_data: 營養數據字典
    """
    # 計算各營養素的克數和熱量
    total_calories = round(nutrition_data.get('total_calories', 0))
    carbs_calories = nutrition_data.get('carbs_calories', 0)
    protein_calories = nutrition_data.get('protein_calories', 0)
    fat_calories = nutrition_data.get('fat_calories', 0)
    sugar_calories = nutrition_data.get('sugar_calories', 0)
    is_estimated = nutrition_data.get('is_estimated', False)
    
    # 計算克數
    carbs_grams = round(carbs_calories / 4, 1)
    protein_grams = round(protein_calories / 4, 1)
    fat_grams = round(fat_calories / 9, 1)
    sugar_grams = round(sugar_calories / 4, 1)
    
    # 計算百分比（用於判斷是否過高）
    if total_calories > 0:
        carbs_percent = round(carbs_calories / total_calories * 100)
        protein_percent = round(protein_calories / total_calories * 100)
        fat_percent = round(fat_calories / total_calories * 100)
        sugar_percent = round(sugar_calories / total_calories * 100)
    else:
        carbs_percent = protein_percent = fat_percent = sugar_percent = 0
    
    # 創建食物標題
    if len(food_names) > 1:
        food_title = "、".join(food_names[:3])
        if len(food_names) > 3:
            food_title += f" 等{len(food_names)}種食物"
    else:
        food_title = food_names[0] if food_names else "食物"
    
    # 判斷各營養素是否過高並設定顏色
    carbs_color = "#E53935" if carbs_percent > 65 else "#555555"
    sugar_color = "#E53935" if sugar_percent > 20 else "#555555"
    protein_color = "#FF8C00" if protein_percent > 35 else "#555555"
    
    # 生成營養警告
    warnings = []
    if sugar_percent > 20:
        warnings.append("高糖分")
    if carbs_percent > 65:
        warnings.append("高碳水")
    if protein_percent > 35:
        warnings.append("高蛋白")
    if fat_percent > 40:
        warnings.append("高脂肪")
    
    warning_text = " ".join(warnings) if warnings else "營養均衡"
    warning_color = "#E53935" if warnings else "#1DB446"
    
    # 創建 Flex Message
    bubble = {
        "type": "bubble",
        "size": "kilo",
        "header": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "text",
                    "text": food_title,
                    "weight": "bold",
                    "size": "lg",
                    "wrap": True,
                    "color": "#1A1A1A"
                }
            ],
            "paddingAll": "15px",
            "backgroundColor": "#FFFFFF",
            "borderColor": "#E0E0E0",
            "borderWidth": "light",
            "cornerRadius": "lg"
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                # 熱量顯示
                {
                    "type": "text",
                    "text": f"{total_calories}",
                    "weight": "bold",
                    "size": "3xl",
                    "align": "center"
                },
                {
                    "type": "text",
                    "text": "大卡",
                    "size": "sm",
                    "color": "#999999",
                    "align": "center",
                    "margin": "sm"
                },
                # 營養成分
                {
                    "type": "box",
                    "layout": "vertical",
                    "margin": "xl",
                    "spacing": "lg",
                    "contents": [
                        {
                            "type": "box",
                            "layout": "horizontal",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": "碳水化合物",
                                    "size": "md",
                                    "color": "#999999",
                                    "flex": 3
                                },
                                {
                                    "type": "text",
                                    "text": f"{carbs_grams}g",
                                    "size": "md",
                                    "color": carbs_color,
                                    "align": "end",
                                    "weight": "bold",
                                    "flex": 2
                                }
                            ]
                        },
                        {
                            "type": "box",
                            "layout": "horizontal",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": "糖分",
                                    "size": "md",
                                    "color": "#999999",
                                    "flex": 3
                                },
                                {
                                    "type": "text",
                                    "text": f"{sugar_grams}g",
                                    "size": "md",
                                    "color": sugar_color,
                                    "align": "end",
                                    "weight": "bold",
                                    "flex": 2
                                }
                            ]
                        },
                        {
                            "type": "box",
                            "layout": "horizontal",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": "蛋白質",
                                    "size": "md",
                                    "color": "#999999",
                                    "flex": 3
                                },
                                {
                                    "type": "text",
                                    "text": f"{protein_grams}g",
                                    "size": "md",
                                    "color": protein_color,
                                    "align": "end",
                                    "weight": "bold",
                                    "flex": 2
                                }
                            ]
                        }
                    ]
                },
                # 分隔線
                {
                    "type": "separator",
                    "margin": "xl",
                    "color": "#E0E0E0"
                },
                # 飲食建議
                {
                    "type": "text",
                    "text": warning_text,
                    "weight": "bold",
                    "size": "lg",
                    "margin": "lg",
                    "align": "center",
                    "color": warning_color
                }
            ],
            "paddingAll": "15px",
            "backgroundColor": "#FFFFFF"
        },
        "styles": {
            "header": {
                "separator": True
            },
            "body": {
                "backgroundColor": "#FFFFFF"
            }
        }
    }
    
    # 如果是估算值，添加提示
    if is_estimated:
        bubble["body"]["contents"].append({
            "type": "text",
            "text": "(AI估算值)",
            "size": "xs",
            "color": "#999999",
            "align": "center",
            "margin": "md"
        })
    
    return FlexSendMessage(alt_text=f"{food_title} 的營養分析", contents=bubble)