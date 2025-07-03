import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import json
import re
from datetime import datetime
import base64
import io
from openai import OpenAI

# 页面配置
st.set_page_config(
    page_title="🃏 德州扑克战绩分析师 🃏",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 硬编码API密钥 - 请替换为你的实际密钥
API_KEY = "sk-4258049c3aca4cf1b0f6426601531c1c"  # 请在这里输入你的API密钥

# 初始化session state
if 'game_data' not in st.session_state:
    st.session_state.game_data = {
        'players': {},  # {player_name: {datetime: {'score': score, 'hands': hands, 'buyin': buyin}}}
        'dates': [],    # [datetime1, datetime2, ...]
        'game_info': {} # {datetime: {'hands': total_hands, 'total_buyin': total_buyin}}
    }

# 初始化已处理文件列表
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

# 添加实时更新标志
if 'table_update_trigger' not in st.session_state:
    st.session_state.table_update_trigger = 0

def image_to_base64(image):
    """将PIL图像转换为base64字符串"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def call_qianwen_api(image_base64, api_key):
    """调用千问多模态API"""
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        prompt = """请分析这张游戏排行榜截图，提取以下信息并以JSON格式返回：

{
    "game_time": "比赛时间（格式：6-20 20:26 或 类似格式，必须包含日期和具体时间）",
    "total_hands": 总手数（数字），
    "players": [
        {
            "name": "玩家昵称",
            "score": 数字分数（正数或负数），
            "hands": 该玩家的手数（数字），
            "buyin": 该玩家的带入金额（数字，如果没有显示则为0）
        }
    ]
}

要求：
1. 仔细识别每个玩家的昵称，去除排名数字
2. 准确提取每个玩家的战绩分数，注意正负号
3. 提取每个玩家的手数信息
4. 提取每个玩家的带入金额（如果图片中有显示）
5. 提取游戏的总手数
6. 比赛时间必须精确到分钟，用于区分不同场次
7. 如果分数、手数、带入有逗号分隔符，请去除
8. 只返回JSON格式的数据，不要其他说明文字"""

        completion = client.chat.completions.create(
            model="qwen-vl-plus",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_base64}
                        },
                        {
                            "type": "text", 
                            "text": prompt
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        content = completion.choices[0].message.content
        if content is None:
            return None

        if not isinstance(content, str):
            try:
                content = str(content)
            except:
                return None
        
        try:
            json_start = content.find('{')
            json_end_pos = content.rfind('}')
            
            if json_start == -1 or json_end_pos == -1:
                return None
            
            if json_start >= json_end_pos:
                return None
            
            json_str = content[json_start:json_end_pos + 1]
            
            if not json_str.strip():
                return None
            
            parsed_data = json.loads(json_str)
            return parsed_data
        except (json.JSONDecodeError, ValueError):
            return None
            
    except Exception as e:
        st.error(f"API调用错误: {e}")
        return None

def validate_player_data(name, score, hands=0, buyin=0):
    """验证玩家数据是否完整有效"""
    # 验证昵称
    if not name or len(name.strip()) < 1:
        return False, "昵称为空"
    
    if len(name.strip()) > 50:
        return False, "昵称过长"
    
    cleaned_name = name.strip()
    
    # 只拒绝明显的乱码
    valid_chars = re.findall(r'[\w\u4e00-\u9fff]', cleaned_name)
    if len(valid_chars) < len(cleaned_name) * 0.1:
        return False, "昵称包含过多特殊符号"
    
    # 验证分数
    try:
        score_num = int(float(score))
    except (ValueError, TypeError):
        return False, "分数格式错误"
    
    # 验证手数
    try:
        hands_num = int(float(hands)) if hands else 0
        if hands_num < 0:
            return False, "手数不能为负数"
    except (ValueError, TypeError):
        return False, "手数格式错误"
    
    # 验证带入
    try:
        buyin_num = int(float(buyin)) if buyin else 0
        if buyin_num < 0:
            return False, "带入不能为负数"
    except (ValueError, TypeError):
        return False, "带入格式错误"
    
    return True, ""

def extract_game_info_with_qianwen(image, api_key):
    """使用千问API提取游戏信息"""
    try:
        image_base64 = image_to_base64(image)
        result = call_qianwen_api(image_base64, api_key)
        
        if result and "players" in result:
            cleaned_players = []
            skipped_players = []
            
            for player in result["players"]:
                name = player.get("name", "").strip()
                score = player.get("score", 0)
                hands = player.get("hands", 0)
                buyin = player.get("buyin", 0)
                
                name = clean_player_name(name)
                
                if isinstance(score, str):
                    score = parse_number_string(score)
                if isinstance(hands, str):
                    hands = parse_number_string(hands)
                if isinstance(buyin, str):
                    buyin = parse_number_string(buyin)
                
                # 验证数据完整性
                is_valid, error_msg = validate_player_data(name, score, hands, buyin)
                
                if is_valid:
                    cleaned_players.append({
                        "name": name,
                        "score": int(score),
                        "hands": int(hands) if hands else 0,
                        "buyin": int(buyin) if buyin else 0
                    })
                else:
                    skipped_players.append({
                        "name": name,
                        "score": score,
                        "hands": hands,
                        "buyin": buyin,
                        "error": error_msg
                    })
            
            game_time = result.get("game_time", "")
            total_hands = result.get("total_hands", 0)
            
            if not game_time:
                game_time = datetime.now().strftime("%m-%d %H:%M")
            
            return {
                "game_time": game_time,
                "total_hands": int(total_hands) if total_hands else 0,
                "players": cleaned_players,
                "skipped_players": skipped_players,
                "raw_response": result
            }
        else:
            return None
            
    except Exception as e:
        st.error(f"千问API解析错误: {e}")
        return None

def clean_player_name(name):
    """清理玩家昵称"""
    if not name:
        return ""
    
    # 移除开头的排名数字
    cleaned = re.sub(r'^\d+[\.\s]*', '', name)
    
    # 保留中文、英文、数字和常见符号
    cleaned = re.sub(r'[^\w\u4e00-\u9fff\-_\.\(\)\[\]（）【】\{\}]', '', cleaned)
    
    return cleaned.strip()

def parse_number_string(num_str):
    """解析数字字符串"""
    if isinstance(num_str, (int, float)):
        return int(num_str)  # 确保返回 int
    
    try:
        num_str = str(num_str).replace(',', '').replace(' ', '')
        return int(float(num_str))
    except (ValueError, TypeError):
        return 0

def normalize_datetime(game_time):
    """标准化日期时间格式 - 保留完整的日期和时间信息"""
    try:
        patterns = [
            r'(\d{4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{2})',
            r'(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{2})',
            r'(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, game_time)
            if match:
                groups = match.groups()
                if len(groups) == 5:
                    year, month, day, hour, minute = groups
                    return f"{year}-{month.zfill(2)}-{day.zfill(2)} {hour.zfill(2)}:{minute}"
                elif len(groups) == 4:
                    month, day, hour, minute = groups
                    current_year = datetime.now().year
                    return f"{current_year}-{month.zfill(2)}-{day.zfill(2)} {hour.zfill(2)}:{minute}"
        
        # 如果没有匹配到时间，使用当前时间
        return datetime.now().strftime("%Y-%m-%d %H:%M")
    except Exception as e:
        return datetime.now().strftime("%Y-%m-%d %H:%M")

def update_game_data(game_info):
    """更新游戏数据并触发表格刷新"""
    game_datetime = normalize_datetime(game_info['game_time'])
    
    # 添加时间到列表（如果不存在）
    if game_datetime not in st.session_state.game_data['dates']:
        st.session_state.game_data['dates'].append(game_datetime)
        st.session_state.game_data['dates'].sort()
    
    # 存储游戏信息
    st.session_state.game_data['game_info'][game_datetime] = {
        'total_hands': game_info.get('total_hands', 0),
        'total_buyin': sum([p.get('buyin', 0) for p in game_info['players']])
    }
    
    for player in game_info['players']:
        name = player['name']
        score = player['score']
        hands = player.get('hands', 0)
        buyin = player.get('buyin', 0)
        
        # 初始化玩家数据
        if name not in st.session_state.game_data['players']:
            st.session_state.game_data['players'][name] = {}
        
        # 只有当该玩家在该时间还没有数据时才添加
        if game_datetime not in st.session_state.game_data['players'][name]:
            st.session_state.game_data['players'][name][game_datetime] = {
                'score': score,
                'hands': hands,
                'buyin': buyin
            }
    
    # 触发表格更新
    st.session_state.table_update_trigger += 1

def delete_player(player_name):
    """删除玩家"""
    if player_name in st.session_state.game_data['players']:
        del st.session_state.game_data['players'][player_name]
        st.session_state.table_update_trigger += 1

def delete_date(date_time):
    """删除日期列"""
    if date_time in st.session_state.game_data['dates']:
        st.session_state.game_data['dates'].remove(date_time)
        for player_name in st.session_state.game_data['players']:
            if date_time in st.session_state.game_data['players'][player_name]:
                del st.session_state.game_data['players'][player_name][date_time]
        if date_time in st.session_state.game_data['game_info']:
            del st.session_state.game_data['game_info'][date_time]
        st.session_state.table_update_trigger += 1

def create_table_dataframe():
    """创建表格数据框 - 修改后的德州扑克统计表格"""
    players = st.session_state.game_data['players']
    dates = st.session_state.game_data['dates']
    
    if not players or not dates:
        return pd.DataFrame()
    
    # 获取所有玩家名称，按字母顺序排序
    all_players = sorted(players.keys())
    
    # 创建表格数据
    data = []
    
    for player_name in all_players:
        row = {'🃏 玩家昵称': player_name}
        total_score = 0
        total_hands = 0
        total_buyin = 0
        
        # 添加每场比赛的分数
        for date in dates:
            game_data = players[player_name].get(date, {'score': 0, 'hands': 0, 'buyin': 0})
            score = game_data.get('score', 0)
            hands = game_data.get('hands', 0)
            buyin = game_data.get('buyin', 0)
            
            # 使用完整的日期时间作为列名
            col_name = date  # 保留完整时间信息
            row[col_name] = score
            total_score += score
            total_hands += hands
            total_buyin += buyin
        
        # 添加统计列
        row['📊 总分'] = total_score
        row['🎯 总手数'] = total_hands
        row['💰 总带入'] = total_buyin
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    if df.empty:
        return df
    
    # 添加列求和行（最后一行）
    totals_row = {}
    
    # 计算每列的总和
    for col in df.columns:
        if col == '🃏 玩家昵称':
            totals_row[col] = '📈 总计'
        elif col in ['📊 总分', '🎯 总手数', '💰 总带入']:
            totals_row[col] = int(df[col].sum())
        else:
            # 每场比赛的总分
            totals_row[col] = int(df[col].sum())
    
    # 添加总计行
    totals_df = pd.DataFrame([totals_row])
    df = pd.concat([df, totals_df], ignore_index=True)
    
    return df

def handle_table_editing(edited_df, original_df):
    """处理表格编辑，同步更新数据并重新计算求和"""
    if edited_df is None or original_df is None:
        return
    
    try:
        dates = st.session_state.game_data['dates']
        
        # 更新除总计行外的所有数据
        for idx in range(len(edited_df) - 1):  # 排除最后一行总计行
            row = edited_df.iloc[idx]
            player_name = row['🃏 玩家昵称']
            
            if player_name and player_name != '📈 总计':
                # 更新每场比赛的分数
                for date in dates:
                    if date in row:
                        try:
                            new_score = row[date]
                            if pd.notna(new_score):
                                if player_name not in st.session_state.game_data['players']:
                                    st.session_state.game_data['players'][player_name] = {}
                                if date not in st.session_state.game_data['players'][player_name]:
                                    st.session_state.game_data['players'][player_name][date] = {
                                        'score': 0, 'hands': 0, 'buyin': 0
                                    }
                                st.session_state.game_data['players'][player_name][date]['score'] = int(float(new_score))
                            else:
                                if (player_name in st.session_state.game_data['players'] and 
                                    date in st.session_state.game_data['players'][player_name]):
                                    st.session_state.game_data['players'][player_name][date]['score'] = 0
                        except (ValueError, TypeError):
                            pass
        
        # 触发表格重新计算
        st.session_state.table_update_trigger += 1
        
    except Exception as e:
        st.error(f"更新数据时出错: {e}")

def display_interactive_table():
    """显示可交互的表格"""
    df = create_table_dataframe()
    
    if df.empty:
        st.info("🃏 请上传游戏截图开始统计你的德州扑克战绩")
        return None
    
    # 动态生成列配置
    column_config = {
        "🃏 玩家昵称": st.column_config.TextColumn("🃏 玩家昵称", width="medium", disabled=True),
        "📊 总分": st.column_config.NumberColumn("📊 总分", format="%d", disabled=True),
        "🎯 总手数": st.column_config.NumberColumn("🎯 总手数", format="%d", disabled=True),
        "💰 总带入": st.column_config.NumberColumn("💰 总带入", format="%d", disabled=True),
    }
    
    # 为每个比赛场次添加列配置
    dates = st.session_state.game_data['dates']
    for date in dates:
        # 显示简化的列名，但保留完整时间作为提示
        display_name = date.split(' ')
        if len(display_name) >= 2:
            short_name = f"{display_name[0].split('-')[-2]}-{display_name[0].split('-')[-1]} {display_name[1]}"
        else:
            short_name = date
        
        column_config[date] = st.column_config.NumberColumn(
            short_name,
            format="%d",
            help=f"比赛时间: {date}",
            width="small"
        )
    
    # 设置禁用列（包括统计列）
    disabled_columns = ['🃏 玩家昵称', '📊 总分', '🎯 总手数', '💰 总带入']
    
    # 显示可编辑表格
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        num_rows="fixed",
        disabled=disabled_columns,
        column_config=column_config,
        key=f"poker_table_{st.session_state.table_update_trigger}",
        hide_index=True
    )
    
    # 处理表格编辑
    if not edited_df.equals(df):
        handle_table_editing(edited_df, df)
        st.rerun()
    
    return df

def calculate_player_stats():
    """计算所有玩家的统计数据"""
    players = st.session_state.game_data['players']
    if not players:
        return {}
    
    stats = {}
    for player_name, games in players.items():
        total_score = sum([game.get('score', 0) for game in games.values()])
        total_hands = sum([game.get('hands', 0) for game in games.values()])
        total_buyin = sum([game.get('buyin', 0) for game in games.values()])
        games_played = len([game for game in games.values() if game.get('score', 0) != 0])
        
        stats[player_name] = {
            'total_score': total_score,
            'total_hands': total_hands,
            'total_buyin': total_buyin,
            'games_played': games_played,
            'avg_score_per_game': round(total_score / games_played, 2) if games_played > 0 else 0
        }
    
    return stats

def process_single_image(uploaded_file, container):
    """处理单张图片并实时更新显示"""
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    
    if file_id in st.session_state.processed_files:
        return
    
    with container:
        with st.spinner(f"🔄 正在解析 {uploaded_file.name}..."):
            try:
                image = Image.open(uploaded_file)
                
                # 显示正在解析的图片
                col_img, col_info = st.columns([1, 2])
                
                with col_img:
                    st.image(image, caption=f"解析中: {uploaded_file.name}", width=200)
                
                with col_info:
                    game_info = extract_game_info_with_qianwen(image, API_KEY)
                    
                    if game_info and game_info['players']:
                        # 更新数据
                        update_game_data(game_info)
                        st.session_state.processed_files.add(file_id)
                        
                        # 显示解析结果
                        st.success(f"✅ 解析成功！")
                        st.write(f"**时间:** {normalize_datetime(game_info['game_time'])}")
                        st.write(f"**总手数:** {game_info.get('total_hands', 0)}")
                        st.write(f"**有效玩家:** {len(game_info['players'])} 个")
                        
                        # 显示有效玩家
                        for player in game_info['players']:
                            st.write(f"✅ {player['name']}: {player['score']} (手数:{player.get('hands', 0)}, 带入:{player.get('buyin', 0)})")
                        
                        # 显示跳过的玩家
                        if game_info.get('skipped_players'):
                            with st.expander("⚠️ 跳过的数据"):
                                for skipped in game_info['skipped_players']:
                                    st.write(f"❌ {skipped['name']} ({skipped['error']})")
                        
                        # 强制刷新页面以更新统计表格
                        st.rerun()
                        
                    else:
                        st.error(f"❌ 解析失败或无有效数据")
                        
            except Exception as e:
                st.error(f"处理出错: {e}")

def main():
    st.title("🃏 德州扑克战绩分析师 🎰")
    st.markdown("### 🎯 *让数据说话，让技术提升！* 📈")
    st.markdown("---")
    
    # 创建主要布局
    left_col, right_col = st.columns([1, 3])
    
    with left_col:
        st.header("🎮 操作面板")
        
        # 文件上传
        st.subheader("📷 上传游戏截图")
        uploaded_files = st.file_uploader(
            "选择图片文件",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            accept_multiple_files=True,
            help="支持多张图片，逐个解析"
        )
        
        # 实时统计看板
        st.markdown("---")
        st.subheader("🏆 实时战绩")
        
        player_stats = calculate_player_stats()
        if player_stats:
            # 找出最佳表现的玩家
            best_player = max(player_stats.items(), key=lambda x: x[1]['total_score'])
            worst_player = min(player_stats.items(), key=lambda x: x[1]['total_score'])
            most_active = max(player_stats.items(), key=lambda x: x[1]['total_hands'])
            
            st.success(f"🥇 **最佳表现**: {best_player[0]}")
            st.write(f"💰 总分: {best_player[1]['total_score']}")
            
            st.error(f"🐟 **需要加油**: {worst_player[0]}")
            st.write(f"💸 总分: {worst_player[1]['total_score']}")
            
            st.info(f"🎯 **最勤奋玩家**: {most_active[0]}")
            st.write(f"🃏 {most_active[1]['total_hands']} 手牌")
        
        # 批量删除功能
        st.markdown("---")
        st.subheader("🗑️ 数据管理")
        
        if st.session_state.game_data['players']:
            # 删除玩家
            player_to_delete = st.selectbox(
                "删除玩家",
                [""] + list(st.session_state.game_data['players'].keys()),
                help="删除某个玩家的所有数据"
            )
            if st.button("🗑️ 删除玩家", disabled=not player_to_delete):
                delete_player(player_to_delete)
                st.success(f"已删除玩家: {player_to_delete}")
                st.rerun()
        
        if st.session_state.game_data['dates']:
            # 删除场次
            date_to_delete = st.selectbox(
                "删除整个场次",
                [""] + st.session_state.game_data['dates'],
                help="删除某个场次的所有数据"
            )
            if st.button("🗑️ 删除场次", disabled=not date_to_delete):
                delete_date(date_to_delete)
                st.success(f"已删除场次: {date_to_delete}")
                st.rerun()
        
        if st.button("💣 清空所有数据", type="secondary"):
            if st.button("⚠️ 确认清空", type="secondary"):
                st.session_state.game_data = {'players': {}, 'dates': [], 'game_info': {}}
                st.session_state.processed_files = set()
                st.session_state.table_update_trigger += 1
                st.rerun()
        
        # 统计信息
        st.markdown("---")
        st.subheader("📊 数据概览")
        st.metric("🎰 已解析场次", len(st.session_state.game_data['dates']))
        st.metric("🃏 参与玩家", len(st.session_state.game_data['players']))
        
        total_hands = sum([sum([game.get('hands', 0) for game in games.values()]) 
                          for games in st.session_state.game_data['players'].values()])
        st.metric("🎯 总手数", total_hands)
    
    with right_col:
        # 创建标签页
        tab1, tab2 = st.tabs(["📊 德州战绩分析", "🔍 解析进度"])
        
        with tab1:
            st.header("🃏 德州扑克战绩统计表")
            
            # 表格说明
            if st.session_state.game_data['players']:
                st.info("💡 **表格说明**: 每行是一个玩家，每列是一场比赛(精确到分钟)，显示总分、总手数、总带入")
                st.info("✏️ **编辑功能**: 可以直接点击分数格子进行修改，修改后统计数据会自动重新计算")
                st.warning("⚠️ **注意**: 最后一行'总计'不参与排序，相同日期不同时间会分为不同场次")
            
            # 显示交互式表格
            df = display_interactive_table()
            
            if df is not None and not df.empty:
                # 快捷操作按钮
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button("🔄 刷新表格"):
                        st.session_state.table_update_trigger += 1
                        st.rerun()
                
                with col2:
                    if st.button("📊 重新计算"):
                        st.session_state.table_update_trigger += 1
                        st.rerun()
                
                with col3:
                    # 下载功能
                    csv = df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 下载CSV",
                        data=csv,
                        file_name=f"德州扑克战绩_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col4:
                    # 导出详细统计
                    player_stats = calculate_player_stats()
                    if player_stats:
                        stats_df = pd.DataFrame.from_dict(player_stats, orient='index')
                        stats_csv = stats_df.to_csv(encoding='utf-8-sig')
                        st.download_button(
                            label="📈 统计报告",
                            data=stats_csv,
                            file_name=f"玩家统计_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
        
        with tab2:
            st.header("🔍 解析进度")
            
            # 解析进度容器
            progress_container = st.container()
            
            # 处理上传的文件
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    process_single_image(uploaded_file, progress_container)
            
            # 显示已处理文件
            if st.session_state.processed_files:
                st.subheader("✅ 已处理文件")
                for file_id in st.session_state.processed_files:
                    file_name = file_id.split('_')[0]
                    st.write(f"📁 {file_name}")
    
    # 使用说明
    with st.expander("💡 德州扑克统计使用说明"):
        st.markdown("""
        **🃏 功能特点：**
        - 📊 **专业统计**: 总分、总手数、总带入等德州扑克专用指标
        - 🎯 **精确时间**: 相同日期不同时间的比赛会分为不同场次
        - ✏️ **灵活编辑**: 直接在表格中修改分数，所有统计自动更新
        - 🏆 **排行榜**: 实时显示最佳/最差表现和最活跃玩家
        
        **📈 统计指标说明：**
        - **总分**: 该玩家所有比赛的总分数
        - **总手数**: 该玩家参与的总手牌数量
        - **总带入**: 该玩家所有比赛的带入金额总和
        - **最后一行**: 总计行，不参与排序
        
        **🎮 操作指南：**
        1. 上传德州扑克游戏截图，系统自动解析战绩、手数和带入
        2. 在统计表中直接点击分数格子进行修改
        3. 查看实时战绩排行榜了解表现
        4. 下载CSV文件进行更深入的数据分析
        
        **⏰ 时间识别：**
        - 系统会精确识别到分钟级别的时间
        - 相同日期不同时间的比赛会分为不同列
        - 只有完全相同的日期和时间才会归为同一场比赛
        """)

if __name__ == "__main__":
    main()

