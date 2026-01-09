"""
Element Extraction and Tracking Utilities for Explorer

提供从 AXTree 中提取可交互元素、解析 action 中的元素 ID 等功能
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_interactive_elements(axtree_txt: str, extra_properties: dict = None) -> list[dict]:
    """
    从 AXTree 文本中提取所有可交互元素
    
    Args:
        axtree_txt: AXTree 的文本表示
        extra_properties: 额外的元素属性（如 clickable, visibility）
        
    Returns:
        list[dict]: 可交互元素列表
        [
            {
                "bid": "123",
                "role": "button",
                "text": "Add to Cart",
                "clickable": True,
                "visible": True
            },
            ...
        ]
    """
    elements = []
    
    # 可交互的元素类型
    interactive_roles = {
        'button', 'link', 'textbox', 'combobox', 'checkbox', 
        'radio', 'menuitem', 'tab', 'searchbox', 'listitem',
        'img'  
    }
    
    # 正则模式：匹配 [bid] role 'text' 或 [bid] role "text"
    # 例如：[1398] button 'Add to Cart'
    #       [256] link 'Beauty & Personal Care'
    pattern = r'\[(\d+)\]\s+(\w+)(?:\s+[\'"](.*?)[\'"])?'
    
    for match in re.finditer(pattern, axtree_txt):
        bid = match.group(1)
        role = match.group(2).lower()
        text = match.group(3) if match.group(3) else ""
        
        if role in interactive_roles:
            # 过滤掉 listitem 且 text 为空的元素（通常是容器，不需要交互）
            if role == 'listitem' and not text.strip():
                continue
            
            element = {
                "bid": bid,
                "role": role,
                "text": text[:100] if text else "", 
            }
            
            # 添加额外属性（如果有）
            if extra_properties and bid in extra_properties:
                props = extra_properties[bid]
                element["clickable"] = props.get("clickable", True)
                element["visible"] = props.get("visibility", 0) > 0.5
            else:
                element["clickable"] = True
                element["visible"] = True
            
            elements.append(element)
    
    logger.debug(f"Extracted {len(elements)} interactive elements from AXTree")
    return elements


def extract_bid_from_action(action: str) -> Optional[str]:
    """
    从 action 字符串中提取 bid
    
    Args:
        action: 如 "click('1398')" 或 'fill("123", "text")'
        
    Returns:
        str: bid (如 "1398")，如果未找到则返回 None
    """
    # 匹配第一个引号中的数字
    match = re.search(r"['\"](\d+)['\"]", action)
    if match:
        return match.group(1)
    return None


def extract_action_type(action: str) -> Optional[str]:
    """
    从 action 字符串中提取动作类型
    
    Args:
        action: 如 "click('1398')"
        
    Returns:
        str: 动作类型 (如 "click", "fill", "hover")
    """
    if not action:
        return None
    
    # 提取第一个单词（动作类型）
    # 支持有括号的（如 click('123')）和无括号的（如 scroll）
    match = re.match(r'^(\w+)', action)
    if match:
        action_type = match.group(1).lower()
        # 只返回动作类型，如果后面有括号的话
        if '(' in action:
            return action_type
        # scroll 等特殊动作没有括号，也返回
        return action_type
    
    return None


def find_element_in_axtree(bid: str, axtree_txt: str) -> Optional[dict]:
    """
    在 AXTree 中查找指定 bid 的元素信息
    
    Args:
        bid: 元素的 bid
        axtree_txt: AXTree 文本
        
    Returns:
        dict: 元素信息，如 {"role": "button", "text": "Submit"}
    """
    # 精确匹配该 bid 的行
    pattern = rf'\[{bid}\]\s+(\w+)(?:\s+[\'"](.*?)[\'"])?'
    match = re.search(pattern, axtree_txt)
    
    if match:
        return {
            "role": match.group(1).lower(),
            "text": match.group(2) if match.group(2) else ""
        }
    
    return None


# def filter_unvisited_elements(
#     all_elements: list[dict], 
#     visited_elements: dict[str, dict]
# ) -> list[dict]:
#     """
#     过滤出未访问的元素
    
#     Args:
#         all_elements: 当前页面的所有可交互元素
#         visited_elements: 已访问的元素字典 {bid: interaction_info}
        
#     Returns:
#         list[dict]: 未访问的元素列表
#     """
#     unvisited = []
#     for elem in all_elements:
#         bid = elem["bid"]
#         if bid not in visited_elements:
#             unvisited.append(elem)
    
#     return unvisited

