"""
Element Extraction and Tracking Utilities for Explorer

"""
import re
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def extract_interactive_elements(axtree_txt: str, extra_properties: dict = None) -> list[dict]:
    """
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


