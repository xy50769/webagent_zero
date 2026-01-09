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
    
    # 扩充后的可交互元素类型列表
    interactive_roles = {
        'button', 'link', 'textbox', 'combobox', 'checkbox', 
        'radio', 'menuitem', 'tab', 'searchbox', 'listitem',
        'image', 'img',
        'option',
        'switch', 'slider',
        'menuitemcheckbox', 'menuitemradio'
    }
    
    # 使用回溯引用的正则表达式，解决嵌套引号问题（如 12'' Phone）
    # \3 必须匹配和 Group 3 一模一样的结束引号
    pattern = re.compile(r'^\s*\[(\d+)\]\s+(\w+)(?:\s+([\'"])(.*?)\3)?')
    
    # 逐行处理，避免跨行匹配干扰
    lines = axtree_txt.splitlines()
    
    for line in lines:
        if not line.strip():
            continue
            
        match = pattern.match(line)
        if match:
            bid = match.group(1)
            role = match.group(2).lower()
            text = match.group(4) if match.group(4) else ""
            
            if role in interactive_roles:
                # 过滤掉 listitem 且 text 为空的元素（通常是容器，不需要交互）
                if role == 'listitem' and not text.strip():
                    continue
                
                element = {
                    "bid": bid,
                    "role": role,
                    "text": text[:200],
                }
                
                # 宽松的属性合并逻辑
                if extra_properties and bid in extra_properties:
                    props = extra_properties[bid]
                    element["clickable"] = props.get("clickable", True)
                    # 降低可见性阈值到 0.1，防止误杀
                    element["visible"] = props.get("visibility", 1.0) > 0.1
                else:
                    # 如果没有额外属性，默认全部开放，宁滥勿缺
                    element["clickable"] = True
                    element["visible"] = True
                
                elements.append(element)
    
    logger.info(f"Extracted {len(elements)} interactive elements from AXTree")
    return elements


