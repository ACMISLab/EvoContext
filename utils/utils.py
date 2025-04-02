import json
import os
def load_json(text):
    try:
        return json.loads(text)
    except:
        import re
        json_pattern = re.compile(r"```(\s*json)?$(.*?)^```", re.MULTILINE | re.DOTALL)
        matches = re.findall(json_pattern, text)
        if len(matches) > 0:
            text_json = matches[0][1].strip()
        else:
            print("没有找到JSON代码块")
            return {}
        try:
            return json.loads(text_json)
        except:
            print(f"JSON解析错误")
            return {}

def load_python(text):
    try:
        import re
        pattern = re.compile(r"```(\s*python)?$(.*?)^```", re.MULTILINE | re.DOTALL)
        matches = re.findall(pattern, text)
        if len(matches) > 0:
            return True, matches[0][1].strip()
        else:
            return False, "没有找到Python代码块"
    except Exception as e:
        return False, f"python文本解析错误: 错误为\n{e}"

def get_project_root():
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 向上一级
    current_directory = os.path.dirname(current_file_path)
    # 再次向上一级
    project_root = os.path.dirname(current_directory)

    return project_root
