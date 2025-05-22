import os
import yaml


def load_config(config_path='config.yaml'):
    """
    加载YAML配置文件
    
    参数:
        config_path: 配置文件路径
    
    返回:
        config: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建必要的目录
    os.makedirs(os.path.dirname(config['output']['model_path']), exist_ok=True)
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    
    return config
