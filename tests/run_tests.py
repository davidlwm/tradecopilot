#!/usr/bin/env python
"""
多因子买入信号评分系统 - 测试运行脚本

运行所有测试或指定的测试模块
"""

import sys
import unittest
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 测试模块映射
TEST_MODULES = {
    'data': 'test_data',
    'indicators': 'test_indicators',
    'scorer': 'test_scorer',
    'visualization': 'test_visualization',
    'integration': 'test_integration',
    'all': None  # 表示所有测试
}

def run_tests(modules=None):
    """
    运行指定的测试模块或所有测试
    
    参数:
    modules (list): 要运行的测试模块列表，如果为None则运行所有测试
    """
    if modules and 'all' in modules:
        modules = None
    
    if modules:
        # 运行指定的测试模块
        suite = unittest.TestSuite()
        
        for module_name in modules:
            if module_name not in TEST_MODULES:
                print(f"警告：未知的测试模块 '{module_name}'")
                continue
                
            module_path = TEST_MODULES[module_name]
            try:
                # 导入测试模块
                module = __import__(module_path, fromlist=['*'])
                
                # 添加模块中的所有测试用例
                for name in dir(module):
                    obj = getattr(module, name)
                    if isinstance(obj, type) and issubclass(obj, unittest.TestCase) and obj != unittest.TestCase:
                        suite.addTest(unittest.makeSuite(obj))
                        
            except ImportError as e:
                print(f"无法导入测试模块 '{module_path}': {e}")
        
        # 运行测试套件
        if suite.countTestCases() > 0:
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            return result.wasSuccessful()
        else:
            print("没有找到可运行的测试")
            return False
    else:
        # 运行所有测试
        suite = unittest.defaultTestLoader.discover(os.path.dirname(__file__))
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return result.wasSuccessful()

def print_usage():
    """打印使用说明"""
    print("用法: python run_tests.py [module1 module2 ...]")
    print("可用的测试模块:")
    for module in sorted(TEST_MODULES.keys()):
        print(f"  {module}")
    print("\n如果不指定模块，将运行所有测试")

if __name__ == "__main__":
    # 解析命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print_usage()
            sys.exit(0)
            
        modules = sys.argv[1:]
    else:
        modules = ['all']
    
    # 运行测试
    print(f"运行测试模块: {', '.join(modules)}")
    success = run_tests(modules)
    
    # 设置退出代码
    sys.exit(0 if success else 1)