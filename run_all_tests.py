#!/usr/bin/env python3
"""
项目级统一测试运行器
自动发现并运行deeplearning-platform下所有测试
"""

import os
import sys
import subprocess
from pathlib import Path


class ProjectTestRunner:
    """项目级测试运行器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_root = self.project_root / "test"
        self.test_results = {}
        
    def find_test_directories(self):
        """查找所有测试目录"""
        test_dirs = []
        
        if self.test_root.exists():
            for item in self.test_root.iterdir():
                if item.is_dir():
                    # 检查是否有测试文件
                    test_files = list(item.glob("test_*.py")) + list(item.glob("*_test.py"))
                    if test_files:
                        test_dirs.append(item)
        
        return test_dirs
    
    def run_pytest_in_directory(self, test_dir):
        """在指定目录运行pytest"""
        dir_name = test_dir.name
        print(f"\n📁 运行 {dir_name} 的测试...")
        print("-" * 50)
        
        try:
            # 检查是否有pytest配置文件
            pytest_files = list(test_dir.glob("test_*.py")) + list(test_dir.glob("*_test.py"))
            
            if not pytest_files:
                print(f"⚠️  {dir_name} 没有找到测试文件")
                return False
            
            # 运行pytest
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 
                str(test_dir),
                '-v',
                '--tb=short'
            ], capture_output=True, text=True, cwd=str(test_dir))
            
            if result.returncode == 0:
                print(f"✅ {dir_name} 测试通过!")
                self.test_results[dir_name] = "PASSED"
                return True
            else:
                print(f"❌ {dir_name} 测试失败:")
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(result.stderr)
                self.test_results[dir_name] = "FAILED"
                return False
                
        except Exception as e:
            print(f"❌ 运行 {dir_name} 测试时出错: {e}")
            self.test_results[dir_name] = "ERROR"
            return False
    
    def run_integration_tests(self, test_dir):
        """运行集成测试"""
        dir_name = test_dir.name
        integration_files = list(test_dir.glob("integration_test.py")) + list(test_dir.glob("*integration*.py"))
        
        for integration_file in integration_files:
            print(f"\n🔗 运行 {dir_name} 的集成测试: {integration_file.name}")
            print("-" * 50)
            
            try:
                result = subprocess.run([
                    sys.executable, str(integration_file)
                ], capture_output=True, text=True, cwd=str(test_dir))
                
                if result.returncode == 0:
                    print(f"✅ {dir_name} 集成测试通过!")
                    self.test_results[f"{dir_name}_integration"] = "PASSED"
                else:
                    print(f"❌ {dir_name} 集成测试失败:")
                    if result.stdout:
                        print(result.stdout)
                    if result.stderr:
                        print(result.stderr)
                    self.test_results[f"{dir_name}_integration"] = "FAILED"
                    
            except Exception as e:
                print(f"❌ 运行 {dir_name} 集成测试时出错: {e}")
                self.test_results[f"{dir_name}_integration"] = "ERROR"
    
    def run_custom_test_scripts(self):
        """运行自定义测试脚本"""
        custom_scripts = [
            "run_tests.py",
            "test_runner.py",
            "run_all_tests.py"
        ]
        
        for test_dir in self.find_test_directories():
            for script_name in custom_scripts:
                script_path = test_dir / script_name
                if script_path.exists() and script_path != Path(__file__):
                    dir_name = test_dir.name
                    print(f"\n⚙️  运行 {dir_name} 的自定义测试: {script_name}")
                    print("-" * 50)
                    
                    try:
                        result = subprocess.run([
                            sys.executable, str(script_path)
                        ], capture_output=True, text=True, cwd=str(test_dir))
                        
                        if result.returncode == 0:
                            print(f"✅ {dir_name} 自定义测试通过!")
                            self.test_results[f"{dir_name}_custom"] = "PASSED"
                        else:
                            print(f"❌ {dir_name} 自定义测试失败")
                            self.test_results[f"{dir_name}_custom"] = "FAILED"
                            
                    except Exception as e:
                        print(f"❌ 运行 {dir_name} 自定义测试时出错: {e}")
                        self.test_results[f"{dir_name}_custom"] = "ERROR"
    
    def run_all_tests(self, mode="all"):
        """运行所有测试"""
        print("🚀 项目级测试运行器")
        print("=" * 60)
        print(f"项目根目录: {self.project_root}")
        print(f"测试目录: {self.test_root}")
        
        test_dirs = self.find_test_directories()
        
        if not test_dirs:
            print("⚠️  没有找到测试目录")
            return False
        
        print(f"\n📊 发现 {len(test_dirs)} 个测试模块:")
        for dir_path in test_dirs:
            print(f"  - {dir_path.name}")
        
        print("\n" + "=" * 60)
        
        # 根据模式运行测试
        if mode in ["all", "unit"]:
            for test_dir in test_dirs:
                self.run_pytest_in_directory(test_dir)
        
        if mode in ["all", "integration"]:
            for test_dir in test_dirs:
                self.run_integration_tests(test_dir)
        
        if mode in ["all", "custom"]:
            self.run_custom_test_scripts()
        
        # 显示总结
        self.show_summary()
        
        # 返回是否所有测试都通过
        return all(status == "PASSED" for status in self.test_results.values())
    
    def show_summary(self):
        """显示测试总结"""
        print("\n" + "=" * 60)
        print("📊 测试结果总结:")
        print("-" * 30)
        
        passed = sum(1 for status in self.test_results.values() if status == "PASSED")
        failed = len(self.test_results) - passed
        
        for test_name, status in self.test_results.items():
            status_icon = "✅" if status == "PASSED" else "❌"
            print(f"{status_icon} {test_name}: {status}")
        
        print(f"\n总计: {len(self.test_results)} 个测试套件")
        print(f"通过: {passed}, 失败: {failed}")
        
        if failed == 0:
            print("🎉 所有测试通过!")
        else:
            print(f"⚠️  {failed} 个测试套件失败")
    
    def list_available_tests(self):
        """列出所有可用测试"""
        print("📋 可用测试列表:")
        print("=" * 40)
        
        test_dirs = self.find_test_directories()
        
        for test_dir in test_dirs:
            print(f"\n📁 {test_dir.name}:")
            
            # 查找测试文件
            test_files = list(test_dir.glob("test_*.py")) + list(test_dir.glob("*_test.py"))
            for test_file in test_files:
                print(f"  - {test_file.name}")
            
            # 查找集成测试
            integration_files = list(test_dir.glob("integration_test.py")) + list(test_dir.glob("*integration*.py"))
            for int_file in integration_files:
                print(f"  - {int_file.name} (集成测试)")
            
            # 查找自定义脚本
            custom_scripts = ["run_tests.py", "test_runner.py", "run_all_tests.py"]
            for script in custom_scripts:
                script_path = test_dir / script
                if script_path.exists() and script_path != Path(__file__):
                    print(f"  - {script} (自定义)")


def main():
    """主函数"""
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "all"
    
    runner = ProjectTestRunner()
    
    if mode == "list":
        runner.list_available_tests()
        return 0
    
    elif mode in ["all", "unit", "integration", "custom"]:
        success = runner.run_all_tests(mode)
        return 0 if success else 1
    
    else:
        print("使用方法:")
        print("  python run_all_tests.py           # 运行所有测试")
        print("  python run_all_tests.py all       # 运行所有测试")
        print("  python run_all_tests.py unit      # 只运行单元测试")
        print("  python run_all_tests.py integration  # 只运行集成测试")
        print("  python run_all_tests.py custom    # 只运行自定义测试")
        print("  python run_all_tests.py list      # 列出所有可用测试")
        return 1


if __name__ == "__main__":
    sys.exit(main())