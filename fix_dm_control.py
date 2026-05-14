#!/usr/bin/env python3
"""
修复 dm_control 在无头服务器上的渲染环境。
运行方式: python fix_dm_control.py
之后在同一个终端中即可正常使用 dm_control 的渲染功能。

原理：
1. 永久修复 PyOpenGL 的 OSMesa platform（修改 osmesa.py）
2. 创建 sitecustomize.py 自动修复 libstdc++ 问题
3. 启动 Xvfb 虚拟显示
"""

import os
import sys
import subprocess
import time
import shutil


def fix_pyopengl_osmesa():
    """永久修复 PyOpenGL 的 OSMesa platform"""
    # 定位 osmesa.py
    import OpenGL.platform.osmesa as m
    src = m.__file__

    bak = src + '.bak'
    if not os.path.exists(bak):
        shutil.copy2(src, bak)
        print(f'[INFO] 已备份原文件: {bak}')

    with open(src) as f:
        content = f.read()

    # 检查是否已经修复过
    if "loadLibrary(ctypes.cdll, 'GL'" in content:
        print('[OK] PyOpenGL OSMesa platform 已修复（无需重复修复）')
        return True

    # 1. 添加 LD_PRELOAD 修复（在 import 之后，from OpenGL.platform 之前）
    old_header = """import ctypes, ctypes.util
from OpenGL.platform import baseplatform, ctypesloader"""

    new_header = """import ctypes, ctypes.util
import os as _os

# 修复 conda 环境的 libstdc++ 版本过旧问题
_system_libstdcpp = '/usr/lib/x86_64-linux-gnu/libstdc++.so.6'
if _os.path.exists(_system_libstdcpp):
    _old_preload = _os.environ.get('LD_PRELOAD', '')
    if _system_libstdcpp not in _old_preload:
        _os.environ['LD_PRELOAD'] = _system_libstdcpp + (':' + _old_preload if _old_preload else '')

from OpenGL.platform import baseplatform, ctypesloader"""

    content = content.replace(old_header, new_header)

    # 2. 修改 GL 属性：加载 libGL.so 而不是 libOSMesa.so
    old_gl = """    def GL(self):
        try:
            return ctypesloader.loadLibrary(
                ctypes.cdll,
                'OSMesa', 
                mode=ctypes.RTLD_GLOBAL 
            ) 
        except OSError as err:
            raise ImportError("Unable to load OpenGL library", *err.args)"""

    new_gl = """    def GL(self):
        try:
            return ctypesloader.loadLibrary(
                ctypes.cdll,
                'GL', 
                mode=ctypes.RTLD_GLOBAL 
            ) 
        except OSError as err:
            raise ImportError("Unable to load OpenGL library", *err.args)"""

    content = content.replace(old_gl, new_gl)

    # 3. 修改 OSMesa 属性：单独加载 libOSMesa.so
    old_osmesa = """    def OSMesa( self ): return self.GL"""
    new_osmesa = """    def OSMesa( self ):
        try:
            return ctypesloader.loadLibrary(
                ctypes.cdll,
                'OSMesa', 
                mode=ctypes.RTLD_GLOBAL 
            ) 
        except OSError as err:
            raise ImportError("Unable to load OSMesa library", *err.args)"""

    content = content.replace(old_osmesa, new_osmesa)

    with open(src, 'w') as f:
        f.write(content)

    print(f'[OK] PyOpenGL OSMesa platform 已修复 ({src})')
    return True


def fix_sitecustomize():
    """创建 sitecustomize.py 自动修复 libstdc++ 问题"""
    python_lib = os.path.join(os.path.dirname(sys.executable), '../lib/python3.12/site-packages')
    python_lib = os.path.normpath(python_lib)

    if not os.path.exists(python_lib):
        python_lib = os.path.join(os.path.dirname(sys.executable), '../lib/python*/site-packages')
        import glob
        matches = glob.glob(python_lib)
        if matches:
            python_lib = matches[0]
        else:
            print(f'[ERROR] 找不到 site-packages 目录')
            return False

    site_path = os.path.join(python_lib, 'sitecustomize.py')

    # 检查是否已存在
    if os.path.exists(site_path):
        with open(site_path) as f:
            if 'libstdc++' in f.read():
                print('[OK] sitecustomize.py 已存在且包含修复代码')
                return True

    with open(site_path, 'w') as f:
        f.write('''"""Auto-fix libstdc++ for conda environments."""
import os
import ctypes

_system_libstdcpp = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
if os.path.exists(_system_libstdcpp):
    try:
        ctypes.cdll.LoadLibrary(_system_libstdcpp)
    except Exception:
        pass
''')

    print(f'[OK] sitecustomize.py 已创建 ({site_path})')
    return True


def ensure_xvfb():
    """确保 Xvfb 虚拟显示在运行"""
    if 'DISPLAY' in os.environ:
        print(f'[OK] DISPLAY 已设置: {os.environ["DISPLAY"]}')
        return True

    if not subprocess.run(['which', 'Xvfb'], capture_output=True).returncode == 0:
        print('[ERROR] Xvfb 未安装，请执行: apt install xvfb')
        return False

    # 杀掉旧的 Xvfb
    subprocess.run(['pkill', '-f', 'Xvfb'], capture_output=True)
    time.sleep(0.5)

    display_num = os.environ.get('DISPLAY_NUM', '99')
    lock_file = f'/tmp/.X{display_num}-lock'
    if os.path.exists(lock_file):
        os.remove(lock_file)

    proc = subprocess.Popen(
        ['Xvfb', f':{display_num}', '-screen', '0', '1024x768x24'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(1)

    if proc.poll() is None:
        os.environ['DISPLAY'] = f':{display_num}'
        print(f'[OK] Xvfb 已启动 (:{display_num})')
        return True
    else:
        print('[ERROR] Xvfb 启动失败')
        return False



def setup_autostart():
    """设置 Xvfb 开机自启"""
    # 尝试 systemd
    service_path = '/etc/systemd/system/xvfb.service'
    if os.path.exists('/run/systemd/system'):
        service = '''[Unit]
Description=X Virtual Frame Buffer Service
After=network.target

[Service]
ExecStart=/usr/bin/Xvfb :99 -screen 0 1024x768x24
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
'''
        with open(service_path, 'w') as f:
            f.write(service)
        subprocess.run(['systemctl', 'enable', 'xvfb'], capture_output=True)
        print('[OK] 已设置 Xvfb 开机自启 (systemd)')
        return True

    # 尝试 rc.local
    rc_local = '/etc/rc.local'
    if os.path.exists(rc_local):
        with open(rc_local) as f:
            if 'Xvfb' in f.read():
                print('[OK] rc.local 中已有 Xvfb 自启配置')
                return True

    with open(rc_local, 'w') as f:
        f.write('#!/bin/bash\n/usr/bin/Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &\nexit 0\n')
    os.chmod(rc_local, 0o755)
    print('[OK] 已设置 Xvfb 开机自启 (rc.local)')
    return True

def main():
    print('=== dm_control 渲染环境修复 ===')
    print()

    ok = all([
        fix_pyopengl_osmesa(),
        fix_sitecustomize(),
        ensure_xvfb(),
        setup_autostart(),
    ])

    print()
    if ok:
        print('=== 修复完成！现在可以正常使用 dm_control 渲染 ===')
        print()
        print('验证方法:')
        print('  python -c "')
        print('  from dm_control import suite')
        print('  import numpy as np')
        print('  env = suite.load(\"cheetah\", \"run\")')
        print('  env.reset()')
        print('  obs = env.physics.render(64, 64, camera_id=0).astype(np.uint8)')
        print('  print(\"渲染成功, shape:\", obs.shape)')
        print('  "')
    else:
        print('=== 部分修复失败，请检查上方警告信息 ===')
        sys.exit(1)


if __name__ == '__main__':
    main()

# fix_dm_control.py 的原理是解决三个问题：                                                                                                 

# 1. PyOpenGL 的 osmesa.py 有 bug                                                                                                          

# 原版 OSMesaPlatform 类的 GL 属性加载的是 libOSMesa.so，但 libOSMesa.so 里没有 glGetError 这个函数。当 OpenGL 的 _errors.py 调用          
# _p.GL.glGetError 时，就报 AttributeError: 'NoneType' object has no attribute 'glGetError'。                                              

# 修复脚本把 GL 属性改为加载 libGL.so（里面有 glGetError），把 OSMesa 属性改为单独加载 libOSMesa.so（原版直接返回 self.GL，也是错的）。    

# 2. conda 环境的 libstdc++.so.6 版本太旧                                                                                                  

# conda 环境里的 libstdc++.so.6 是 6.0.29 版本，缺少 GLIBCXX_3.4.30。而 libOSMesa.so 依赖 libLLVM-15.so.1，后者需要                        
# GLIBCXX_3.4.30，所以加载 libOSMesa.so 时会失败。                                                                                         

# 修复脚本创建了 sitecustomize.py，在 Python 启动时预先加载系统的 libstdc++.so.6（在 /usr/lib/x86_64-linux-gnu/ 下，版本是                 
# 6.0.30）。同时也在 osmesa.py 文件顶部设置了 LD_PRELOAD 环境变量，双重保障。                                                              

# 3. 无头服务器没有显示环境                                                                                                                

# dm_control 的渲染需要 OpenGL context，而无头服务器没有显示器。修复脚本启动了 Xvfb（虚拟帧缓冲）来模拟一个虚拟显示器，并设置 DISPLAY=:99。

# -----------------------------------------------------------------------------------------------------------------------------------------

# 这三个修复中，前两个是永久性的（改了文件），第三个（Xvfb）是进程级的，容器重启后需要重新启动。     