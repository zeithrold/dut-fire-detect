# -*- mode: python ; coding: utf-8 -*-
import platform
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []
datas += collect_data_files('gradio')
datas += collect_data_files('gradio_client')
tmp_ret = collect_all('onnxruntime')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

if platform.system() == 'Windows':
    temp_ret = collect_all('msvc-runtime')
    datas += temp_ret[0]; binaries += temp_ret[1]; hiddenimports += temp_ret[2]

a = Analysis(
    ['dut_fire_detect/__main__.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
    module_collection_mode={
        'gradio': 'py',  # Collect gradio package as source .py files
    },
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='fire-detect',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='fire-detect',
)
