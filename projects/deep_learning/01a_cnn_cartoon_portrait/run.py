"""从项目目录直接运行；兼容常规环境安装和项目内 .deps 安装。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
if (ROOT / ".deps").is_dir():
    sys.path.insert(0, str(ROOT / ".deps"))

from cartoon_portrait.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
