"""对已有漫画结果生成细粒度观察页。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
if (ROOT / ".deps").is_dir():
    sys.path.insert(0, str(ROOT / ".deps"))

from cartoon_portrait.inspection import main

if __name__ == "__main__":
    raise SystemExit(main())
