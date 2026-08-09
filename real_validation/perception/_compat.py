"""perception 共享的 opencv 惰性导入。

import 本模块零副作用(保持 import 卫生:只 import perception.skeleton 时不拉入 cv2)。
"""

try:
    import cv2
    _CV2_ERR = None
except ImportError as exc:  # pragma: no cover
    cv2 = None
    _CV2_ERR = exc


def require_cv2():
    if cv2 is None:
        raise RuntimeError(f"需要 opencv：{_CV2_ERR}")
    return cv2
