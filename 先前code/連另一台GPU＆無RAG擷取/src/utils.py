import re
from datetime import date

# ===== 身分證檢查碼 =====
LETTER_MAP = {
    'A':10,'B':11,'C':12,'D':13,'E':14,'F':15,'G':16,'H':17,'I':34,
    'J':18,'K':19,'L':20,'M':21,'N':22,'O':35,'P':23,'Q':24,'R':25,
    'S':26,'T':27,'U':28,'V':29,'W':32,'X':30,'Y':31,'Z':33
}
ID_RE = re.compile(r'\b([A-Z])[12]\d{8}\b')

def validate_tw_id(twid: str) -> bool:
    m = ID_RE.fullmatch(twid)
    if not m:
        return False
    letter = m.group(1)
    nums = [int(x) for x in twid[1:]]  # 9 digits
    code = LETTER_MAP.get(letter)
    if code is None:
        return False
    a1, a2 = divmod(code, 10)
    weights = [1,9,8,7,6,5,4,3,2,1,1]  # A1,A2,d1..d9
    digits = [a1, a2] + nums
    return sum(w*d for w, d in zip(weights, digits)) % 10 == 0

# ===== 電話抽取與正規化 =====
# 支援：
# (03) 12345678 轉 123
# 02-12345678轉123或456
# 0212345678#123
# 03 1234567、(02)1234-5678 等
PHONE_PATTERNS = [
    r'\(?(0\d{1,2})\)?[ \-]?(\d{3,4})[ \-]?(\d{3,4})(?:[ \-]*(?:轉|ext\.?|#|分機)[ \-]?(?:([0-9\-、或/]{1,10})))?',
]
PHONE_RE = re.compile('|'.join(f'(?:{p})' for p in PHONE_PATTERNS))

def normalize_phone(m: re.Match) -> str:
    # 取出群組：1區碼, 2前段, 3後段, 4分機（可能含「或」）
    area, p1, p2, ext = m.group(1), m.group(2), m.group(3), m.group(4)
    base = f"{area}-{p1}{p2}"
    if ext:
        # 將「或」「、」「/」保留為「或」提示
        ext_norm = ext.replace('、', '或').replace('/', '或')
        return f"{base}#{ext_norm}"
    return base

def extract_phone(text: str) -> str | None:
    # 在「電話/聯絡/承辦」附近優先
    for kw in ['電話', '聯絡', '分機', '承辦', '書記官']:
        for m in PHONE_RE.finditer(text):
            span = m.span()
            ctx = text[max(0, span[0]-15):min(len(text), span[1]+15)]
            if kw in ctx:
                return normalize_phone(m)
    # 否則抓第一個
    m = PHONE_RE.search(text)
    return normalize_phone(m) if m else None

# ===== 承辦資訊抽取（職稱或姓名）=====
ROLE_WORDS = ['書記官','承辦人','聯絡人','承辦','股員','股長','專員']
NAME_HINT = r'[^\s，、()（）]{2,4}'  # 2-4字中文名的寬鬆提示
DATE_ROC_RE = re.compile(r'民國\s*(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日')
DATE_ISO_RE = re.compile(r'(20\d{2})[./\-年]?\s*(\d{1,2})[./\-月]?\s*(\d{1,2})\s*日?')

def normalize_date_roc_to_iso(y:int, m:int, d:int) -> str | None:
    try:
        y = 1911 + y
        return datetime(y, m, d).strftime("%Y-%m-%d")
    except Exception:
        return None

def extract_doc_date(text: str) -> str | None:
    """
    依關鍵詞就近抓一組日期；支援民國/西元多種寫法。
    優先關鍵詞：發文日期/發文日/來文日期/來文日/發文時間/來文時間
    """
    KEYWORDS = ['發文日期','發文日','來文日期','來文日','發文時間','來文時間']
    # 先掃有關鍵詞的區域
    for m in re.finditer('|'.join(map(re.escape, KEYWORDS)), text):
        start, end = m.start(), m.end()
        win = text[max(0, start-20):min(len(text), end+30)]
        # 民國優先
        m1 = DATE_ROC_RE.search(win)
        if m1:
            y, mm, dd = map(int, m1.groups())
            iso = normalize_date_roc_to_iso(y, mm, dd)
            if iso: return iso
        # 西元
        m2 = DATE_ISO_RE.search(win)
        if m2:
            y, mm, dd = map(int, m2.groups())
            try:
                return datetime(y, mm, dd).strftime("%Y-%m-%d")
            except Exception:
                pass
    # 無關鍵詞時，降級為全文第一組日期
    m1 = DATE_ROC_RE.search(text)
    if m1:
        y, mm, dd = map(int, m1.groups())
        return normalize_date_roc_to_iso(y, mm, dd)
    m2 = DATE_ISO_RE.search(text)
    if m2:
        y, mm, dd = map(int, m2.groups())
        try:
            return datetime(y, mm, dd).strftime("%Y-%m-%d")
        except Exception:
            return None
    return None

def extract_officer(text: str) -> tuple[str|None, str|None]:
    # 標記型態：「承辦人：王小明」、「書記官：林OO」、「聯絡人：張小姐」
    m = re.search(rf'({"|".join(ROLE_WORDS)})[：:]\s*({NAME_HINT})', text)
    if m:
        role = m.group(1)
        name = m.group(2)
        # 濾掉太明顯不是姓名的詞彙（可視情況擴充）
        if len(name) > 5 or '電話' in name:
            name = None
        return role, name
    # 只有職稱出現
    m = re.search(rf'({"|".join(ROLE_WORDS)})[：:]?', text)
    if m:
        return m.group(1), None
    return None, None

# ===== 發文字號抽取（來文）=====
# 支援：發文字號：XXX、文號：XXX、○字第XXXX號、○○府授人字第…號 等
DOCNO_CANDIDATE_RE = re.compile(
    r'(?:發文字號|文號)[：:]\s*([^\n\r，。、]{4,30})'
    r'|([一二三四五六七八九十○零〇台臺北新高桃竹苗中彰投雲嘉南高屏宜花東金馬\w]{1,6}字第[^\s，。、]{3,20}號)'
)

def extract_doc_no(text: str) -> str | None:
    for m in DOCNO_CANDIDATE_RE.finditer(text):
        g = next((x for x in m.groups() if x), None)
        if g:
            # 清理尾端標點
            return g.strip().strip('，。；；、 ')
    return None

def today_fields():
    t = date.today()
    return {
        "today_yyyy": f"{t.year}",
        "today_mm": f"{t.month:02d}",
        "today_dd": f"{t.day:02d}",
        "today_iso": t.isoformat(),
        "today_yyyymmdd": t.strftime("%Y%m%d"),
    }
