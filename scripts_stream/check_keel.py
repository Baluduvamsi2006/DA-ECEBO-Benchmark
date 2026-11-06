import urllib.request
import re

url = 'https://sci2s.ugr.es/keel/imbalanced.php'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    html = urllib.request.urlopen(req, timeout=15).read().decode('utf-8', errors='ignore')
    matches = re.findall(r'href=[\"\']dataset\.php\?cod=(\d+)[\"\']>([^<]+)</a>', html)
    print(f"Found {len(matches)} datasets on KEEL imbalanced page.")
    for cod, name in matches:
        print(f"{cod}\t{name}")
except Exception as e:
    print("Error:", e)

