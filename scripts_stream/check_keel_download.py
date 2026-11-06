import urllib.request
import re

url = 'https://sci2s.ugr.es/keel/dataset.php?cod=140' # ecoli-0_vs_1
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    html = urllib.request.urlopen(req, timeout=15).read().decode('utf-8', errors='ignore')
    links = re.findall(r'href=[\"\']([^\"\']+\.(?:dat|zip|rar))[\"\']', html)
    print("Download links found:", links)
except Exception as e:
    print("Error:", e)

