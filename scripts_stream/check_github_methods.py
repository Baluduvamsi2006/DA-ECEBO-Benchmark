import urllib.request
import urllib.parse
import json

queries = [
    ('NCLO-SMOTE', 'NCLO-SMOTE'),
    ('MLOS', 'MLOS oversampling'),
    ('HS-SMOTE', 'HS-SMOTE'),
    ('DGS', 'DGS oversampling'),
    ('ENU', 'ENU oversampling'),
]

for name, q in queries:
    url = f'https://api.github.com/search/repositories?q={urllib.parse.quote(q)}'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            items = [(item['full_name'], item.get('description', '')) for item in data.get('items', [])[:3]]
            print(f'=== {name} ===')
            for fn, desc in items:
                print(f'  {fn}: {desc}')
            if not items:
                print('  No relevant repos found.')
    except Exception as e:
        print(f'{name}: error {e}')

