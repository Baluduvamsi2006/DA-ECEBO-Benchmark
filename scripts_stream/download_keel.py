"""
Download KEEL imbalanced datasets
"""
import urllib.request
import zipfile
import io
import os
import re

os.makedirs('datasets_keel', exist_ok=True)

# List of all candidate KEEL datasets
candidate_names = [
    # Table 1 common KEEL 48 benchmark datasets
    'ecoli-0_vs_1', 'wisconsin', 'pima', 'iris0', 'glass0', 'yeast1', 'haberman',
    'vehicle2', 'vehicle1', 'vehicle3', 'vehicle0', 'ecoli1', 'new-thyroid1',
    'new-thyroid2', 'ecoli2', 'segment0', 'glass6', 'yeast3', 'ecoli3',
    'page-blocks0', 'yeast-2_vs_4', 'yeast-0-5-6-7-9_vs_4', 'vowel0',
    'glass-0-1-6_vs_2', 'glass2', 'shuttle-c0-vs-c4', 'yeast-1_vs_7', 'glass4',
    'ecoli4', 'page-blocks-1-3_vs_4', 'abalone9-18', 'glass-0-1-6_vs_5',
    'shuttle-c2-vs-c4', 'yeast-1-4-5-8_vs_7', 'glass5', 'yeast-2_vs_8', 'yeast4',
    'yeast-1-2-8-9_vs_7', 'yeast5', 'ecoli-0-1-3-7_vs_2-6', 'yeast6', 'abalone19',
    'dermatology-6', 'flare-F', 'car-good', 'car-vgood', 'winequality-red-4',
    'winequality-white-9_vs_4', 'winequality-red-8_vs_6', 'winequality-white-3_vs_7',
    'winequality-red-8_vs_6-7', 'winequality-white-3-9_vs_5', 'winequality-red-3_vs_5'
]

obtained = []
failed = []

for name in candidate_names:
    dat_path = os.path.join('datasets_keel', f'{name}.dat')
    csv_path = os.path.join('datasets_keel', f'{name}.csv')
    
    # Check if already present in local datasets/
    local_csv = os.path.join('datasets', f'{name}.csv')
    if os.path.exists(local_csv):
        obtained.append((name, 'local datasets/ folder'))
        continue
        
    if os.path.exists(csv_path) or os.path.exists(dat_path):
        obtained.append((name, 'already cached in datasets_keel/'))
        continue
        
    url = f'https://sci2s.ugr.es/keel/dataset/data/imbalanced/{name}.zip'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req, timeout=12)
        content = resp.read()
        z = zipfile.ZipFile(io.BytesIO(content))
        for member in z.namelist():
            if member.endswith('.dat'):
                with open(dat_path, 'wb') as fp:
                    fp.write(z.read(member))
                break
        obtained.append((name, f'downloaded from {url}'))
        print(f"Obtained: {name}")
    except Exception as e:
        failed.append((name, str(e)))
        print(f"Failed {name}: {e}")

print(f"\nTotal obtained: {len(obtained)}")
print(f"Total failed: {len(failed)}")

