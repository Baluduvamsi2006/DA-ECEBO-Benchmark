"""
Live Status Server for DA-ECEBO Benchmark Dashboard
===================================================
Serves real-time progress data from results_stream/P2_raw.csv,
results_stream/P2_blocks.csv, and active task execution logs.

Endpoints:
  - GET /            -> Serves results_stream/live_dashboard.html
  - GET /api/status  -> Returns JSON with live benchmark progress metrics
"""
import os
import sys
import json
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import pandas as pd

PORT = 8088
LOG_FILE = r'C:\Users\VAMSI\.gemini\antigravity\brain\306ec904-0d66-4da1-9dfc-67760191c62c\.system_generated\tasks\task-6620.log'
HTML_FILE = 'results_stream/live_dashboard.html'
RAW_CSV = 'results_stream/P2_raw.csv'
BLOCKS_CSV = 'results_stream/P2_blocks.csv'

ALL_DATASETS = [
    ('ecoli2', 'row order'),
    ('yeast3', 'row order'),
    ('synth_sea_p5', 'synthetic'),
    ('synth_hyperplane_p5', 'synthetic'),
    ('synth_random_rbf_p5', 'synthetic'),
    ('electricity_ir100', 'subsampled'),
    ('electricity_ir20', 'subsampled'),
    ('creditcard', 'Natural (75/25)')
]

def get_live_data():
    total_raw = 0
    total_blocks = 0
    dataset_counts = {}
    recent_rows = []
    
    if os.path.exists(RAW_CSV):
        try:
            df_raw = pd.read_csv(RAW_CSV, keep_default_na=False)
            total_raw = len(df_raw)
            dataset_counts = df_raw.groupby('dataset')['method'].count().to_dict()
            
            # Last 10 rows
            tail_df = df_raw.tail(10)
            for _, r in tail_df.iterrows():
                recent_rows.append({
                    'dataset': str(r.get('dataset', '')),
                    'classifier': str(r.get('classifier', '')),
                    'method': str(r.get('method', '')),
                    'seed': int(r.get('seed', 0)) if pd.notnull(r.get('seed')) else 0,
                    'Macro_Gmean': float(r.get('Macro-Gmean', 0.0)) if pd.notnull(r.get('Macro-Gmean')) else 0.0,
                    'Minority_F1': float(r.get('Minority-F1', 0.0)) if pd.notnull(r.get('Minority-F1')) else 0.0,
                    'AUC': float(r.get('AUC', 0.0)) if pd.notnull(r.get('AUC')) else 0.0
                })
        except Exception as e:
            pass
            
    if os.path.exists(BLOCKS_CSV):
        try:
            df_b = pd.read_csv(BLOCKS_CSV, keep_default_na=False)
            total_blocks = len(df_b)
        except Exception:
            pass

    # Dataset breakdown
    dsets_info = []
    for dname, dtype in ALL_DATASETS:
        cnt = dataset_counts.get(dname, 0)
        pct = round((cnt / 520.0) * 100.0, 1)
        if cnt >= 520:
            status = 'Done'
        elif cnt > 0:
            status = 'In Progress'
        else:
            status = 'Queued'
        dsets_info.append({
            'name': dname,
            'type': dtype,
            'completed': cnt,
            'total': 520,
            'percent': min(100.0, pct),
            'status': status
        })
        
    # Read last 6 lines of log
    log_lines = []
    active_log = 'results_stream/eval.log' if os.path.exists('results_stream/eval.log') else LOG_FILE
    if os.path.exists(active_log):
        try:
            with open(active_log, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                valid = [l.strip() for l in lines if l.strip() and not l.startswith('warnings.') and not l.startswith('C:\\')]
                log_lines = valid[-6:]
        except Exception:
            pass

    pct_total = round((total_raw / 4160.0) * 100.0, 2)
    overall_project = round(40.0 + (pct_total * 0.60), 1) # Part 1 + Part 3 are complete
    remaining_cells = max(0, 4160 - total_raw)

    # Dynamic ETA calculation based on dataset workloads
    # Remaining cells in current dataset (electricity_ir20): ~3.5s per cell
    # Remaining cells in creditcard (520 cells): ~12.5s per cell (due to 426k rows CART + MLOS/ASUWO sampling)
    cc_done = dataset_counts.get('creditcard', 0)
    cc_remaining = max(0, 520 - cc_done)
    non_cc_remaining = remaining_cells - cc_remaining
    est_seconds_remaining = int((non_cc_remaining * 3.5) + (cc_remaining * 12.5))

    hrs = est_seconds_remaining // 3600
    mins = (est_seconds_remaining % 3600) // 60
    secs = est_seconds_remaining % 60
    eta_formatted = f"{hrs} hrs {mins} min {secs} sec"
    
    target_timestamp = datetime.fromtimestamp(time.time() + est_seconds_remaining).strftime('%I:%M:%S %p')

    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_raw': total_raw,
        'total_expected': 4160,
        'remaining_cells': remaining_cells,
        'percent_raw': pct_total,
        'total_blocks': total_blocks,
        'overall_project': min(100.0, overall_project),
        'est_seconds_remaining': est_seconds_remaining,
        'eta_formatted': eta_formatted,
        'target_time': target_timestamp,
        'datasets': dsets_info,
        'recent_evaluations': recent_rows,
        'log_lines': log_lines
    }

class LiveHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/status':
            data = get_live_data()
            content = json.dumps(data).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        else:
            # Serve dashboard HTML
            path = HTML_FILE if os.path.exists(HTML_FILE) else 'progress_dashboard.html'
            try:
                with open(path, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(f"Not found: {e}".encode('utf-8'))

    def log_message(self, format, *args):
        # Silence console log spam
        return

def main():
    server = HTTPServer(('127.0.0.1', PORT), LiveHandler)
    print(f"Live dashboard server running on http://127.0.0.1:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == '__main__':
    main()

