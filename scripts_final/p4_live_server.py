import os
import sys
import json
import time
import shutil
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = 8088
PROGRESS_FILE = 'results_final/progress.json'
DONE_LOG = 'results_final/P4_done_log.csv'
MANIFEST_FILE = 'results_final/P4_manifest.csv'

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Phase 4 Final Benchmark Monitor</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    @keyframes pulse-dot { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.95); } }
    .pulse-dot { animation: pulse-dot 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 antialiased p-4 sm:p-6 min-h-screen">
  <div class="max-w-7xl mx-auto space-y-6">
    
    <!-- Top Header -->
    <header class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col md:flex-row md:items-center justify-between gap-4">
      <div>
        <div class="flex items-center gap-3">
          <span id="statusDot" class="w-3.5 h-3.5 rounded-full bg-slate-500 pulse-dot"></span>
          <h1 class="text-xl sm:text-2xl font-black tracking-tight text-white">DA-ECEBO Phase 4 Final Benchmark</h1>
          <span id="statusBadge" class="px-2.5 py-0.5 rounded-full text-xs font-bold bg-slate-800 text-slate-300 border border-slate-700">CHECKING</span>
        </div>
        <p class="text-slate-400 text-xs sm:text-sm mt-1">Unified Multi-day Benchmark, Streaming Concept Drift & Rigorous Empirical Audit</p>
      </div>
      <div class="flex items-center gap-6 text-right">
        <div>
          <div class="text-[10px] text-slate-500 uppercase tracking-wider font-bold">Elapsed Time</div>
          <div id="elapsedTime" class="text-sm font-mono text-slate-200 font-semibold">0h 0m</div>
        </div>
        <div>
          <div class="text-[10px] text-slate-500 uppercase tracking-wider font-bold">ETA Remaining</div>
          <div id="etaTime" class="text-sm font-mono text-amber-400 font-semibold">Estimating...</div>
        </div>
        <div>
          <div class="text-[10px] text-slate-500 uppercase tracking-wider font-bold">Disk Free</div>
          <div id="diskFree" class="text-sm font-mono text-emerald-400 font-semibold">-- GB</div>
        </div>
      </div>
    </header>

    <!-- Overall Progress Banner -->
    <div class="bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border border-indigo-500/30 rounded-2xl p-6 shadow-2xl relative overflow-hidden">
      <div class="flex justify-between items-baseline mb-2">
        <span class="text-xs font-bold uppercase tracking-wider text-indigo-400">Master Progress</span>
        <span id="overallPercent" class="text-3xl sm:text-4xl font-black text-indigo-200">0.0%</span>
      </div>
      <div class="w-full bg-slate-800 rounded-full h-4 overflow-hidden p-0.5 border border-slate-700/60">
        <div id="overallBar" class="bg-gradient-to-r from-indigo-500 to-emerald-400 h-full rounded-full transition-all duration-500" style="width: 0%"></div>
      </div>
      <div class="flex justify-between items-center mt-3 text-xs text-slate-400">
        <div>Cells Done: <span id="donePlanned" class="font-bold text-white">0 / 0</span></div>
        <div>Current Task: <span id="currentTask" class="font-mono text-indigo-300 font-medium">Initializing...</span></div>
      </div>
    </div>

    <!-- Grid: Parts & Datasets Progress -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      
      <!-- Progress By Part (A..F) -->
      <div class="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
        <h2 class="text-sm font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
          <svg class="w-4 h-4 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path></svg>
          Progress By Benchmark Part
        </h2>
        <div id="partsContainer" class="space-y-3 text-xs">
          <!-- Dynamically populated -->
        </div>
      </div>

      <!-- Progress By Dataset -->
      <div class="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
        <h2 class="text-sm font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
          <svg class="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7v10c0 2 1 3 3 3h10c2 0 3-1 3-3V7c0-2-1-3-3-3H7C5 4 4 5 4 7z"></path></svg>
          Progress By Dataset
        </h2>
        <div id="datasetsContainer" class="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs max-h-72 overflow-y-auto pr-1">
          <!-- Dynamically populated -->
        </div>
      </div>

    </div>

    <!-- Tables: Last 20 Finished & Last 20 Errors -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      
      <!-- Last 20 Finished Cells -->
      <div class="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-3">
        <h2 class="text-sm font-bold uppercase tracking-wider text-slate-300">Last 20 Finished Cells</h2>
        <div class="overflow-x-auto max-h-80 overflow-y-auto pr-1">
          <table class="w-full text-left text-xs font-mono">
            <thead class="bg-slate-950 text-slate-400 sticky top-0 border-b border-slate-800">
              <tr>
                <th class="p-2">Dataset</th>
                <th class="p-2">Classifier</th>
                <th class="p-2">Method</th>
                <th class="p-2">Seed</th>
                <th class="p-2 text-right">G-mean</th>
              </tr>
            </thead>
            <tbody id="finishedTable" class="divide-y divide-slate-800/60 text-slate-300">
              <tr><td colspan="5" class="p-3 text-center text-slate-500">No completed cells logged yet</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Last 20 Errors -->
      <div class="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-3">
        <h2 class="text-sm font-bold uppercase tracking-wider text-rose-400">Last 20 Errors / Timeouts</h2>
        <div class="overflow-x-auto max-h-80 overflow-y-auto pr-1">
          <table class="w-full text-left text-xs font-mono">
            <thead class="bg-slate-950 text-slate-400 sticky top-0 border-b border-slate-800">
              <tr>
                <th class="p-2">Target</th>
                <th class="p-2">Error Message</th>
                <th class="p-2 text-right">Time</th>
              </tr>
            </thead>
            <tbody id="errorsTable" class="divide-y divide-slate-800/60 text-slate-300">
              <tr><td colspan="3" class="p-3 text-center text-slate-500">Zero errors recorded</td></tr>
            </tbody>
          </table>
        </div>
      </div>

    </div>

    <!-- Active Process Log Stream (Last 30 Lines) -->
    <div class="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-2">
      <div class="flex justify-between items-center text-xs">
        <span class="font-bold uppercase tracking-wider text-slate-300">Live Execution Log (Last 30 Lines)</span>
        <span class="text-slate-500">Auto-refreshing every 10s</span>
      </div>
      <pre id="logStream" class="p-3.5 bg-slate-950 rounded-xl border border-slate-800 font-mono text-[11px] text-slate-300 overflow-x-auto max-h-60 overflow-y-auto leading-relaxed">Waiting for active run logs...</pre>
    </div>

  </div>

  <script>
    async function updateDashboard() {
      try {
        const res = await fetch('/api/progress');
        if (!res.ok) return;
        const data = await res.json();
        
        // Status indicator
        const isRunning = data.status === 'RUNNING';
        const dot = document.getElementById('statusDot');
        const badge = document.getElementById('statusBadge');
        if (isRunning) {
          dot.className = 'w-3.5 h-3.5 rounded-full bg-emerald-500 pulse-dot';
          badge.className = 'px-2.5 py-0.5 rounded-full text-xs font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30';
          badge.innerText = 'RUNNING';
        } else {
          dot.className = 'w-3.5 h-3.5 rounded-full bg-amber-500';
          badge.className = 'px-2.5 py-0.5 rounded-full text-xs font-bold bg-amber-500/10 text-amber-400 border border-amber-500/30';
          badge.innerText = 'STOPPED / IDLE';
        }

        // Header metrics
        document.getElementById('elapsedTime').innerText = data.elapsed_formatted || '0h 0m';
        document.getElementById('etaTime').innerText = data.eta_hours ? data.eta_hours.toFixed(1) + ' hrs' : 'Calculating...';
        document.getElementById('diskFree').innerText = (data.disk_free_gb || 0).toFixed(1) + ' GB';

        // Overall progress
        const pct = (data.overall_percent || 0).toFixed(1);
        document.getElementById('overallPercent').innerText = pct + '%';
        document.getElementById('overallBar').style.width = pct + '%';
        document.getElementById('donePlanned').innerText = `${data.done_cells || 0} / ${data.planned_cells || 0}`;
        document.getElementById('currentTask').innerText = data.current_task || 'Idle';

        // Parts progress
        const partsCont = document.getElementById('partsContainer');
        if (data.per_part) {
          partsCont.innerHTML = Object.entries(data.per_part).map(([pName, pData]) => `
            <div>
              <div class="flex justify-between mb-1">
                <span class="font-semibold text-slate-300">${pName}</span>
                <span class="text-slate-400 font-mono">${pData.done || 0} / ${pData.planned || 0} (${(pData.percent || 0).toFixed(1)}%)</span>
              </div>
              <div class="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                <div class="bg-indigo-500 h-full rounded-full" style="width: ${pData.percent || 0}%"></div>
              </div>
            </div>
          `).join('');
        }

        // Datasets progress
        const dsetsCont = document.getElementById('datasetsContainer');
        if (data.per_dataset) {
          dsetsCont.innerHTML = Object.entries(data.per_dataset).map(([dName, dData]) => `
            <div class="p-2.5 bg-slate-950/60 border border-slate-800 rounded-lg">
              <div class="flex justify-between font-mono text-[11px] mb-1">
                <span class="font-bold text-slate-200 truncate">${dName}</span>
                <span class="text-slate-400">${dData.done}/${dData.planned}</span>
              </div>
              <div class="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                <div class="bg-emerald-500 h-full rounded-full" style="width: ${dData.percent || 0}%"></div>
              </div>
            </div>
          `).join('');
        }

        // Last 20 finished
        const finTable = document.getElementById('finishedTable');
        if (data.last_20_finished && data.last_20_finished.length > 0) {
          finTable.innerHTML = data.last_20_finished.map(row => `
            <tr class="hover:bg-slate-800/30">
              <td class="p-2">${row.dataset || '-'}</td>
              <td class="p-2 text-indigo-400">${row.classifier || '-'}</td>
              <td class="p-2">${row.method || '-'}</td>
              <td class="p-2 text-slate-400">${row.seed || '-'}</td>
              <td class="p-2 text-right font-bold text-emerald-400">${row.Macro_Gmean != null ? Number(row.Macro_Gmean).toFixed(4) : '-'}</td>
            </tr>
          `).join('');
        }

        // Errors
        const errTable = document.getElementById('errorsTable');
        if (data.errors && data.errors.length > 0) {
          errTable.innerHTML = data.errors.slice(-20).map(err => `
            <tr class="hover:bg-slate-800/30 text-rose-300">
              <td class="p-2">${err.target || '-'}</td>
              <td class="p-2 text-rose-400 break-all">${err.error || '-'}</td>
              <td class="p-2 text-right text-slate-400">${err.timestamp || '-'}</td>
            </tr>
          `).join('');
        }

        // Log tail
        if (data.log_tail && data.log_tail.length > 0) {
          document.getElementById('logStream').innerText = data.log_tail.join('\\n');
        }

      } catch (err) {
        console.error('Failed to sync progress:', err);
      }
    }

    setInterval(updateDashboard, 10000);
    window.addEventListener('DOMContentLoaded', updateDashboard);
  </script>
</body>
</html>
"""

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
        elif self.path == '/api/progress':
            data = {}
            if os.path.exists(PROGRESS_FILE):
                try:
                    with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception as e:
                    data = {'error': str(e)}
            
            # Check if running (modified in last 5 minutes)
            is_running = False
            if os.path.exists(PROGRESS_FILE):
                try:
                    mtime = os.path.getmtime(PROGRESS_FILE)
                    if time.time() - mtime < 300:
                        is_running = True
                except Exception:
                    pass
            data['status'] = 'RUNNING' if is_running else 'STOPPED'

            # Calculate disk free
            try:
                total, used, free = shutil.disk_usage('.')
                data['disk_free_gb'] = free / (1024**3)
            except Exception:
                data['disk_free_gb'] = 0.0

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run_server():
    server = HTTPServer(('0.0.0.0', PORT), RequestHandler)
    print(f"Phase 4 Live Server active at http://127.0.0.1:{PORT}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == '__main__':
    run_server()
