"""
Poker Tournament Analysis Tool - Web GUI
A browser-based interface for poker tournament analysis.
Built with Flask for Milestone 1 preview.
"""

from flask import Flask, render_template_string, request, jsonify
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.expanduser('~/poker-analysis'))

from src.equity.calculator import hand_vs_range_equity, hand_vs_hand_equity
from src.icm.calculator import icm_equity, bubble_factor
from src.mtt.tournament import BlindLevel, categorize_stack, push_fold_range
from src.analysis.analyzer import TournamentAnalyzer

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Poker Tournament Analyzer</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0a0e17;
    color: #e0e6ed;
    min-height: 100vh;
}
.header {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
    border-bottom: 2px solid #2d7d46;
    padding: 20px 30px;
    text-align: center;
}
.header h1 {
    font-size: 28px;
    color: #4ade80;
    letter-spacing: 1px;
}
.header p {
    color: #8892a4;
    margin-top: 5px;
    font-size: 14px;
}
.container {
    max-width: 1100px;
    margin: 0 auto;
    padding: 25px;
}
.tabs {
    display: flex;
    gap: 8px;
    margin-bottom: 25px;
    flex-wrap: wrap;
}
.tab {
    padding: 12px 20px;
    background: #151b28;
    border: 1px solid #2a3142;
    border-radius: 8px;
    color: #8892a4;
    cursor: pointer;
    font-size: 14px;
    font-weight: 500;
    transition: all 0.2s;
}
.tab:hover { border-color: #4ade80; color: #c0c8d4; }
.tab.active {
    background: #1a2d1f;
    border-color: #4ade80;
    color: #4ade80;
}
.panel {
    display: none;
    background: #111827;
    border: 1px solid #2a3142;
    border-radius: 12px;
    padding: 30px;
}
.panel.active { display: block; }
.panel h2 {
    color: #4ade80;
    font-size: 22px;
    margin-bottom: 8px;
}
.panel .desc {
    color: #6b7688;
    margin-bottom: 25px;
    font-size: 14px;
}
.form-row {
    display: flex;
    gap: 20px;
    margin-bottom: 18px;
    flex-wrap: wrap;
}
.form-group {
    flex: 1;
    min-width: 200px;
}
.form-group label {
    display: block;
    color: #8892a4;
    font-size: 13px;
    margin-bottom: 6px;
    font-weight: 500;
}
.form-group input, .form-group select {
    width: 100%;
    padding: 10px 14px;
    background: #0a0e17;
    border: 1px solid #2a3142;
    border-radius: 6px;
    color: #e0e6ed;
    font-size: 14px;
    font-family: 'Consolas', 'Courier New', monospace;
}
.form-group input:focus, .form-group select:focus {
    outline: none;
    border-color: #4ade80;
    box-shadow: 0 0 0 2px rgba(74, 222, 128, 0.15);
}
.form-group .hint {
    color: #4a5568;
    font-size: 11px;
    margin-top: 4px;
}
.btn {
    padding: 12px 30px;
    background: linear-gradient(135deg, #2d7d46 0%, #22633a 100%);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    margin-top: 10px;
}
.btn:hover {
    background: linear-gradient(135deg, #37955a 0%, #2d7d46 100%);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(74, 222, 128, 0.2);
}
.btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
}
.results {
    margin-top: 25px;
    display: none;
}
.results.show { display: block; }
.result-card {
    background: #0d1117;
    border: 1px solid #2a3142;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 15px;
}
.result-card h3 {
    color: #4ade80;
    font-size: 16px;
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 1px solid #1e2736;
}
.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 15px;
}
.stat-item {
    text-align: center;
    padding: 15px;
    background: #151b28;
    border-radius: 8px;
}
.stat-value {
    font-size: 28px;
    font-weight: 700;
    color: #4ade80;
    font-family: 'Consolas', monospace;
}
.stat-value.fold { color: #f87171; }
.stat-value.call { color: #4ade80; }
.stat-label {
    font-size: 12px;
    color: #6b7688;
    margin-top: 5px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.decision-box {
    text-align: center;
    padding: 20px;
    border-radius: 10px;
    margin-top: 15px;
}
.decision-box.fold {
    background: linear-gradient(135deg, #3b1111 0%, #1a0808 100%);
    border: 2px solid #f87171;
}
.decision-box.call {
    background: linear-gradient(135deg, #11331a 0%, #081a0c 100%);
    border: 2px solid #4ade80;
}
.decision-text {
    font-size: 36px;
    font-weight: 800;
    letter-spacing: 3px;
}
.decision-text.fold { color: #f87171; }
.decision-text.call { color: #4ade80; }
.decision-reason {
    color: #8892a4;
    margin-top: 8px;
    font-size: 14px;
}
table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
}
th {
    background: #151b28;
    color: #4ade80;
    padding: 12px 15px;
    text-align: left;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
td {
    padding: 10px 15px;
    border-bottom: 1px solid #1e2736;
    font-size: 14px;
}
tr:hover td { background: #151b2866; }
.zone-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
}
.zone-green { background: #166534; color: #4ade80; }
.zone-yellow { background: #854d0e; color: #fbbf24; }
.zone-orange { background: #9a3412; color: #fb923c; }
.zone-red { background: #991b1b; color: #f87171; }
.zone-danger { background: #7f1d1d; color: #fca5a5; }
.zone-dead { background: #44403c; color: #a8a29e; }
.loading {
    display: none;
    text-align: center;
    padding: 30px;
    color: #4ade80;
}
.loading.show { display: block; }
.spinner {
    display: inline-block;
    width: 30px;
    height: 30px;
    border: 3px solid #2a3142;
    border-top-color: #4ade80;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.equity-bar {
    height: 24px;
    background: #1e2736;
    border-radius: 12px;
    overflow: hidden;
    margin: 10px 0;
}
.equity-fill {
    height: 100%;
    border-radius: 12px;
    transition: width 0.5s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 700;
    color: #fff;
}
.footer {
    text-align: center;
    padding: 20px;
    color: #3a4255;
    font-size: 12px;
    margin-top: 30px;
}
</style>
</head>
<body>

<div class="header">
    <h1>♠ POKER TOURNAMENT ANALYZER ♠</h1>
    <p>Offline GTO & ICM Analysis Engine — Milestone 1 Preview</p>
</div>

<div class="container">
    <div class="tabs">
        <div class="tab active" onclick="showTab('equity')">♠ Hand vs Hand</div>
        <div class="tab" onclick="showTab('range')">♦ Hand vs Range</div>
        <div class="tab" onclick="showTab('icm')">♣ ICM Calculator</div>
        <div class="tab" onclick="showTab('spot')">♥ Spot Analysis</div>
        <div class="tab" onclick="showTab('pushfold')">↑ Push/Fold</div>
        <div class="tab" onclick="showTab('finaltable')">🏆 Final Table</div>
    </div>

    <!-- HAND VS HAND -->
    <div id="panel-equity" class="panel active">
        <h2>Hand vs Hand Equity</h2>
        <p class="desc">Calculate equity between two specific hands</p>
        <div class="form-row">
            <div class="form-group">
                <label>Your Hand</label>
                <input type="text" id="eq-hand1" placeholder="e.g. AhKs" value="AhKs">
                <div class="hint">Format: RankSuit (e.g. Ah = Ace of hearts)</div>
            </div>
            <div class="form-group">
                <label>Opponent's Hand</label>
                <input type="text" id="eq-hand2" placeholder="e.g. QdQc" value="QdQc">
            </div>
        </div>
        <div class="form-row">
            <div class="form-group">
                <label>Board (optional)</label>
                <input type="text" id="eq-board" placeholder="e.g. Jh Ts 4c (leave blank for preflop)">
            </div>
        </div>
        <button class="btn" onclick="calcEquity()">Calculate Equity</button>
        <div id="eq-loading" class="loading"><div class="spinner"></div><p>Running Monte Carlo simulation...</p></div>
        <div id="eq-results" class="results"></div>
    </div>

    <!-- HAND VS RANGE -->
    <div id="panel-range" class="panel">
        <h2>Hand vs Range Equity</h2>
        <p class="desc">Calculate equity of your hand against a range of hands</p>
        <div class="form-row">
            <div class="form-group">
                <label>Your Hand</label>
                <input type="text" id="rng-hand" placeholder="e.g. AhKs" value="AhKs">
            </div>
            <div class="form-group">
                <label>Villain's Range</label>
                <input type="text" id="rng-range" placeholder="e.g. JJ, TT, AQs, KQs" value="JJ, TT, AQs, KQs">
                <div class="hint">Comma-separated: pairs, suited (s), offsuit (o)</div>
            </div>
        </div>
        <div class="form-row">
            <div class="form-group">
                <label>Board (optional)</label>
                <input type="text" id="rng-board" placeholder="Leave blank for preflop">
            </div>
        </div>
        <button class="btn" onclick="calcRange()">Calculate Range Equity</button>
        <div id="rng-loading" class="loading"><div class="spinner"></div><p>Simulating against range...</p></div>
        <div id="rng-results" class="results"></div>
    </div>

    <!-- ICM CALCULATOR -->
    <div id="panel-icm" class="panel">
        <h2>ICM Calculator</h2>
        <p class="desc">Calculate Independent Chip Model equity for tournament payouts</p>
        <div class="form-row">
            <div class="form-group">
                <label>Player Names (comma-separated)</label>
                <input type="text" id="icm-names" placeholder="e.g. Hero, Villain1, Villain2" value="Hero, Player2, Player3, Player4, Player5">
            </div>
        </div>
        <div class="form-row">
            <div class="form-group">
                <label>Chip Stacks (comma-separated)</label>
                <input type="text" id="icm-stacks" placeholder="e.g. 25000, 40000, 15000" value="25000, 40000, 15000, 30000, 10000">
            </div>
            <div class="form-group">
                <label>Payouts (comma-separated)</label>
                <input type="text" id="icm-payouts" placeholder="e.g. 50, 30, 20" value="50, 30, 20">
            </div>
        </div>
        <button class="btn" onclick="calcICM()">Calculate ICM</button>
        <div id="icm-results" class="results"></div>
    </div>

    <!-- TOURNAMENT SPOT ANALYSIS -->
    <div id="panel-spot" class="panel">
        <h2>Tournament Spot Analysis</h2>
        <p class="desc">Complete tournament decision: equity + ICM + stack depth analysis</p>
        <div class="form-row">
            <div class="form-group">
                <label>Your Hand</label>
                <input type="text" id="spot-hand" placeholder="e.g. AhKs" value="AhKs">
            </div>
            <div class="form-group">
                <label>Villain's Range</label>
                <input type="text" id="spot-range" placeholder="e.g. JJ, TT, 99, AQs" value="JJ, TT, 99, AQs, KQs">
            </div>
            <div class="form-group">
                <label>Board (optional)</label>
                <input type="text" id="spot-board" placeholder="Leave blank for preflop">
            </div>
        </div>
        <div class="form-row">
            <div class="form-group">
                <label>Your Stack</label>
                <input type="number" id="spot-hero" placeholder="25000" value="25000">
            </div>
            <div class="form-group">
                <label>Villain's Stack</label>
                <input type="number" id="spot-villain" placeholder="40000" value="40000">
            </div>
            <div class="form-group">
                <label>Blinds (SB/BB)</label>
                <input type="text" id="spot-blinds" placeholder="e.g. 500/1000" value="500/1000">
            </div>
        </div>
        <div class="form-row">
            <div class="form-group">
                <label>All Stacks (comma-separated, include yours)</label>
                <input type="text" id="spot-stacks" placeholder="e.g. 25000,40000,15000,30000" value="25000,40000,15000,30000,10000,20000">
            </div>
            <div class="form-group">
                <label>Payouts (comma-separated)</label>
                <input type="text" id="spot-payouts" placeholder="e.g. 50,30,20" value="50,30,20">
            </div>
        </div>
        <button class="btn" onclick="calcSpot()">Analyze Spot</button>
        <div id="spot-loading" class="loading"><div class="spinner"></div><p>Running full tournament analysis...</p></div>
        <div id="spot-results" class="results"></div>
    </div>

    <!-- PUSH/FOLD -->
    <div id="panel-pushfold" class="panel">
        <h2>Push/Fold Helper</h2>
        <p class="desc">Get push/fold ranges based on your stack depth and position</p>
        <div class="form-row">
            <div class="form-group">
                <label>Your Stack</label>
                <input type="number" id="pf-stack" placeholder="12000" value="12000">
            </div>
            <div class="form-group">
                <label>Big Blind</label>
                <input type="number" id="pf-bb" placeholder="1000" value="1000">
            </div>
            <div class="form-group">
                <label>Position</label>
                <select id="pf-pos">
                    <option value="btn">Button (BTN)</option>
                    <option value="co">Cutoff (CO)</option>
                    <option value="mp">Middle (MP)</option>
                    <option value="ep">Early (EP)</option>
                    <option value="sb">Small Blind (SB)</option>
                </select>
            </div>
        </div>
        <button class="btn" onclick="calcPushFold()">Get Push/Fold Range</button>
        <div id="pf-results" class="results"></div>
    </div>

    <!-- FINAL TABLE -->
    <div id="panel-finaltable" class="panel">
        <h2>Final Table Report</h2>
        <p class="desc">Full final table ICM analysis with deal equity</p>
        <div class="form-row">
            <div class="form-group">
                <label>Player Names (comma-separated)</label>
                <input type="text" id="ft-names" value="Hero, Villain1, Villain2, Villain3, Villain4">
            </div>
        </div>
        <div class="form-row">
            <div class="form-group">
                <label>Chip Stacks (comma-separated)</label>
                <input type="text" id="ft-stacks" value="45000, 35000, 25000, 15000, 10000">
            </div>
            <div class="form-group">
                <label>Payouts (comma-separated)</label>
                <input type="text" id="ft-payouts" value="50000, 30000, 20000, 15000, 10000">
            </div>
        </div>
        <div class="form-row">
            <div class="form-group">
                <label>Blinds (SB/BB)</label>
                <input type="text" id="ft-blinds" value="1000/2000">
            </div>
        </div>
        <button class="btn" onclick="calcFinalTable()">Generate Report</button>
        <div id="ft-loading" class="loading"><div class="spinner"></div><p>Generating final table report...</p></div>
        <div id="ft-results" class="results"></div>
    </div>
</div>

<div class="footer">
    Poker Tournament Analysis System v1.0 — Offline Engine — All calculations run locally
</div>

<script>
function showTab(name) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    event.target.classList.add('active');
    document.getElementById('panel-' + name).classList.add('active');
}

function parseCards(s) {
    return s.trim().split(/\\s+/).filter(c => c.length > 0);
}

async function apiCall(endpoint, data) {
    const resp = await fetch('/api/' + endpoint, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
    });
    return await resp.json();
}

function getZoneClass(zone) {
    const z = zone.toLowerCase();
    if (z === 'green') return 'zone-green';
    if (z === 'yellow') return 'zone-yellow';
    if (z === 'orange') return 'zone-orange';
    if (z === 'red') return 'zone-red';
    if (z === 'danger') return 'zone-danger';
    return 'zone-dead';
}

function equityBar(pct, color) {
    const bg = color || (pct > 50 ? '#4ade80' : '#f87171');
    return `<div class="equity-bar"><div class="equity-fill" style="width:${pct}%;background:${bg}">${pct.toFixed(1)}%</div></div>`;
}

async function calcEquity() {
    const hand1 = document.getElementById('eq-hand1').value.trim();
    const hand2 = document.getElementById('eq-hand2').value.trim();
    const board = document.getElementById('eq-board').value.trim();
    document.getElementById('eq-loading').classList.add('show');
    document.getElementById('eq-results').classList.remove('show');

    const data = await apiCall('equity', {hand1, hand2, board});
    document.getElementById('eq-loading').classList.remove('show');

    if (data.error) {
        document.getElementById('eq-results').innerHTML = `<div class="result-card"><p style="color:#f87171">Error: ${data.error}</p></div>`;
    } else {
        document.getElementById('eq-results').innerHTML = `
            <div class="result-card">
                <h3>Equity Results</h3>
                <div style="margin-bottom:15px">
                    <div style="display:flex;justify-content:space-between;margin-bottom:5px">
                        <span><strong>${hand1}</strong></span>
                        <span style="color:#4ade80;font-weight:700">${data.equity1.toFixed(1)}%</span>
                    </div>
                    ${equityBar(data.equity1, '#4ade80')}
                </div>
                <div>
                    <div style="display:flex;justify-content:space-between;margin-bottom:5px">
                        <span><strong>${hand2}</strong></span>
                        <span style="color:#f87171;font-weight:700">${data.equity2.toFixed(1)}%</span>
                    </div>
                    ${equityBar(data.equity2, '#f87171')}
                </div>
                ${data.board ? `<p style="color:#6b7688;margin-top:15px">Board: ${data.board}</p>` : '<p style="color:#6b7688;margin-top:15px">Preflop</p>'}
            </div>`;
    }
    document.getElementById('eq-results').classList.add('show');
}

async function calcRange() {
    const hand = document.getElementById('rng-hand').value.trim();
    const range = document.getElementById('rng-range').value.trim();
    const board = document.getElementById('rng-board').value.trim();
    document.getElementById('rng-loading').classList.add('show');
    document.getElementById('rng-results').classList.remove('show');

    const data = await apiCall('range', {hand, range, board});
    document.getElementById('rng-loading').classList.remove('show');

    if (data.error) {
        document.getElementById('rng-results').innerHTML = `<div class="result-card"><p style="color:#f87171">Error: ${data.error}</p></div>`;
    } else {
        document.getElementById('rng-results').innerHTML = `
            <div class="result-card">
                <h3>Range Equity</h3>
                <div style="display:flex;justify-content:space-between;margin-bottom:5px">
                    <span><strong>${hand}</strong> vs <strong>${range}</strong></span>
                    <span style="color:#4ade80;font-weight:700">${data.equity.toFixed(1)}%</span>
                </div>
                ${equityBar(data.equity)}
                ${data.board ? `<p style="color:#6b7688;margin-top:10px">Board: ${data.board}</p>` : '<p style="color:#6b7688;margin-top:10px">Preflop</p>'}
            </div>`;
    }
    document.getElementById('rng-results').classList.add('show');
}

async function calcICM() {
    const names = document.getElementById('icm-names').value.split(',').map(s=>s.trim());
    const stacks = document.getElementById('icm-stacks').value.split(',').map(s=>parseInt(s.trim()));
    const payouts = document.getElementById('icm-payouts').value.split(',').map(s=>parseFloat(s.trim()));

    const data = await apiCall('icm', {names, stacks, payouts});

    if (data.error) {
        document.getElementById('icm-results').innerHTML = `<div class="result-card"><p style="color:#f87171">Error: ${data.error}</p></div>`;
    } else {
        let rows = '';
        const totalChips = stacks.reduce((a,b) => a+b, 0);
        data.players.forEach(p => {
            rows += `<tr>
                <td><strong>${p.name}</strong></td>
                <td>${p.chips.toLocaleString()}</td>
                <td>${(p.chips/totalChips*100).toFixed(1)}%</td>
                <td style="color:#4ade80;font-weight:600">$${p.icm_equity.toFixed(2)}</td>
                <td>${p.icm_pct.toFixed(1)}%</td>
            </tr>`;
        });
        document.getElementById('icm-results').innerHTML = `
            <div class="result-card">
                <h3>ICM Equity Results</h3>
                <table>
                    <thead><tr><th>Player</th><th>Chips</th><th>Chip %</th><th>ICM $</th><th>ICM %</th></tr></thead>
                    <tbody>${rows}</tbody>
                </table>
                <p style="color:#6b7688;margin-top:15px;font-size:13px">Prize pool: $${data.total_prizes.toFixed(2)} | Total chips: ${totalChips.toLocaleString()}</p>
            </div>`;
    }
    document.getElementById('icm-results').classList.add('show');
}

async function calcSpot() {
    const hand = document.getElementById('spot-hand').value.trim();
    const range = document.getElementById('spot-range').value.trim();
    const board = document.getElementById('spot-board').value.trim();
    const heroStack = parseInt(document.getElementById('spot-hero').value);
    const villainStack = parseInt(document.getElementById('spot-villain').value);
    const blindsStr = document.getElementById('spot-blinds').value.trim();
    const stacks = document.getElementById('spot-stacks').value.split(',').map(s=>parseInt(s.trim()));
    const payouts = document.getElementById('spot-payouts').value.split(',').map(s=>parseFloat(s.trim()));
    const blinds = blindsStr.split('/').map(s=>parseInt(s.trim()));

    document.getElementById('spot-loading').classList.add('show');
    document.getElementById('spot-results').classList.remove('show');

    const data = await apiCall('spot', {hand, range, board, heroStack, villainStack, stacks, payouts, sb: blinds[0], bb: blinds[1]});
    document.getElementById('spot-loading').classList.remove('show');

    if (data.error) {
        document.getElementById('spot-results').innerHTML = `<div class="result-card"><p style="color:#f87171">Error: ${data.error}</p></div>`;
    } else {
        const isFold = data.decision === 'FOLD';
        const decClass = isFold ? 'fold' : 'call';
        const zoneClass = getZoneClass(data.zone);

        document.getElementById('spot-results').innerHTML = `
            <div class="decision-box ${decClass}">
                <div class="decision-text ${decClass}">${data.decision}</div>
                <div class="decision-reason">${isFold
                    ? `Card equity (${data.card_equity.toFixed(1)}%) is below required equity (${data.required_equity.toFixed(1)}%) after ICM adjustment`
                    : `Card equity (${data.card_equity.toFixed(1)}%) exceeds required equity (${data.required_equity.toFixed(1)}%) — +EV call`}</div>
            </div>
            <div class="result-card" style="margin-top:15px">
                <h3>Analysis Breakdown</h3>
                <div class="stat-grid">
                    <div class="stat-item">
                        <div class="stat-value">${data.card_equity.toFixed(1)}%</div>
                        <div class="stat-label">Card Equity</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${data.bubble_factor.toFixed(2)}x</div>
                        <div class="stat-label">Bubble Factor</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${data.required_equity.toFixed(1)}%</div>
                        <div class="stat-label">Required Equity</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${data.bb_count}bb</div>
                        <div class="stat-label">Stack Depth</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value"><span class="${zoneClass}" style="font-size:18px;padding:5px 15px">${data.zone}</span></div>
                        <div class="stat-label">Stack Zone</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value ${decClass}">${data.ev_positive ? '+EV' : '-EV'}</div>
                        <div class="stat-label">Expected Value</div>
                    </div>
                </div>
            </div>`;
    }
    document.getElementById('spot-results').classList.add('show');
}

async function calcPushFold() {
    const stack = parseInt(document.getElementById('pf-stack').value);
    const bb = parseInt(document.getElementById('pf-bb').value);
    const pos = document.getElementById('pf-pos').value;

    const data = await apiCall('pushfold', {stack, bb, position: pos});

    if (data.error) {
        document.getElementById('pf-results').innerHTML = `<div class="result-card"><p style="color:#f87171">Error: ${data.error}</p></div>`;
    } else {
        const zoneClass = getZoneClass(data.zone);
        document.getElementById('pf-results').innerHTML = `
            <div class="result-card">
                <h3>Push/Fold Analysis</h3>
                <div class="stat-grid" style="margin-bottom:20px">
                    <div class="stat-item">
                        <div class="stat-value">${data.bb_count.toFixed(1)}</div>
                        <div class="stat-label">Big Blinds</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value"><span class="${zoneClass}" style="font-size:18px;padding:5px 15px">${data.zone}</span></div>
                        <div class="stat-label">Harrington Zone</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" style="font-size:18px">${data.status}</div>
                        <div class="stat-label">Status</div>
                    </div>
                </div>
                <h3>Push Range from ${pos.toUpperCase()}</h3>
                <p style="font-family:Consolas,monospace;font-size:15px;color:#4ade80;margin-top:10px;line-height:1.8;background:#0a0e17;padding:15px;border-radius:8px">${data.range}</p>
                <p style="color:#6b7688;margin-top:15px;font-size:13px">${data.strategy}</p>
            </div>`;
    }
    document.getElementById('pf-results').classList.add('show');
}

async function calcFinalTable() {
    const names = document.getElementById('ft-names').value.split(',').map(s=>s.trim());
    const stacks = document.getElementById('ft-stacks').value.split(',').map(s=>parseInt(s.trim()));
    const payouts = document.getElementById('ft-payouts').value.split(',').map(s=>parseFloat(s.trim()));
    const blindsStr = document.getElementById('ft-blinds').value.trim();
    const blinds = blindsStr.split('/').map(s=>parseInt(s.trim()));

    document.getElementById('ft-loading').classList.add('show');
    document.getElementById('ft-results').classList.remove('show');

    const data = await apiCall('finaltable', {names, stacks, payouts, sb: blinds[0], bb: blinds[1]});
    document.getElementById('ft-loading').classList.remove('show');

    if (data.error) {
        document.getElementById('ft-results').innerHTML = `<div class="result-card"><p style="color:#f87171">Error: ${data.error}</p></div>`;
    } else {
        let rows = '';
        data.players.forEach(p => {
            const zoneClass = getZoneClass(p.zone);
            rows += `<tr>
                <td><strong>${p.name}</strong></td>
                <td>${p.chips.toLocaleString()}</td>
                <td>${p.chip_pct.toFixed(1)}%</td>
                <td style="color:#4ade80;font-weight:600">$${p.icm_equity.toLocaleString(undefined,{minimumFractionDigits:2})}</td>
                <td>${p.icm_pct.toFixed(1)}%</td>
                <td>${p.bb_count.toFixed(0)}bb</td>
                <td><span class="zone-badge ${zoneClass}">${p.zone}</span></td>
            </tr>`;
        });

        let dealRows = '';
        if (data.deals) {
            data.deals.forEach(d => {
                dealRows += `<tr>
                    <td><strong>${d.name}</strong></td>
                    <td>${d.chips.toLocaleString()}</td>
                    <td>${d.chip_share.toFixed(1)}%</td>
                    <td style="color:#4ade80;font-weight:600">$${d.icm_equity.toLocaleString(undefined,{minimumFractionDigits:2})}</td>
                </tr>`;
            });
        }

        document.getElementById('ft-results').innerHTML = `
            <div class="result-card">
                <h3>Final Table Standings</h3>
                <table>
                    <thead><tr><th>Player</th><th>Chips</th><th>Chip %</th><th>ICM $</th><th>ICM %</th><th>Depth</th><th>Zone</th></tr></thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
            ${dealRows ? `<div class="result-card">
                <h3>ICM Deal Equity</h3>
                <table>
                    <thead><tr><th>Player</th><th>Chips</th><th>Chip Share</th><th>Deal Amount</th></tr></thead>
                    <tbody>${dealRows}</tbody>
                </table>
            </div>` : ''}`;
    }
    document.getElementById('ft-results').classList.add('show');
}
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/equity', methods=['POST'])
def api_equity():
    try:
        d = request.json
        h1 = d['hand1'].strip().split() if ' ' in d['hand1'] else [d['hand1'][:2], d['hand1'][2:]]
        h2 = d['hand2'].strip().split() if ' ' in d['hand2'] else [d['hand2'][:2], d['hand2'][2:]]
        board = d.get('board', '').strip()
        board_cards = board.split() if board else []
        eq = hand_vs_hand_equity(h1, h2, board_cards if board_cards else None)
        return jsonify({'equity1': eq, 'equity2': 100 - eq, 'board': board or None})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/range', methods=['POST'])
def api_range():
    try:
        d = request.json
        hand = d['hand'].strip().split() if ' ' in d['hand'] else [d['hand'][:2], d['hand'][2:]]
        board = d.get('board', '').strip()
        board_cards = board.split() if board else []
        eq = hand_vs_range_equity(hand, d['range'], board_cards if board_cards else None)
        return jsonify({'equity': eq, 'board': board or None})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/icm', methods=['POST'])
def api_icm():
    try:
        d = request.json
        equities = icm_equity(d['stacks'], d['payouts'])
        total = sum(d['payouts'])
        players = []
        for i, (name, chips, eq) in enumerate(zip(d['names'], d['stacks'], equities)):
            players.append({
                'name': name,
                'chips': chips,
                'icm_equity': eq,
                'icm_pct': eq / total * 100 if total else 0
            })
        return jsonify({'players': players, 'total_prizes': total})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/spot', methods=['POST'])
def api_spot():
    try:
        d = request.json
        hand_str = d['hand'].strip()
        hand = [hand_str[:2], hand_str[2:]]
        board_str = d.get('board', '').strip()
        board_cards = board_str.split() if board_str else []

        blind = BlindLevel(sb=d['sb'], bb=d['bb'], ante=d['bb'] // 10)
        analyzer = TournamentAnalyzer()

        spot = analyzer.analyze_spot(
            hero_hand=hand,
            villain_range=d['range'],
            hero_stack=d['heroStack'],
            villain_stack=d['villainStack'],
            all_stacks=d['stacks'],
            payouts=d['payouts'],
            board=board_cards if board_cards else None,
            blind_level=blind
        )

        zone = spot['stack_zone']
        zone_name = zone['zone'] if isinstance(zone, dict) else str(zone)

        return jsonify({
            'card_equity': spot['card_equity'],
            'bubble_factor': spot['bubble_factor'],
            'required_equity': spot['required_equity'],
            'zone': zone_name,
            'bb_count': int(spot['bb_count']),
            'ev_positive': spot['ev_positive'],
            'decision': spot['decision'],
            'equity_surplus': spot.get('equity_surplus', 0)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/pushfold', methods=['POST'])
def api_pushfold():
    try:
        d = request.json
        bb_count = d['stack'] / d['bb']
        zone = categorize_stack(bb_count)
        zone_name = zone['zone'] if isinstance(zone, dict) else str(zone)
        status = zone['status'] if isinstance(zone, dict) else ''
        strategy = zone['strategy'] if isinstance(zone, dict) else ''

        ranges = push_fold_range(bb_count, d.get('position', 'btn'))
        range_str = ', '.join(ranges) if isinstance(ranges, list) else str(ranges)

        return jsonify({
            'bb_count': bb_count,
            'zone': zone_name,
            'status': status,
            'strategy': strategy,
            'range': range_str
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/finaltable', methods=['POST'])
def api_finaltable():
    try:
        d = request.json
        blind = BlindLevel(level=1, small_blind=d['sb'], big_blind=d['bb'], ante=d['bb'] // 10)
        analyzer = TournamentAnalyzer()
        names = d.get('names', [])

        report = analyzer.final_table_report(
            stacks=d['stacks'],
            payouts=d['payouts'],
            blind_level=blind
        )

        players = []
        for i, p in enumerate(report['players']):
            zone = p.get('zone', {})
            zone_name = zone['zone'] if isinstance(zone, dict) else str(zone)
            player_name = names[i] if i < len(names) else p.get('name', f'Player {i+1}')
            players.append({
                'name': player_name,
                'chips': p.get('chips', d['stacks'][i] if i < len(d['stacks']) else 0),
                'chip_pct': p.get('chip_pct', 0),
                'icm_equity': p.get('icm_equity', 0),
                'icm_pct': p.get('icm_pct', 0),
                'bb_count': p.get('bb_count', 0),
                'zone': zone_name
            })

        deals = []
        if 'deals' in report:
            for i, dd in enumerate(report['deals']):
                deal_name = names[i] if i < len(names) else dd.get('name', f'Player {i+1}')
                deals.append({
                    'name': deal_name,
                    'chips': dd.get('chips', 0),
                    'chip_share': dd.get('chip_share', 0),
                    'icm_equity': dd.get('icm_equity', 0)
                })

        return jsonify({'players': players, 'deals': deals})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  POKER TOURNAMENT ANALYZER — WEB INTERFACE")
    print("="*60)
    print(f"\n  Open your browser to: http://localhost:5000")
    print(f"  Press Ctrl+C to stop\n")
    app.run(host='0.0.0.0', port=5000, debug=False)