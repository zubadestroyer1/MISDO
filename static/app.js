/**
 * MISDO Dashboard — Client-Side Application Logic
 * =================================================
 * Real satellite-trained models with location selection.
 */

const SPATIAL = 256;
const API = '';

// Location metadata
const LOCATIONS = {
    amazon: {
        name: 'Amazon Rainforest — Rondônia, Brazil',
        coords: '10.5°S, 63.0°W  ·  256×256 px @ ~30m resolution',
        seed: 42,
    },
    borneo: {
        name: 'Borneo Rainforest — Kalimantan, Indonesia',
        coords: '1.5°S, 116.0°E  ·  256×256 px @ ~30m resolution',
        seed: 137,
    },
    congo: {
        name: 'Congo Basin — DRC',
        coords: '1.0°S, 23.5°E  ·  256×256 px @ ~30m resolution',
        seed: 256,
    },
    california: {
        name: 'Sierra Nevada — California, USA',
        coords: '37.5°N, 119.5°W  ·  256×256 px @ ~30m resolution',
        seed: 777,
    },
};

const MODEL_INFO = [
    { name: 'FireRiskNet', icon: '🔥', source: 'VIIRS VNP14IMG', params: '4.03M' },
    { name: 'ForestLossNet', icon: '🌲', source: 'Hansen GFC', params: '1.33M' },
    { name: 'HydroRiskNet', icon: '💧', source: 'SRTM/HydroSHEDS', params: '4.59M' },
    { name: 'SoilRiskNet', icon: '🏜', source: 'SMAP L3', params: '3.88M' },
];

// ═══════════════════════════════════════════════════════════════
// Initialization
// ═══════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    renderModelStatus();
    initSliders();
    loadAgentMasks();
    recalculate();
    loadEnvState();
    initForestClick();
});

// ═══════════════════════════════════════════════════════════════
// Model Status Bar
// ═══════════════════════════════════════════════════════════════

function renderModelStatus() {
    const bar = document.getElementById('model-status-bar');
    bar.innerHTML = MODEL_INFO.map(m => `
        <div class="model-status-chip">
            <div class="status-dot active"></div>
            <div class="status-info">
                <div class="status-name">${m.icon} ${m.name}</div>
                <div class="status-detail">${m.source} · ${m.params}</div>
            </div>
        </div>
    `).join('');
}

// ═══════════════════════════════════════════════════════════════
// Location Selection
// ═══════════════════════════════════════════════════════════════

async function changeLocation() {
    const select = document.getElementById('location-select');
    const loc = LOCATIONS[select.value];
    const btn = document.getElementById('btn-location');

    // Update display
    document.getElementById('location-name').textContent = loc.name;
    document.getElementById('location-coords').textContent = loc.coords;

    btn.disabled = true;
    btn.textContent = '⟳ Analyzing…';
    showToast(`Loading ${loc.name}…`, 'info');

    try {
        const res = await fetch(`${API}/api/change-location`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ location: select.value, seed: loc.seed }),
        });
        const data = await res.json();

        if (data.status === 'ok') {
            // Reload everything
            await loadAgentMasks();
            await recalculate();
            await loadEnvState();
            showToast(`✓ Now analyzing: ${loc.name}`, 'success');
        } else {
            showToast('Location change failed', 'error');
        }
    } catch (err) {
        showToast('Failed to change location', 'error');
        console.error(err);
    } finally {
        btn.disabled = false;
        btn.textContent = '🛰 Analyze Region';
    }
}

// ═══════════════════════════════════════════════════════════════
// Agent Risk Maps
// ═══════════════════════════════════════════════════════════════

async function loadAgentMasks() {
    try {
        const res = await fetch(`${API}/api/agent-masks`);
        const data = await res.json();

        data.masks.forEach((mask, i) => {
            setImage(`agent-img-${i}`, mask.image);
            renderStats(`agent-stats-${i}`, mask.stats);
        });
    } catch (err) {
        showToast('Failed to load agent masks', 'error');
        console.error(err);
    }
}

function renderStats(containerId, stats) {
    const el = document.getElementById(containerId);
    if (!el || !stats) return;
    el.innerHTML = Object.entries(stats).map(([key, val]) =>
        `<div class="stat-chip">${key}: <span>${val}</span></div>`
    ).join('');
}

// ═══════════════════════════════════════════════════════════════
// Aggregator Controls
// ═══════════════════════════════════════════════════════════════

function initSliders() {
    for (let i = 0; i < 4; i++) {
        const slider = document.getElementById(`slider-${i}`);
        const display = document.getElementById(`slider-val-${i}`);
        slider.addEventListener('input', () => {
            display.textContent = parseFloat(slider.value).toFixed(2);
        });
    }
}

function getWeights() {
    return [0, 1, 2, 3].map(i =>
        parseFloat(document.getElementById(`slider-${i}`).value)
    );
}

function resetSliders() {
    for (let i = 0; i < 4; i++) {
        document.getElementById(`slider-${i}`).value = 0.5;
        document.getElementById(`slider-val-${i}`).textContent = '0.50';
    }
    showToast('Weights reset to defaults', 'info');
}

async function recalculate() {
    const btn = document.getElementById('btn-recalculate');
    btn.disabled = true;
    btn.textContent = '⟳ Computing…';

    try {
        const weights = getWeights();
        const res = await fetch(`${API}/api/aggregate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ weights }),
        });
        const data = await res.json();

        setImage('agg-img', data.image);

        document.getElementById('agg-min').textContent = data.stats.min;
        document.getElementById('agg-max').textContent = data.stats.max;
        document.getElementById('agg-mean').textContent = data.stats.mean;
        document.getElementById('agg-nogo').textContent = data.stats.no_go_pct + '%';

        loadEnvState();
        showToast(`Aggregation complete — weights [${weights.map(w => w.toFixed(2)).join(', ')}]`, 'success');
    } catch (err) {
        showToast('Aggregation failed', 'error');
        console.error(err);
    } finally {
        btn.disabled = false;
        btn.textContent = '✦ Recalculate';
    }
}

// ═══════════════════════════════════════════════════════════════
// RL Environment
// ═══════════════════════════════════════════════════════════════

async function loadEnvState() {
    try {
        const res = await fetch(`${API}/api/env-state`);
        const data = await res.json();
        updateEnvUI(data);
    } catch (err) {
        showToast('Failed to load env state', 'error');
        console.error(err);
    }
}

function updateEnvUI(data) {
    setImage('env-harm-img', data.harm_mask);
    setImage('env-forest-img', data.forest_state);
    setImage('env-infra-img', data.infrastructure);

    if (data.stats) {
        document.getElementById('env-harvests').textContent = data.stats.harvest_count;
        document.getElementById('env-forest').textContent = data.stats.forest_remaining_pct + '%';
        document.getElementById('env-infra').textContent = data.stats.infrastructure_pct + '%';
    }

    if (data.step_result) {
        const sr = data.step_result;
        const badge = document.getElementById('env-reward-badge');
        const rewEl = document.getElementById('env-reward');
        badge.style.display = 'flex';
        rewEl.textContent = sr.reward > 0 ? `+${sr.reward}` : sr.reward;
        rewEl.style.color = sr.valid ? '#4ade80' : '#f87171';
    }

    // Impact Analysis
    const impactPanel = document.getElementById('impact-panel');
    if (data.impact && impactPanel) {
        impactPanel.style.display = 'block';
        const imp = data.impact;
        document.getElementById('impact-contagion').textContent = imp.contagion_risk.toFixed(2);
        document.getElementById('impact-pollution').textContent = imp.downstream_pollution.toFixed(2);
        document.getElementById('impact-fragments').textContent = imp.fragmentation_score + ' patches';
        document.getElementById('impact-total').textContent = imp.total_ecosystem_cost.toFixed(2);

        // Color-code the total cost
        const totalEl = document.getElementById('impact-total');
        totalEl.style.color = imp.total_ecosystem_cost > 50 ? '#f87171' :
            imp.total_ecosystem_cost > 20 ? '#fbbf24' : '#4ade80';
    }
}

function initForestClick() {
    const clickArea = document.getElementById('forest-click-area');
    clickArea.addEventListener('click', async (e) => {
        const img = document.getElementById('env-forest-img');
        const rect = img.getBoundingClientRect();

        const x = (e.clientX - rect.left) / rect.width;
        const y = (e.clientY - rect.top) / rect.height;

        const padLeft = 0.10, padRight = 0.15, padTop = 0.08, padBottom = 0.10;
        const normX = (x - padLeft) / (1 - padLeft - padRight);
        const normY = (y - padTop) / (1 - padTop - padBottom);

        if (normX < 0 || normX > 1 || normY < 0 || normY > 1) {
            showToast('Click inside the map area', 'info');
            return;
        }

        const col = Math.floor(normX * SPATIAL);
        const row = Math.floor(normY * SPATIAL);

        showToast(`Harvesting at (${row}, ${col})…`, 'info');

        try {
            const res = await fetch(`${API}/api/env-step`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ row, col }),
            });
            const data = await res.json();
            updateEnvUI(data);

            if (data.step_result) {
                const sr = data.step_result;
                if (sr.valid) {
                    showToast(`✓ Harvest at (${sr.row}, ${sr.col}) — reward: ${sr.reward}`, 'success');
                } else {
                    showToast(`✗ Rejected — block at (${sr.row}, ${sr.col}) not contiguous to infrastructure`, 'error');
                }
                if (sr.terminated) {
                    showToast('🏁 Episode complete — harvest quota reached!', 'success');
                }
            }
        } catch (err) {
            showToast('Step failed', 'error');
            console.error(err);
        }
    });
}

async function resetEnv() {
    try {
        const res = await fetch(`${API}/api/env-reset`, { method: 'POST' });
        const data = await res.json();
        updateEnvUI(data);

        document.getElementById('env-reward-badge').style.display = 'none';
        showToast('Environment reset', 'success');
    } catch (err) {
        showToast('Reset failed', 'error');
        console.error(err);
    }
}

// ═══════════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════════

function setImage(imgId, base64Data) {
    const img = document.getElementById(imgId);
    if (!img) return;
    img.onload = () => img.classList.add('loaded');
    img.src = `data:image/png;base64,${base64Data}`;
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'toast-out 0.3s ease forwards';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Expose to HTML onclick
window.recalculate = recalculate;
window.resetSliders = resetSliders;
window.resetEnv = resetEnv;
window.changeLocation = changeLocation;
