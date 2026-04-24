/**
 * Chart Pattern AI — Frontend Logic
 * Handles file upload, API calls, and results rendering
 */

// ─── DOM Elements ─────────────────────────────────────────
const uploadZone = document.getElementById('uploadZone');
const previewZone = document.getElementById('previewZone');
const fileInput = document.getElementById('fileInput');
const btnBrowse = document.getElementById('btnBrowse');
const btnAnalyze = document.getElementById('btnAnalyze');
const btnClear = document.getElementById('btnClear');
const btnNewAnalysis = document.getElementById('btnNewAnalysis');
const previewImage = document.getElementById('previewImage');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const candlestickBody = document.getElementById('candlestickBody');
const chartBody = document.getElementById('chartBody');
const summaryCard = document.getElementById('summaryCard');
const summaryBody = document.getElementById('summaryBody');
const uploadCard = document.getElementById('uploadCard');
const annotatedImage = document.getElementById('annotatedImage');
const recommendationCard = document.getElementById('recommendationCard');
const recValue = document.getElementById('recValue');
const recIcon = document.getElementById('recIcon');
const recSentiment = document.getElementById('recSentiment');
const recIconWrapper = document.getElementById('recIconWrapper');

// Trade Setup Elements
const inputCapital = document.getElementById('inputCapital');
const inputRisk = document.getElementById('inputRisk');
const inputRR = document.getElementById('inputRR');
const tradeSetupCard = document.getElementById('tradeSetupCard');
const setupActionPanel = document.getElementById('setupActionPanel');
const setupAction = document.getElementById('setupAction');
const setupEntry = document.getElementById('setupEntry');
const setupSL = document.getElementById('setupSL');
const setupTarget = document.getElementById('setupTarget');
const setupMaxRisk = document.getElementById('setupMaxRisk');
const setupRiskPerShare = document.getElementById('setupRiskPerShare');
const setupShares = document.getElementById('setupShares');
const setupWarning = document.getElementById('setupWarning');
const setupPNL = document.getElementById('setupPNL');
const scenariosCard = document.getElementById('scenariosCard');
const preTradeBody = document.getElementById('preTradeBody');
const postTradeBody = document.getElementById('postTradeBody');

let selectedFile = null;
let lastAnalysisData = null; // Store data for re-calculation

// ─── Event Listeners ──────────────────────────────────────

// Click to browse
btnBrowse.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
});

uploadZone.addEventListener('click', () => {
    fileInput.click();
});

// Stop click from inputs from bubbling up to uploadZone
document.getElementById('riskInputs').addEventListener('click', (e) => {
    e.stopPropagation();
});

// File selected via input
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFileSelected(file);
});

// Drag and drop
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFileSelected(file);
    }
});

// Analyze button
btnAnalyze.addEventListener('click', analyzeImage);

// Clear button
btnClear.addEventListener('click', resetUpload);

// New analysis button
btnNewAnalysis.addEventListener('click', () => {
    resultsSection.style.display = 'none';
    lastAnalysisData = null;
    resetUpload();
});

// Update calculations when inputs change
[inputCapital, inputRisk, inputRR].forEach(input => {
    input.addEventListener('input', () => {
        if (lastAnalysisData) {
            calculateTradeSetup(lastAnalysisData);
        }
    });
});

// ─── Handle File Selection ────────────────────────────────
function handleFileSelected(file) {
    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadZone.style.display = 'none';
        previewZone.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

// ─── Reset Upload ─────────────────────────────────────────
function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    uploadZone.style.display = '';
    previewZone.style.display = 'none';
    uploadCard.style.display = '';
}

// ─── Analyze Image ────────────────────────────────────────
async function analyzeImage() {
    if (!selectedFile) return;

    // Show loading
    uploadCard.style.display = 'none';
    loadingSection.style.display = 'block';
    resultsSection.style.display = 'none';

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (data.error) {
            alert('Error: ' + data.error);
            loadingSection.style.display = 'none';
            uploadCard.style.display = '';
            return;
        }

        lastAnalysisData = data;
        displayResults(data);
    } catch (error) {
        alert('Failed to connect to server. Make sure the app is running.');
        console.error(error);
        loadingSection.style.display = 'none';
        uploadCard.style.display = '';
    }
}

// ─── Display Results ──────────────────────────────────────
function displayResults(data) {
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'block';

    // Render candlestick patterns
    renderPatterns(candlestickBody, data.candlestick_patterns, 'candlestick');

    // Render chart patterns
    renderPatterns(chartBody, data.chart_patterns, 'chart');

    // Update recommendation
    updateRecommendation(data);

    // Generate summary
    generateSummary(data);

    // Show annotated image if available
    if (data.annotated_image) {
        annotatedImage.src = `data:image/png;base64,${data.annotated_image}`;
    }
    
    // Calculate Trade Setup (1% Risk Rule)
    calculateTradeSetup(data);

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─── Calculate Trade Setup ────────────────────────────────
function calculateTradeSetup(data) {
    const sentiment = data.sentiment;
    const capital = parseFloat(inputCapital.value) || 0;
    const riskPercent = parseFloat(inputRisk.value) || 0;
    const rrRatio = parseFloat(inputRR.value) || 0;
    
    // Use prices from backend
    const entryPrice = data.trigger_price;
    const stopLoss = data.stop_loss;

    if (!capital || !entryPrice || !stopLoss || sentiment === 'Neutral') {
        tradeSetupCard.style.display = 'none';
        scenariosCard.style.display = 'none';
        return;
    }

    // Determine Action and calculate Risk
    let action = 'BUY';
    let riskPerShare = Math.abs(entryPrice - stopLoss);
    
    // Calculate Dynamic Target based on User's RR Ratio
    let targetPrice = 0;
    if (sentiment.includes('Bullish')) {
        action = 'BUY';
        targetPrice = entryPrice + (riskPerShare * rrRatio);
        setupActionPanel.style.borderLeftColor = '#10b981';
        setupAction.style.color = '#10b981';
    } else {
        action = 'SELL (SHORT)';
        targetPrice = entryPrice - (riskPerShare * rrRatio);
        setupActionPanel.style.borderLeftColor = '#ef4444';
        setupAction.style.color = '#ef4444';
    }

    // Math for User's Risk Rule
    const maxRisk = capital * (riskPercent / 100);
    const positionSize = riskPerShare > 0 ? Math.floor(maxRisk / riskPerShare) : 0;
    const actualRisk = positionSize * riskPerShare;

    // Populate UI
    tradeSetupCard.style.display = 'block';
    setupAction.textContent = action;
    setupEntry.textContent = `₹${entryPrice.toLocaleString()}`;
    setupSL.textContent = `₹${stopLoss.toLocaleString()}`;
    setupTarget.textContent = `₹${targetPrice.toLocaleString()} (1:${rrRatio})`;
    
    setupMaxRisk.textContent = `₹${maxRisk.toLocaleString()} (${riskPercent}%)`;
    setupRiskPerShare.textContent = `₹${riskPerShare.toLocaleString()}`;
    setupShares.textContent = `${positionSize.toLocaleString()} Shares`;
    
    // Potential P&L
    const potentialProfit = positionSize * Math.abs(targetPrice - entryPrice);
    setupPNL.innerHTML = `<span style="color: #10b981;">₹${Math.round(potentialProfit).toLocaleString()}</span> : <span style="color: #ef4444;">₹${Math.round(actualRisk).toLocaleString()}</span>`;
    
    setupWarning.textContent = `If stop loss hits, your total loss will be strictly limited to ₹${actualRisk.toLocaleString()} (within your ${riskPercent}% limit).`;
    
    // Render Scenarios with User's Inputs
    renderAllScenarios(data, capital, riskPercent, rrRatio);
}

function renderAllScenarios(data, capital, riskPercent, rrRatio) {
    const preScenarios = data.pre_trade;
    const postScenarios = data.post_trade;

    if (!preScenarios || !postScenarios) {
        scenariosCard.style.display = 'none';
        return;
    }

    scenariosCard.style.display = 'block';
    preTradeBody.innerHTML = '';
    postTradeBody.innerHTML = '';

    const riskLimit = capital * (riskPercent / 100);

    // 1. Render Pre-Trade (Dynamic Math)
    preScenarios.forEach(s => {
        const card = document.createElement('div');
        card.style.background = 'rgba(255,255,255,0.03)';
        card.style.padding = '1.5rem';
        card.style.borderRadius = '12px';
        card.style.border = '1px solid rgba(255,255,255,0.05)';

        const riskPerShare = Math.abs(s.entry - s.sl);
        const dynamicTarget = s.header.includes('REVERSAL') ? s.target : (s.entry + (s.entry > s.sl ? 1 : -1) * riskPerShare * rrRatio);
        const shares = riskPerShare > 0 ? Math.floor(riskLimit / riskPerShare) : 0;
        const potentialProfit = shares * Math.abs(dynamicTarget - s.entry);
        const actualRisk = shares * riskPerShare;

        card.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <h4 style="color: #f59e0b; margin: 0; font-size: 1rem; letter-spacing: 0.5px;">${s.header}</h4>
                <span style="background: rgba(245, 158, 11, 0.15); color: #f59e0b; padding: 4px 10px; border-radius: 20px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase;">Action Plan</span>
            </div>
            
            <div style="margin-bottom: 1rem;">
                <p style="color: #9ca3af; font-size: 0.8rem; text-transform: uppercase; margin-bottom: 4px;">📌 Condition</p>
                <p style="color: #fff; font-weight: 500;">${s.condition}</p>
            </div>

            <div style="margin-bottom: 1rem; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 8px;">
                <p style="color: #9ca3af; font-size: 0.8rem; text-transform: uppercase; margin-bottom: 4px;">🎯 Trade</p>
                <p style="color: #fff; font-size: 0.9rem;">Entry: <strong>${s.entry.toFixed(1)}</strong></p>
                <p style="color: #ef4444; font-size: 0.9rem;">Stoploss: <strong>${s.sl.toFixed(1)}</strong> (${riskPerShare.toFixed(1)} pts risk)</p>
                <p style="color: #10b981; font-size: 0.9rem;">Target: <strong>${dynamicTarget.toFixed(1)}</strong> (1:${rrRatio} RR)</p>
                <p style="color: #fff; font-size: 0.85rem; margin-top: 5px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 5px;">
                    Potential P&L: <span style="color: #10b981;">₹${Math.round(potentialProfit).toLocaleString()}</span> : <span style="color: #ef4444;">₹${Math.round(actualRisk).toLocaleString()}</span>
                </p>
            </div>

            <div style="margin-bottom: 1rem;">
                <p style="color: #9ca3af; font-size: 0.8rem; text-transform: uppercase; margin-bottom: 4px;">💰 Position Size & Risk Management</p>
                <p style="color: #fff; font-size: 0.9rem;">Risk per share = ${riskPerShare.toFixed(1)}</p>
                <p style="color: #f59e0b; font-size: 0.9rem;">Maximum Risk Allowed: <strong>₹${riskLimit.toLocaleString()}</strong></p>
                <p style="color: #00e5ff; font-size: 1.1rem; font-weight: 700; margin-top: 5px;">₹${riskLimit.toLocaleString()} / ${riskPerShare.toFixed(1)} \u2248 ${shares.toLocaleString()} shares</p>
            </div>

            <p style="color: #9ca3af; font-size: 0.85rem; font-style: italic;">👉 ${s.note}</p>
        `;
        preTradeBody.appendChild(card);
    });

    // 2. Render Post-Trade (The action points)
    postScenarios.forEach(s => {
        const card = document.createElement('div');
        card.style.background = 'rgba(255,255,255,0.02)';
        card.style.padding = '1.2rem';
        card.style.borderRadius = '10px';
        card.style.border = '1px solid rgba(255,255,255,0.03)';
        
        let pointsHtml = s.points.map(p => `
            <li style="margin-bottom: 0.4rem; color: #9ca3af; font-size: 0.85rem; display: flex; align-items: flex-start; gap: 0.4rem;">
                <span class="material-icons-round" style="font-size: 0.9rem; color: #a855f7; margin-top: 0.1rem;">bolt</span>
                <span>${p}</span>
            </li>
        `).join('');

        card.innerHTML = `
            <h4 style="color: #fff; margin-bottom: 0.8rem; font-size: 0.95rem; border-left: 2px solid #a855f7; padding-left: 8px;">${s.title}</h4>
            <ul style="list-style: none; padding: 0;">
                ${pointsHtml}
            </ul>
        `;
        postTradeBody.appendChild(card);
    });
}

// ─── Render Pattern List ──────────────────────────────────
function renderPatterns(container, patterns, category) {
    container.innerHTML = '';

    if (!patterns || patterns.length === 0) {
        container.innerHTML = `
            <div class="no-pattern">
                <span class="material-icons-round">search_off</span>
                <p>No ${category} patterns detected above threshold</p>
            </div>
        `;
        return;
    }

    patterns.forEach((pattern, index) => {
        const typeClass = getTypeClass(pattern.type);
        const colorClass = getColorClass(pattern.type);

        const item = document.createElement('div');
        item.className = 'pattern-item';
        item.style.animationDelay = `${index * 0.1}s`;

        item.innerHTML = `
            <div class="pattern-name">${pattern.display_name}</div>
            <span class="pattern-type ${typeClass}">${pattern.type}</span>
            <div class="confidence-bar-container">
                <div class="confidence-bar">
                    <div class="confidence-fill ${colorClass}" data-width="${pattern.confidence}"></div>
                </div>
                <span class="confidence-text">${pattern.confidence}%</span>
            </div>
        `;

        container.appendChild(item);

        // Animate confidence bar after a small delay
        requestAnimationFrame(() => {
            setTimeout(() => {
                const fill = item.querySelector('.confidence-fill');
                fill.style.width = `${pattern.confidence}%`;
            }, 100 + index * 150);
        });
    });
}

// ─── Generate Summary ─────────────────────────────────────
function generateSummary(data) {
    const csPatterns = data.candlestick_patterns.filter(p => p.confidence > 20);
    const chartPatterns = data.chart_patterns.filter(p => p.confidence > 20);

    if (csPatterns.length === 0 && chartPatterns.length === 0) {
        summaryCard.style.display = 'none';
        return;
    }

    summaryCard.style.display = '';
    let html = '';

    if (csPatterns.length > 0) {
        const topCS = csPatterns[0];
        const colorClass = getColorClass(topCS.type);
        html += `<p>📊 <strong>Candlestick:</strong> The strongest candlestick signal is 
            <span class="summary-highlight ${colorClass}">${topCS.display_name}</span> 
            with <strong>${topCS.confidence}%</strong> confidence — a 
            <span class="summary-highlight ${colorClass}">${topCS.type}</span> signal.</p>`;
    }

    if (chartPatterns.length > 0) {
        const topChart = chartPatterns[0];
        const colorClass = getColorClass(topChart.type);
        html += `<p>📈 <strong>Chart Pattern:</strong> The dominant chart pattern is 
            <span class="summary-highlight ${colorClass}">${topChart.display_name}</span> 
            with <strong>${topChart.confidence}%</strong> confidence — a 
            <span class="summary-highlight ${colorClass}">${topChart.type}</span> signal.</p>`;
    }

    summaryBody.innerHTML = html;
}

function updateRecommendation(data) {
    if (!data.recommendation) {
        recommendationCard.style.display = 'none';
        return;
    }

    recommendationCard.style.display = 'flex';
    recValue.textContent = data.recommendation;
    recSentiment.textContent = `Market sentiment is ${data.sentiment}.`;

    // Update styles based on sentiment
    recommendationCard.classList.remove('state-bullish', 'state-bearish', 'state-neutral');

    if (data.sentiment === 'Bullish') {
        recommendationCard.classList.add('state-bullish');
        recIcon.textContent = 'trending_up';
    } else if (data.sentiment === 'Bearish') {
        recommendationCard.classList.add('state-bearish');
        recIcon.textContent = 'trending_down';
    } else {
        recommendationCard.classList.add('state-neutral');
        recIcon.textContent = 'trending_flat';
    }
}

// ─── Animations Removed ───────────────────────────────────────
// (initThreeJS and initTiltEffect removed per user request)

// ─── Helpers ──────────────────────────────────────────────
function getTypeClass(type) {
    const t = type.toLowerCase();
    if (t.includes('bullish')) return 'type-bullish';
    if (t.includes('bearish')) return 'type-bearish';
    return 'type-neutral';
}

function getColorClass(type) {
    const t = type.toLowerCase();
    if (t.includes('bullish')) return 'bullish';
    if (t.includes('bearish')) return 'bearish';
    return 'neutral';
}
