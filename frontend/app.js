        // ──────────────────────────────────────────────────────────
        //  CONFIGURATION
        // ──────────────────────────────────────────────────────────

        const API_BASE = window.location.protocol === 'file:' ? 'http://localhost:8001' : '';

        // Pattern icons for the results list
        const PATTERN_ICONS = {
            'cup_and_handle': { emoji: '☕', bg: '#312e81' },
            'bull_flag': { emoji: '🟢', bg: '#14532d' },
            'bear_flag': { emoji: '🔴', bg: '#7f1d1d' },
            'pennant': { emoji: '🔺', bg: '#78350f' },
            'head_and_shoulders': { emoji: '👤', bg: '#1e3a5f' },
            'double_top': { emoji: '🔝', bg: '#4a1d6a' },
            'double_bottom': { emoji: '🔽', bg: '#164e63' },
        };

        // Beginner-friendly descriptions for each pattern
        const PATTERN_DESCRIPTIONS = {
            'cup_and_handle': `The Cup and Handle pattern looks like a tea cup on the chart. The price dropped and then slowly recovered (the cup), then pulled back slightly (the handle). Traders often watch for this pattern because it can signal that the stock is about to move higher after the handle phase completes. The quality score reflects how clean the cup shape is and whether volume confirmed the move. This does not guarantee a price increase — it is one of many signals traders use.`,
            'bull_flag': `The Bull Flag pattern starts with a strong upward move (the "pole") followed by a tight, slightly downward drift (the "flag"). Think of it like a flag on a pole — the pole is the rally, and the flag is a brief pause. Traders watch for a breakout above the flag, which can signal the upward move will continue. The tighter the flag and the stronger the pole, the more reliable this pattern tends to be.`,
            'bear_flag': `The Bear Flag is the opposite of a Bull Flag. It starts with a sharp drop (the pole) followed by a slight upward drift (the flag — a weak bounce). When the price breaks below the flag, it can signal that the downward move will continue. This is a bearish pattern, meaning it suggests prices may keep falling. Volume typically decreases during the flag and spikes on the breakdown.`,
            'pennant': `A Pennant looks like a small triangle that forms after a strong price move. The highs get lower and the lows get higher, squeezing the price into a tighter range — like two converging trendlines meeting at a point. The breakout usually happens in the same direction as the original move (the pole). If the pole was upward, expect an upward breakout; if downward, expect a downward breakout.`,
            'head_and_shoulders': `The Head and Shoulders pattern has three peaks: two smaller ones (the "shoulders") on either side of a taller one (the "head"). It typically forms at the top of an uptrend and signals a reversal to the downside. The "neckline" connects the two low points between the peaks. When the price breaks below the neckline, it confirms the pattern. Declining volume from left shoulder to right shoulder strengthens the signal.`,
            'double_top': `The Double Top pattern looks like the letter "M" on the chart. The price reaches a high, pulls back, then tries to reach the same high again but fails. This suggests that buyers ran out of steam and the price may reverse downward. When the price breaks below the valley between the two peaks, the pattern is confirmed. It's a bearish reversal signal — meaning the uptrend may be ending.`,
            'double_bottom': `The Double Bottom pattern looks like the letter "W" on the chart. The price drops to a low, bounces up, then drops to roughly the same low again but holds. This suggests that sellers ran out of steam and the price may reverse upward. When the price breaks above the peak between the two lows, the pattern is confirmed. It's a bullish reversal signal — meaning the downtrend may be ending.`,
        };

        // ──────────────────────────────────────────────────────────
        //  STATE
        // ──────────────────────────────────────────────────────────

        let scanResults = [];       // Current scan results
        let selectedResult = null;  // Currently selected result
        let chart = null;           // TradingView chart instance
        let candleSeries = null;    // Candlestick series
        let volumeSeries = null;    // Volume histogram series
        let priceLines = [];        // Current price level lines on chart
        let isDarkMode = true;      // Theme state
        let liveResults = [];       // Live scan results (separate from backtest)
        let livePopupOpen = false;  // Whether the popup is visible

        // ──────────────────────────────────────────────────────────
        //  DOM REFERENCES
        // ──────────────────────────────────────────────────────────

        const els = {
            loadingOverlay: document.getElementById('loadingOverlay'),
            loadingText: document.getElementById('loadingText'),
            errorBanner: document.getElementById('errorBanner'),
            marketBadge: document.getElementById('marketBadge'),
            marketStatus: document.getElementById('marketStatus'),
            lastScanInfo: document.getElementById('lastScanInfo'),
            scanStats: document.getElementById('scanStats'),
            refreshBtn: document.getElementById('refreshBtn'),
            themeToggle: document.getElementById('themeToggle'),
            patternSelect: document.getElementById('patternSelect'),
            intervalSelect: document.getElementById('intervalSelect'),
            lookbackSelect: document.getElementById('lookbackSelect'),
            liveAlertsToggle: document.getElementById('liveAlertsToggle'),
            scanBtn: document.getElementById('scanBtn'),
            sortSelect: document.getElementById('sortSelect'),
            resultsList: document.getElementById('resultsList'),
            resultsCount: document.getElementById('resultsCount'),
            chartContainer: document.getElementById('chart'),
            chartOverlay: document.getElementById('chartOverlay'),
            detailPanel: document.getElementById('detailPanel'),
            // Live popup elements
            liveScanToggle: document.getElementById('liveScanToggle'),
            liveAlertsBadge: document.getElementById('liveAlertsBadge'),
            livePopup: document.getElementById('livePopup'),
            livePopupOverlay: document.getElementById('livePopupOverlay'),
            livePopupCount: document.getElementById('livePopupCount'),
            livePopupList: document.getElementById('livePopupList'),
            showFormingToggle: document.getElementById('showFormingToggle'),
            closePopupBtn: document.getElementById('closePopupBtn'),
        };

        // ──────────────────────────────────────────────────────────
        //  UTILITY FUNCTIONS
        // ──────────────────────────────────────────────────────────

        function showLoading(text) {
            els.loadingText.textContent = text || 'Loading...';
            els.loadingOverlay.classList.add('active');
        }

        function hideLoading() {
            els.loadingOverlay.classList.remove('active');
        }

        function showError(msg) {
            els.errorBanner.textContent = '⚠️ ' + msg;
            els.errorBanner.classList.add('visible');
        }

        function hideError() {
            els.errorBanner.classList.remove('visible');
        }

        function formatPrice(price) {
            if (price == null) return '—';
            return '₹' + Number(price).toLocaleString('en-IN', {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            });
        }

        // ──────────────────────────────────────────────────────────
        //  API CALLS (using fetch + async/await)
        // ──────────────────────────────────────────────────────────

        async function apiGet(endpoint) {
            const response = await fetch(API_BASE + endpoint);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP ${response.status}`);
            }
            return await response.json();
        }

        async function checkHealth() {
            try {
                await apiGet('/api/health');
                hideError();
                return true;
            } catch (e) {
                showError('Backend is not running. Start it with: uvicorn backend.main:app --reload --port 8000');
                return false;
            }
        }

        async function fetchMarketStatus() {
            try {
                const data = await apiGet('/api/market_status');
                const badge = els.marketBadge;
                const statusText = els.marketStatus;

                if (data.is_open) {
                    badge.className = 'market-badge open';
                    statusText.textContent = 'MARKET OPEN';
                } else {
                    badge.className = 'market-badge closed';
                    statusText.textContent = 'MARKET CLOSED';
                }

                badge.title = data.status + ' • ' + data.next_event;
            } catch (e) {
                els.marketStatus.textContent = 'Unknown';
            }
        }

        async function runScan() {
            const pattern = els.patternSelect.value;
            const interval = els.intervalSelect.value;
            const lookback = els.lookbackSelect.value;
            const liveMode = els.liveAlertsToggle.checked;

            showLoading(`Scanning ${pattern === 'all' ? 'all patterns' : pattern} on Nifty stocks...`);

            try {
                const data = await apiGet(
                    `/api/scan?pattern=${pattern}&interval=${interval}&lookback=${lookback}&live_mode=${liveMode}`
                );

                scanResults = data.matches || [];
                renderResults();

                // Update header info
                const scanTime = new Date(data.scan_time);
                els.lastScanInfo.textContent = `Last scan: ${scanTime.toLocaleTimeString()}`;
                els.scanStats.textContent =
                    `${data.total_scanned} scanned • ${scanResults.length} found • ${data.scan_duration_seconds}s` +
                    (data.from_cache ? ' (cached)' : '');

            } catch (e) {
                showError('Scan failed: ' + e.message);
            } finally {
                hideLoading();
            }
        }

        async function loadCandles(ticker, interval, lookback) {
            showLoading(`Loading chart for ${ticker}...`);

            try {
                const data = await apiGet(
                    `/api/candles?ticker=${encodeURIComponent(ticker)}&interval=${interval}&lookback=${lookback}`
                );
                return data;
            } catch (e) {
                showError('Failed to load candles: ' + e.message);
                return null;
            } finally {
                hideLoading();
            }
        }

        // ──────────────────────────────────────────────────────────
        //  RESULTS LIST RENDERING
        // ──────────────────────────────────────────────────────────

        function renderResults() {
            const sorted = sortResults([...scanResults]);
            els.resultsCount.textContent = sorted.length;

            if (sorted.length === 0) {
                els.resultsList.innerHTML = `
                    <div class="empty-state">
                        <div class="icon">📭</div>
                        <div class="text">No patterns found with current settings.<br>Try a different interval or lookback period.</div>
                    </div>`;
                return;
            }

            els.resultsList.innerHTML = sorted.map((result, i) => {
                const icon = PATTERN_ICONS[result.pattern_type] || { emoji: '📊', bg: '#333' };
                const isActive = selectedResult && selectedResult.ticker === result.ticker &&
                    selectedResult.pattern_type === result.pattern_type;
                return `
                    <div class="result-item ${isActive ? 'active' : ''}"
                         onclick="selectResult(${i})" data-index="${i}">
                        <div class="result-icon" style="background:${icon.bg}">${icon.emoji}</div>
                        <div class="result-info">
                            <div class="result-ticker">${result.ticker.replace('.NS', '')}</div>
                            <div class="result-pattern">${result.pattern}</div>
                        </div>
                        <div class="result-score">${result.quality_score}</div>
                    </div>`;
            }).join('');
        }

        function sortResults(results) {
            const sortBy = els.sortSelect.value;
            if (sortBy === 'score') {
                results.sort((a, b) => b.quality_score - a.quality_score);
            } else if (sortBy === 'alpha') {
                results.sort((a, b) => a.ticker.localeCompare(b.ticker));
            } else if (sortBy === 'pattern') {
                results.sort((a, b) => a.pattern.localeCompare(b.pattern));
            }
            return results;
        }

        // ──────────────────────────────────────────────────────────
        //  CHART — TradingView Lightweight Charts
        // ──────────────────────────────────────────────────────────

        function getChartOptions() {
            return {
                layout: {
                    background: { color: isDarkMode ? '#0d1117' : '#ffffff' },
                    textColor: isDarkMode ? '#9ca3af' : '#64748b',
                    fontFamily: 'Inter, sans-serif',
                    fontSize: 11,
                },
                grid: {
                    vertLines: { color: isDarkMode ? 'rgba(99,102,241,0.06)' : 'rgba(99,102,241,0.06)' },
                    horzLines: { color: isDarkMode ? 'rgba(99,102,241,0.06)' : 'rgba(99,102,241,0.06)' },
                },
                crosshair: {
                    mode: 0,
                    vertLine: { color: 'rgba(99,102,241,0.3)', width: 1, style: 2 },
                    horzLine: { color: 'rgba(99,102,241,0.3)', width: 1, style: 2 },
                },
                rightPriceScale: {
                    borderColor: isDarkMode ? 'rgba(99,102,241,0.12)' : 'rgba(99,102,241,0.1)',
                    scaleMargins: { top: 0.1, bottom: 0.25 },
                },
                timeScale: {
                    borderColor: isDarkMode ? 'rgba(99,102,241,0.12)' : 'rgba(99,102,241,0.1)',
                    timeVisible: true,
                    secondsVisible: false,
                },
                handleScroll: true,
                handleScale: true,
            };
        }

        function initChart() {
            if (chart) {
                chart.remove();
            }

            chart = LightweightCharts.createChart(els.chartContainer, getChartOptions());

            // Candlestick series
            candleSeries = chart.addCandlestickSeries({
                upColor: '#22c55e',
                downColor: '#ef4444',
                borderDownColor: '#ef4444',
                borderUpColor: '#22c55e',
                wickDownColor: '#ef4444',
                wickUpColor: '#22c55e',
            });

            // Volume histogram
            volumeSeries = chart.addHistogramSeries({
                color: '#6366f1',
                priceFormat: { type: 'volume' },
                priceScaleId: '',
            });

            volumeSeries.priceScale().applyOptions({
                scaleMargins: { top: 0.85, bottom: 0 },
            });

            // Resize handler
            const resizeObserver = new ResizeObserver(() => {
                if (chart) {
                    chart.applyOptions({
                        width: els.chartContainer.clientWidth,
                        height: els.chartContainer.clientHeight,
                    });
                }
            });
            resizeObserver.observe(els.chartContainer);
        }

        function setChartData(candlesData) {
            if (!chart || !candleSeries || !candlesData || candlesData.length === 0) return;

            // Clear previous price lines
            priceLines.forEach(line => {
                try { candleSeries.removePriceLine(line); } catch (e) { }
            });
            priceLines = [];

            // Set candle data
            const ohlc = candlesData.map(c => ({
                time: c.time,
                open: c.open,
                high: c.high,
                low: c.low,
                close: c.close,
            }));
            candleSeries.setData(ohlc);

            // Set volume data
            const vol = candlesData.map(c => ({
                time: c.time,
                value: c.volume,
                color: c.close >= c.open
                    ? 'rgba(34,197,94,0.25)'
                    : 'rgba(239,68,68,0.25)',
            }));
            volumeSeries.setData(vol);

            // Fit content
            chart.timeScale().fitContent();

            // Hide overlay
            els.chartOverlay.style.display = 'none';
        }

        function addPatternMarkers(keyLevels, candlesData) {
            if (!candleSeries || !keyLevels || keyLevels.length === 0) return;

            // Clear old price lines
            priceLines.forEach(line => {
                try { candleSeries.removePriceLine(line); } catch (e) { }
            });
            priceLines = [];

            // Add horizontal price lines for each key level
            keyLevels.forEach(level => {
                if (level.price == null) return;

                const priceLine = candleSeries.createPriceLine({
                    price: level.price,
                    color: level.color,
                    lineWidth: 1,
                    lineStyle: 2, // Dashed
                    axisLabelVisible: true,
                    title: `${level.label} ₹${Number(level.price).toLocaleString('en-IN')}`,
                });
                priceLines.push(priceLine);
            });

            // Add markers at key dates on the candlestick series
            const markers = [];
            keyLevels.forEach(level => {
                if (!level.date) return;

                // Find the candle time matching this date
                const matchingCandle = candlesData.find(c => {
                    if (typeof c.time === 'string') {
                        return level.date.startsWith(c.time);
                    } else {
                        // c.time is unix timestamp (UTC-aligned naive time). Reconstruct string:
                        const d = new Date(c.time * 1000);
                        const iso = d.toISOString(); // e.g. "2024-05-15T10:15:00.000Z"
                        const str = iso.replace('T', ' ').substring(0, 16); // "2024-05-15 10:15"
                        return level.date.startsWith(str);
                    }
                });

                if (matchingCandle) {
                    markers.push({
                        time: matchingCandle.time,
                        position: level.price > (candlesData[Math.floor(candlesData.length / 2)]?.close || 0) ? 'aboveBar' : 'belowBar',
                        color: level.color,
                        shape: 'circle',
                        text: level.label,
                    });
                }
            });

            if (markers.length > 0) {
                // Sort markers by time (required by the library)
                markers.sort((a, b) => {
                    if (typeof a.time === 'string' && typeof b.time === 'string') {
                        return a.time.localeCompare(b.time);
                    }
                    return a.time - b.time;
                });
                candleSeries.setMarkers(markers);
            }
        }

        // ──────────────────────────────────────────────────────────
        //  DETAIL PANEL
        // ──────────────────────────────────────────────────────────

        function renderDetailPanel(result) {
            const panel = els.detailPanel;
            panel.classList.add('visible');

            // Determine signal class
            const signal = result.signal || '';
            let signalClass = 'neutral';
            if (signal.toLowerCase().includes('bullish')) signalClass = 'bullish';
            if (signal.toLowerCase().includes('bearish')) signalClass = 'bearish';

            // Build checks HTML
            const checksHTML = (result.checks || []).map(check => {
                const statusClass = check.status === 'PASS' ? 'pass' :
                    check.status === 'FAIL' ? 'fail' : 'na';
                const icon = check.status === 'PASS' ? '✅' :
                    check.status === 'FAIL' ? '❌' : '➖';
                return `<span class="check-badge ${statusClass}" title="${check.detail || ''}">${icon} ${check.name}</span>`;
            }).join('');

            // Build key levels HTML
            const levelsHTML = (result.key_levels || []).map(level => {
                return `
                    <div class="key-level">
                        <div class="key-level-dot" style="background:${level.color}"></div>
                        <span class="key-level-label">${level.label}:</span>
                        <span class="key-level-price">${formatPrice(level.price)}</span>
                    </div>`;
            }).join('');

            // Quality score bar (all patterns use 0-10 scale)
            const maxScore = 10;
            const scorePct = Math.min(100, (result.quality_score / maxScore) * 100);

            // Pattern description
            const description = PATTERN_DESCRIPTIONS[result.pattern_type] || '';

            panel.innerHTML = `
                <div class="detail-header">
                    <div class="detail-title">${result.ticker.replace('.NS', '')} — ${result.pattern}</div>
                    <span class="detail-signal ${signalClass}">${signal}</span>
                </div>

                <div class="quality-bar-container">
                    <span class="quality-bar-label">Quality Score</span>
                    <div class="quality-bar-track">
                        <div class="quality-bar-fill" style="width:${scorePct}%"></div>
                    </div>
                    <span class="quality-bar-value">${result.quality_score}</span>
                </div>

                <div class="checks-grid">${checksHTML}</div>

                <div class="key-levels">${levelsHTML}</div>

                ${description ? `<div class="pattern-description">💡 <strong>What this means:</strong> ${description}</div>` : ''}
            `;
        }

        // ──────────────────────────────────────────────────────────
        //  EVENT HANDLERS
        // ──────────────────────────────────────────────────────────

        function clearChartState() {
            if (!candleSeries) return;

            console.log(`Clearing markers for previous selection`);

            // 1. Clear markers
            candleSeries.setMarkers([]);

            // 2. Remove price lines
            priceLines.forEach(line => {
                try { candleSeries.removePriceLine(line); } catch (e) { }
            });
            priceLines = [];

            // 3. Remove shaded regions (if any were added in the future, they would be cleared here)
        }

        async function selectResult(index) {
            const sorted = sortResults([...scanResults]);
            const result = sorted[index];
            if (!result) return;

            selectedResult = result;

            // Highlight active item
            document.querySelectorAll('.result-item').forEach((el, i) => {
                el.classList.toggle('active', i === index);
            });

            // Explicitly clear ALL previous chart state FIRST before loading new data
            clearChartState();

            // Load candles
            const interval = els.intervalSelect.value;
            const lookback = els.lookbackSelect.value;
            const data = await loadCandles(result.ticker, interval, lookback);

            // RACE CONDITION FIX: Discard response if user clicked another result while this fetch was in-flight
            if (selectedResult !== result) {
                console.warn(`Discarding stale response for ${result.ticker} (User clicked a different result)`);
                return;
            }

            if (data && data.candles && data.candles.length > 0) {
                setChartData(data.candles);
                console.log(`Setting markers for: ${result.pattern_type}-${result.ticker}`);
                addPatternMarkers(result.key_levels, data.candles);
            }

            // Show detail panel
            renderDetailPanel(result);
        }

        // Make selectResult globally accessible for onclick
        window.selectResult = selectResult;

        // ──────────────────────────────────────────────────────────
        //  THEME TOGGLE
        // ──────────────────────────────────────────────────────────

        function toggleTheme() {
            isDarkMode = !isDarkMode;
            document.body.classList.toggle('light-mode', !isDarkMode);
            els.themeToggle.textContent = isDarkMode ? '🌙' : '☀️';

            // Re-apply chart theme
            if (chart) {
                chart.applyOptions(getChartOptions());
            }
        }

        // ──────────────────────────────────────────────────────────
        //  LIVE SCAN POPUP
        // ──────────────────────────────────────────────────────────

        let liveScanIntervalId = null;

        function showToast(title, message) {
            const container = document.getElementById('toastContainer');
            const toast = document.createElement('div');
            toast.className = 'toast';
            toast.innerHTML = `
                <div class="toast-header">
                    <div class="toast-title">${title}</div>
                    <div class="toast-close" onclick="this.parentElement.parentElement.classList.add('hiding'); setTimeout(() => this.parentElement.parentElement.remove(), 400)">&times;</div>
                </div>
                <div class="toast-body">${message}</div>
            `;
            container.appendChild(toast);

            // Auto remove after 6 seconds
            setTimeout(() => {
                if (toast.parentElement) {
                    toast.classList.add('hiding');
                    setTimeout(() => toast.remove(), 400);
                }
            }, 6000);
        }

        function toggleLivePopup() {
            livePopupOpen = !livePopupOpen;
            els.livePopup.classList.toggle('hidden', !livePopupOpen);
            els.livePopupOverlay.classList.toggle('hidden', !livePopupOpen);
        }

        function closeLivePopup() {
            livePopupOpen = false;
            els.livePopup.classList.add('hidden');
            els.livePopupOverlay.classList.add('hidden');
        }

        /**
         * Standalone rendering function for the live popup.
         * This is completely separate from renderResults() (used for backtest).
         * Sorts: Confirmed first, then Forming, within each group by quality score desc.
         */
        function renderLivePopupResults() {
            const showForming = els.showFormingToggle.checked;

            // Filter based on toggle
            let filtered = [...liveResults];
            if (!showForming) {
                filtered = filtered.filter(r => r.status !== 'forming');
            }

            // Sort: confirmed first, then forming, within each group by quality desc
            filtered.sort((a, b) => {
                const statusOrder = { 'confirmed': 0, 'forming': 1 };
                const sA = statusOrder[a.status] ?? 2;
                const sB = statusOrder[b.status] ?? 2;
                if (sA !== sB) return sA - sB;
                return (b.quality_score || 0) - (a.quality_score || 0);
            });

            // Update count
            els.livePopupCount.textContent = filtered.length;

            // Update badge
            const badgeCount = liveResults.length;
            els.liveAlertsBadge.textContent = badgeCount;
            els.liveAlertsBadge.classList.toggle('has-alerts', badgeCount > 0);

            if (filtered.length === 0) {
                els.livePopupList.innerHTML = `
                    <div class="empty-state">
                        <div class="icon">📭</div>
                        <div class="text">${showForming ? 'No live results yet. Run a live scan.' : 'No confirmed patterns in live results.<br>Toggle "Show Forming" to see forming patterns.'}</div>
                    </div>`;
                return;
            }

            // Group into confirmed and forming sections
            const confirmed = filtered.filter(r => r.status === 'confirmed');
            const forming = filtered.filter(r => r.status === 'forming');

            let html = '';

            if (confirmed.length > 0) {
                html += `<div class="live-popup-separator">✅ Confirmed (${confirmed.length})</div>`;
                html += confirmed.map(r => buildLiveResultItem(r)).join('');
            }

            if (forming.length > 0) {
                html += `<div class="live-popup-separator">⏳ Forming (${forming.length})</div>`;
                html += forming.map(r => buildLiveResultItem(r)).join('');
            }

            els.livePopupList.innerHTML = html;
        }

        function buildLiveResultItem(result) {
            const icon = PATTERN_ICONS[result.pattern_type] || { emoji: '📊', bg: '#333' };
            const status = result.status || 'confirmed';
            const statusLabel = status === 'forming' ? '⏳ Forming' : '✅ Confirmed';
            const statusClass = status === 'forming' ? 'forming' : 'confirmed';

            return `
                <div class="live-result-item status-${statusClass}"
                     onclick="selectLiveResult('${result.ticker}', '${result.pattern_type}')">
                    <div class="result-icon" style="background:${icon.bg}">${icon.emoji}</div>
                    <div class="live-result-info">
                        <div class="live-result-ticker">${result.ticker.replace('.NS', '')}</div>
                        <div class="live-result-pattern">${result.pattern}</div>
                    </div>
                    <div class="live-result-meta">
                        <div class="live-result-score">${result.quality_score}</div>
                        <span class="status-badge ${statusClass}">${statusLabel}</span>
                    </div>
                </div>`;
        }

        async function selectLiveResult(ticker, patternType) {
            // Find the result in liveResults
            const result = liveResults.find(r => r.ticker === ticker && r.pattern_type === patternType);
            if (!result) return;

            selectedResult = result;

            // Close popup
            closeLivePopup();

            // Clear previous chart state
            clearChartState();

            // Load candles
            const interval = els.intervalSelect.value;
            const lookback = els.lookbackSelect.value;
            const data = await loadCandles(result.ticker, interval, lookback);

            if (selectedResult !== result) return;

            if (data && data.candles && data.candles.length > 0) {
                setChartData(data.candles);
                addPatternMarkers(result.key_levels, data.candles);
            }

            renderDetailPanel(result);
        }

        window.selectLiveResult = selectLiveResult;

        async function runLiveScan() {
            if (!els.liveAlertsToggle.checked) return;

            const patterns = els.patternSelect.value;
            const interval = els.intervalSelect.value;
            const lookback = els.lookbackSelect.value;

            try {
                const response = await fetch(`${API_BASE}/api/live_scan?patterns=${patterns}&interval=${interval}&lookback=${lookback}`);
                if (!response.ok) throw new Error('Live scan failed');

                const data = await response.json();
                if (data.alerts && data.alerts.length > 0) {
                    // Merge new alerts into liveResults (dedup by ticker + pattern_type)
                    data.alerts.forEach(alert => {
                        const existsIdx = liveResults.findIndex(
                            r => r.ticker === alert.ticker && r.pattern_type === alert.pattern_type
                        );
                        if (existsIdx >= 0) {
                            liveResults[existsIdx] = alert; // Update existing
                        } else {
                            liveResults.push(alert);
                        }
                    });

                    renderLivePopupResults();

                    // Show toast for genuinely new alerts
                    const newCount = data.alerts.length;
                    if (newCount > 0) {
                        showToast('🔔 Live Scan', `${newCount} pattern${newCount > 1 ? 's' : ''} detected`);
                    }
                }
            } catch (err) {
                console.error("Live scan polling error:", err);
            }
        }

        function toggleLiveAlerts() {
            if (els.liveAlertsToggle.checked) {
                showToast("Live Alerts ON", "Scanning for new patterns every minute.");
                els.liveScanToggle.classList.add('active');
                els.liveScanToggle.innerHTML = '🟢 Live Scan <span class="live-badge" id="liveAlertsBadge">' + liveResults.length + '</span>';
                els.liveAlertsBadge = document.getElementById('liveAlertsBadge');
                // Run one immediately, then every 60 seconds
                runLiveScan();
                liveScanIntervalId = setInterval(runLiveScan, 60000);
            } else {
                showToast("Live Alerts OFF", "Background polling stopped.");
                els.liveScanToggle.classList.remove('active');
                els.liveScanToggle.innerHTML = '🔴 Live Scan <span class="live-badge" id="liveAlertsBadge">' + liveResults.length + '</span>';
                els.liveAlertsBadge = document.getElementById('liveAlertsBadge');
                if (liveScanIntervalId) {
                    clearInterval(liveScanIntervalId);
                    liveScanIntervalId = null;
                }
            }
        }

        // ──────────────────────────────────────────────────────────
        //  INITIALIZATION
        // ──────────────────────────────────────────────────────────

        async function init() {
            // Attach event listeners
            els.scanBtn.addEventListener('click', runScan);
            els.refreshBtn.addEventListener('click', runScan);
            els.themeToggle.addEventListener('click', toggleTheme);
            els.sortSelect.addEventListener('change', renderResults);
            els.liveAlertsToggle.addEventListener('change', toggleLiveAlerts);
            els.liveScanToggle.addEventListener('click', toggleLivePopup);
            els.closePopupBtn.addEventListener('click', closeLivePopup);
            els.livePopupOverlay.addEventListener('click', closeLivePopup);
            els.showFormingToggle.addEventListener('change', renderLivePopupResults);

            // Initialize chart
            initChart();

            // Check backend health
            const isHealthy = await checkHealth();
            if (!isHealthy) return;

            // Fetch market status
            await fetchMarketStatus();

            // Load live alerts history from API and populate the popup
            try {
                const response = await fetch(`${API_BASE}/api/live_alerts`);
                if (response.ok) {
                    const data = await response.json();
                    if (data.alerts && data.alerts.length > 0) {
                        liveResults = data.alerts;
                        renderLivePopupResults();
                    }
                }
            } catch (e) {
                console.error("Failed to load live alerts history", e);
            }

            // Auto-refresh market status every 60 seconds
            setInterval(fetchMarketStatus, 60000);

            // Auto-run default scan
            await runScan();
        }

        // Start the app when the page loads
        init();
