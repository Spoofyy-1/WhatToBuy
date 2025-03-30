// Chart management
const charts = {};

// Load charts for all stock elements on page
document.addEventListener('DOMContentLoaded', function() {
    console.log("Initializing charts");
    
    // Find all chart canvases
    const chartCanvases = document.querySelectorAll('canvas[id^="chart-"]');
    
    // Initialize each chart
    chartCanvases.forEach(canvas => {
        const symbol = canvas.id.replace('chart-', '');
        console.log(`Initializing chart for ${symbol}`);
        loadStockChart(symbol);
    });
});

/**
 * Load stock chart data and render
 * @param {string} symbol - Stock symbol
 */
function loadStockChart(symbol) {
    // Show loading indicator
    const loadingEl = document.getElementById(`chart-loading-${symbol}`);
    const errorEl = document.getElementById(`chart-error-${symbol}`);
    const canvas = document.getElementById(`chart-${symbol}`);
    
    if (!canvas) {
        console.error(`Canvas element for ${symbol} not found`);
        return;
    }
    
    if (loadingEl) loadingEl.classList.remove('d-none');
    if (errorEl) errorEl.classList.add('d-none');
    
    console.log(`Fetching chart data for ${symbol}`);
    
    // Fetch stock data with retry mechanism
    fetchWithRetry(`/api/stock-data/${symbol}?days=60`, 3)
        .then(data => {
            console.log(`Received chart data for ${symbol}:`, data);
            
            if (loadingEl) loadingEl.classList.add('d-none');
            
            if (data.status === 'success' && data.data && data.data.length > 0) {
                renderStockChart(symbol, data.data);
            } else {
                console.error(`Error or empty data for ${symbol}:`, data.message || 'No data returned');
                if (errorEl) {
                    errorEl.textContent = data.message || 'No data available for this stock';
                    errorEl.classList.remove('d-none');
                }
            }
        })
        .catch(error => {
            console.error(`Failed to load chart for ${symbol}:`, error);
            if (loadingEl) loadingEl.classList.add('d-none');
            if (errorEl) {
                errorEl.textContent = `Failed to load chart: ${error.message}`;
                errorEl.classList.remove('d-none');
            }
        });
}

/**
 * Fetch with retry mechanism
 * @param {string} url - API URL
 * @param {number} retries - Number of retries
 * @returns {Promise<Object>} - JSON response
 */
function fetchWithRetry(url, retries = 3) {
    return new Promise((resolve, reject) => {
        const attemptFetch = (attemptsLeft) => {
            fetch(url)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    return response.json();
                })
                .then(resolve)
                .catch(error => {
                    console.warn(`Fetch attempt failed (${attemptsLeft} retries left):`, error);
                    if (attemptsLeft > 0) {
                        setTimeout(() => attemptFetch(attemptsLeft - 1), 1000);
                    } else {
                        reject(error);
                    }
                });
        };
        
        attemptFetch(retries);
    });
}

/**
 * Render stock chart
 * @param {string} symbol - Stock symbol
 * @param {Array} data - Chart data
 */
function renderStockChart(symbol, data) {
    const canvas = document.getElementById(`chart-${symbol}`);
    const ctx = canvas.getContext('2d');
    
    // If we already have a chart for this symbol, destroy it first
    if (charts[symbol]) {
        charts[symbol].destroy();
    }
    
    // Prepare chart data
    const chartData = prepareChartData(data);
    
    // Create new chart
    charts[symbol] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.dates,
            datasets: [
                {
                    label: `${symbol} Price`,
                    data: chartData.prices,
                    borderColor: chartData.priceChange >= 0 ? 'rgba(40, 167, 69, 1)' : 'rgba(220, 53, 69, 1)',
                    backgroundColor: chartData.priceChange >= 0 ? 'rgba(40, 167, 69, 0.1)' : 'rgba(220, 53, 69, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1
                },
                {
                    label: 'SMA 20',
                    data: chartData.sma20,
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += new Intl.NumberFormat('en-US', { 
                                    style: 'currency', 
                                    currency: 'USD' 
                                }).format(context.parsed.y);
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        maxRotation: 0,
                        maxTicksLimit: 8
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toFixed(2);
                        }
                    }
                }
            }
        }
    });
    
    console.log(`Chart for ${symbol} rendered successfully`);
}

/**
 * Prepare chart data
 * @param {Array} data - Raw chart data
 * @returns {Object} - Prepared chart data
 */
function prepareChartData(data) {
    const dates = data.map(item => item.date);
    const prices = data.map(item => item.close);
    const sma20 = data.map(item => item.sma20);
    
    const priceChange = prices[prices.length - 1] - prices[0];
    
    return {
        dates,
        prices,
        sma20,
        priceChange
    };
} 