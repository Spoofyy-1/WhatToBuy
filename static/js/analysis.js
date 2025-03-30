const chartInstances = {};

document.addEventListener('DOMContentLoaded', function() {
    console.log("Document loaded, initializing analysis functions");
    
    // Initialize dropdowns
    initializeDropdowns();
    
    // Calculate portfolio projections with a slight delay to ensure all elements are loaded
    setTimeout(calculatePortfolioProjections, 500);
    
    // Initialize execute trade buttons
    document.querySelectorAll('.execute-trade-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const symbol = this.dataset.symbol;
            const price = parseFloat(this.dataset.price);
            const allocation = parseFloat(this.dataset.allocation);
            const shares = Math.floor(allocation / price * 100) / 100; // Round to 2 decimal places
            
            // Set trade modal values
            document.getElementById('trade-symbol').value = symbol;
            document.getElementById('trade-display-symbol').value = symbol;
            document.getElementById('trade-price').value = `$${price.toFixed(2)}`;
            document.getElementById('trade-allocation').value = `$${allocation.toFixed(2)}`;
            document.getElementById('trade-shares').value = shares.toFixed(2);
            
            // Calculate stop loss and take profit
            const stopLossPercent = parseFloat(document.getElementById('stop_loss').value) / 100;
            const takeProfitPercent = parseFloat(document.getElementById('take_profit').value) / 100;
            
            const stopLossPrice = price * (1 - stopLossPercent);
            const takeProfitPrice = price * (1 + takeProfitPercent);
            
            document.getElementById('trade-stop-loss').value = `$${stopLossPrice.toFixed(2)}`;
            document.getElementById('trade-take-profit').value = `$${takeProfitPrice.toFixed(2)}`;
            
            // Show modal
            const tradeModal = new bootstrap.Modal(document.getElementById('tradeModal'));
            tradeModal.show();
        });
    });
    
    // Handle trade confirmation
    const confirmTradeBtn = document.getElementById('confirm-trade-btn');
    if (confirmTradeBtn) {
        confirmTradeBtn.addEventListener('click', function() {
            const symbol = document.getElementById('trade-symbol').value;
            const priceStr = document.getElementById('trade-price').value;
            const sharesStr = document.getElementById('trade-shares').value;
            const allocationStr = document.getElementById('trade-allocation').value;
            const stopLossStr = document.getElementById('trade-stop-loss').value;
            const takeProfitStr = document.getElementById('trade-take-profit').value;
            const notes = document.getElementById('trade-notes').value;
            
            const price = parseFloat(priceStr.replace('$', ''));
            const shares = parseFloat(sharesStr);
            const allocation = parseFloat(allocationStr.replace('$', ''));
            const stopLoss = parseFloat(stopLossStr.replace('$', ''));
            const takeProfit = parseFloat(takeProfitStr.replace('$', ''));
            
            // Create trade data
            const tradeData = {
                symbol: symbol,
                date: document.getElementById('date')?.value || new Date().toISOString().split('T')[0],
                price: price,
                shares: shares,
                action: 'BUY',
                investment: allocation,
                stop_loss: stopLoss,
                take_profit: takeProfit,
                notes: notes
            };
            
            // Send to server
            fetch('/execute-trade', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(tradeData),
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    // Close modal
                    bootstrap.Modal.getInstance(document.getElementById('tradeModal')).hide();
                    
                    // Show success message
                    const alertDiv = document.createElement('div');
                    alertDiv.className = 'alert alert-success alert-dismissible fade show';
                    alertDiv.role = 'alert';
                    alertDiv.innerHTML = `
                        <i class="bi bi-check-circle me-2"></i>
                        Successfully executed trade for ${shares} shares of ${symbol}
                        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                    `;
                    
                    document.querySelector('.container-fluid').prepend(alertDiv);
                    
                    // Redirect to portfolio page after 2 seconds
                    setTimeout(() => {
                        window.location.href = '/portfolio';
                    }, 2000);
                } else {
                    alert('Error executing trade: ' + data.message);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error executing trade');
            });
        });
    }
    
    // Add debug info at the end
    addDebugInfo();

    // Add a debug button to each chart container
    document.querySelectorAll('[id^="chart-container-"]').forEach(container => {
        const symbol = container.id.replace('chart-container-', '');
        const debugBtn = document.createElement('button');
        debugBtn.className = 'btn btn-sm btn-outline-secondary mt-2';
        debugBtn.innerHTML = '<i class="bi bi-bug"></i> Debug Chart';
        debugBtn.onclick = function() {
            container.innerHTML = debugChartIssue(symbol);
        };
        container.appendChild(debugBtn);
    });

    // Find all analysis rows and add verification buttons
    document.querySelectorAll('.analysis-row').forEach(row => {
        const symbol = row.dataset.symbol;
        const date = document.getElementById('date')?.value || getAnalysisDate();
        
        // Add a verify button to each row
        const actionsCell = row.querySelector('td:last-child');
        if (actionsCell) {
            const verifyBtn = document.createElement('button');
            verifyBtn.className = 'btn btn-sm btn-outline-info ms-1';
            verifyBtn.innerHTML = '<i class="bi bi-check-circle"></i> Verify Price';
            verifyBtn.onclick = function(e) {
                e.preventDefault();
                e.stopPropagation();
                verifyStockPrice(symbol, date);
            };
            actionsCell.appendChild(verifyBtn);
        }
    });

    // Add tooltips to price cells
    document.querySelectorAll('.price-cell').forEach(cell => {
        const symbol = cell.closest('tr').dataset.symbol;
        const price = cell.textContent.trim().replace('Purchase Price', '').trim();
        
        // Add tooltip element
        const tooltip = document.createElement('span');
        tooltip.className = 'price-tooltip';
        tooltip.innerHTML = `<i class="bi bi-info-circle-fill text-info ms-1"></i>`;
        tooltip.title = `This is the price of ${symbol} on the analysis date. This is the price at which the stock would have been purchased.`;
        
        cell.appendChild(tooltip);
    });
});

function loadAnalysisChart(symbol) {
    console.log(`Loading chart for ${symbol}`);
    
    // Get chart container
    const chartContainer = document.getElementById(`chart-container-${symbol}`);
    if (!chartContainer) {
        console.error(`Chart container not found for ${symbol}`);
        return;
    }
    
    // Show loading indicator
    chartContainer.innerHTML = `
        <div class="text-center py-3">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-2">Loading chart data...</p>
        </div>
    `;
    
    // Clear any existing chart instance
    if (chartInstances && chartInstances[symbol]) {
        chartInstances[symbol].destroy();
        delete chartInstances[symbol];
    }
    
    // Get date from the analysis
    const analysisDate = getAnalysisDate();
    console.log(`Analysis date: ${analysisDate}`);
    
    // Calculate date range - look back 90 days from analysis date to include sentiment data
    const startDate = new Date(analysisDate);
    startDate.setDate(startDate.getDate() - 90); // 90 days back to match sentiment history
    
    const endDate = new Date(analysisDate);
    endDate.setDate(endDate.getDate() + 7); // 7 days forward for analysis
    
    // Format dates for API
    const startDateStr = startDate.toISOString().split('T')[0];
    const endDateStr = endDate.toISOString().split('T')[0];
    
    // Add sentiment parameter to API request to ensure we get sentiment data
    fetch(`/api/stock-data/${symbol}?start_date=${startDateStr}&end_date=${endDateStr}&include_sentiment=true`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log(`Received ${data.length} data points for ${symbol}`);
            
            if (data.error) {
                chartContainer.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                return;
            }
            
            if (!data || data.length === 0) {
                chartContainer.innerHTML = `<div class="alert alert-warning">No data available for this time period</div>`;
                return;
            }
            
            // Clear loading indicator and prepare canvas
            chartContainer.innerHTML = `<canvas id="chart-${symbol}" width="100%" height="300"></canvas>`;
            
            // Create the chart
            createAnalysisChart(symbol, data, analysisDate);
        })
        .catch(error => {
            console.error(`Error loading chart for ${symbol}:`, error);
            chartContainer.innerHTML = `
                <div class="alert alert-danger">
                    <strong>Error loading chart:</strong> ${error.message}
                    <button class="btn btn-sm btn-outline-danger float-end" onclick="loadAnalysisChart('${symbol}')">
                        <i class="bi bi-arrow-clockwise"></i> Retry
                    </button>
                </div>
            `;
        });
}

function createAnalysisChart(symbol, data, analysisDate) {
    console.log(`Creating chart for ${symbol} with ${data.length} data points`);
    
    const canvas = document.getElementById(`chart-${symbol}`);
    if (!canvas) {
        console.error(`Chart canvas not found for ${symbol}`);
        return;
    }
    
    const ctx = canvas.getContext('2d');
    
    // Parse dates and sort data chronologically
    data.forEach(item => {
        item.dateObj = new Date(item.date);
    });
    data.sort((a, b) => a.dateObj - b.dateObj);
    
    // Format data for candlestick chart
    const candlestickData = data.map(item => ({
        x: new Date(item.date),
        o: parseFloat(item.open || item.price),
        h: parseFloat(item.high || item.price),
        l: parseFloat(item.low || item.price),
        c: parseFloat(item.close || item.price)
    }));
    
    // Extract data for additional technical indicators
    const dates = data.map(item => new Date(item.date));
    const rsiValues = data.map(item => item.rsi || null);
    const macdValues = data.map(item => item.macd || null);
    const macdSignalValues = data.map(item => item.macd_signal || null);
    const sma20Values = data.map(item => item.sma20 || null);
    const sma50Values = data.map(item => item.sma50 || null);
    
    // Extract sentiment data if available
    const sentimentData = data.map(item => ({
        x: new Date(item.date),
        y: item.sentiment_score || null
    })).filter(item => item.y !== null);
    
    // Determine if we have sentiment data to display
    const hasSentimentData = sentimentData.length > 0;
    
    // Create chart
    const chartConfig = {
        type: 'candlestick',
        data: {
            datasets: [
                {
                    label: `${symbol} Price`,
                    data: candlestickData,
                    color: {
                        up: 'rgba(40, 167, 69, 1)',
                        down: 'rgba(220, 53, 69, 1)',
                        unchanged: 'rgba(54, 162, 235, 1)',
                    },
                    borderWidth: 1,
                    borderColor: {
                        up: 'rgba(40, 167, 69, 1)',
                        down: 'rgba(220, 53, 69, 1)',
                        unchanged: 'rgba(54, 162, 235, 1)',
                    },
                    yAxisID: 'y'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            const label = context.dataset.label || '';
                            
                            if (label.includes('Price')) {
                                const item = context.raw;
                                if (item) {
                                    return [
                                        `Open: $${item.o.toFixed(2)}`,
                                        `High: $${item.h.toFixed(2)}`,
                                        `Low: $${item.l.toFixed(2)}`,
                                        `Close: $${item.c.toFixed(2)}`
                                    ];
                                }
                            } else if (label.includes('Sentiment')) {
                                const value = context.parsed.y;
                                if (value !== null) {
                                    // Format sentiment score nicely
                                    let sentimentDesc = "Neutral";
                                    if (value > 0.2) sentimentDesc = "Very Positive";
                                    else if (value > 0.05) sentimentDesc = "Positive";
                                    else if (value < -0.2) sentimentDesc = "Very Negative";
                                    else if (value < -0.05) sentimentDesc = "Negative";
                                    
                                    return [`Sentiment: ${value.toFixed(2)} (${sentimentDesc})`];
                                }
                            }
                            return [];
                        }
                    }
                },
                title: {
                    display: true,
                    text: `${symbol} Price Chart`,
                    font: {
                        size: 16
                    }
                },
                legend: {
                    display: hasSentimentData,
                    position: 'top'
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day'
                    },
                    adapters: {
                        date: {
                            locale: 'en'
                        }
                    },
                    ticks: {
                        source: 'auto',
                        maxRotation: 0,
                        autoSkip: true
                    }
                },
                y: {
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Price ($)'
                    }
                }
            }
        }
    };
    
    // Add sentiment dataset if available
    if (hasSentimentData) {
        // Add secondary y-axis for sentiment
        chartConfig.options.scales.y1 = {
            position: 'left',
            title: {
                display: true,
                text: 'Sentiment Score'
            },
            min: -1,
            max: 1,
            grid: {
                drawOnChartArea: false
            }
        };
        
        // Add sentiment dataset
        chartConfig.data.datasets.push({
            label: 'Sentiment Score',
            data: sentimentData,
            type: 'line',
            yAxisID: 'y1',
            fill: false,
            pointRadius: 3,
            pointHoverRadius: 5,
            borderColor: 'rgba(153, 102, 255, 1)',
            backgroundColor: 'rgba(153, 102, 255, 0.5)',
            borderWidth: 2,
            tension: 0.1
        });
        
        // Add SMA20 and SMA50 indicators if available
        const validSma20 = sma20Values.some(val => val !== null);
        const validSma50 = sma50Values.some(val => val !== null);
        
        if (validSma20) {
            chartConfig.data.datasets.push({
                label: '20-Day MA',
                data: dates.map((date, i) => ({
                    x: date,
                    y: sma20Values[i]
                })),
                type: 'line',
                yAxisID: 'y',
                borderColor: 'rgba(0, 123, 255, 1)',
                backgroundColor: 'rgba(0, 123, 255, 0.1)',
                borderWidth: 1.5,
                pointRadius: 0,
                fill: false
            });
        }
        
        if (validSma50) {
            chartConfig.data.datasets.push({
                label: '50-Day MA',
                data: dates.map((date, i) => ({
                    x: date,
                    y: sma50Values[i]
                })),
                type: 'line',
                yAxisID: 'y',
                borderColor: 'rgba(255, 193, 7, 1)',
                backgroundColor: 'rgba(255, 193, 7, 0.1)',
                borderWidth: 1.5,
                pointRadius: 0,
                fill: false
            });
        }
    }
    
    // Store chart instance for later reference
    if (!chartInstances) {
        chartInstances = {};
    }
    chartInstances[symbol] = new Chart(ctx, chartConfig);
    
    // Add technical indicators to the sidebar
    updateTechnicalIndicatorsSidebar(symbol, data[data.length - 1]);
    
    console.log(`Chart created for ${symbol}`);
}

// New function to update the technical indicators sidebar
function updateTechnicalIndicatorsSidebar(symbol, latestData) {
    const container = document.querySelector(`#details-${symbol} .technical-indicators`);
    if (!container) return;
    
    const rsi = latestData.rsi || 'N/A';
    const macd = latestData.macd || 'N/A';
    const macdSignal = latestData.macd_signal || 'N/A';
    const sma20 = latestData.sma20 || 'N/A';
    const sma50 = latestData.sma50 || 'N/A';
    const volume = latestData.volume || 'N/A';
    
    // Get sentiment data if available
    const sentimentScore = latestData.sentiment_score || 'N/A';
    const sentimentPrediction = latestData.sentiment_prediction || 'neutral';
    const sentimentConfidence = latestData.sentiment_confidence || 0;
    
    // Determine sentiment indicator classes
    let rsiClass = 'bg-secondary';
    if (rsi !== 'N/A') {
        if (rsi > 70) rsiClass = 'bg-danger';
        else if (rsi < 30) rsiClass = 'bg-success';
    }
    
    let macdClass = 'bg-secondary';
    if (macd !== 'N/A' && macdSignal !== 'N/A') {
        macdClass = macd > macdSignal ? 'bg-success' : 'bg-danger';
    }
    
    let sentimentClass = 'bg-secondary';
    if (sentimentPrediction === 'up') sentimentClass = 'bg-success';
    else if (sentimentPrediction === 'down') sentimentClass = 'bg-danger';
    
    container.innerHTML = `
        <div class="card mb-3">
            <div class="card-body">
                <h6 class="card-title">Technical Indicators</h6>
                <div class="row mb-2">
                    <div class="col-4 text-end fw-bold">RSI:</div>
                    <div class="col-8">
                        <span class="indicator-badge ${rsiClass}">${typeof rsi === 'number' ? rsi.toFixed(2) : rsi}</span>
                        ${rsi > 70 ? '<small class="text-danger">Overbought</small>' : rsi < 30 ? '<small class="text-success">Oversold</small>' : ''}
                    </div>
                </div>

                <div class="row mb-2">
                    <div class="col-4 text-end fw-bold">MACD:</div>
                    <div class="col-8">
                        <span class="indicator-badge ${macdClass}">${typeof macd === 'number' ? macd.toFixed(2) : macd}</span>
                    </div>
                </div>

                <div class="row mb-2">
                    <div class="col-4 text-end fw-bold">SMA 20:</div>
                    <div class="col-8">
                        <span class="indicator-badge bg-info">${typeof sma20 === 'number' ? '$' + sma20.toFixed(2) : sma20}</span>
                    </div>
                </div>

                <div class="row mb-2">
                    <div class="col-4 text-end fw-bold">SMA 50:</div>
                    <div class="col-8">
                        <span class="indicator-badge bg-primary">${typeof sma50 === 'number' ? '$' + sma50.toFixed(2) : sma50}</span>
                    </div>
                </div>

                <div class="row mb-2">
                    <div class="col-4 text-end fw-bold">Volume:</div>
                    <div class="col-8">
                        ${typeof volume === 'number' ? volume.toLocaleString() : volume}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card mb-3">
            <div class="card-body">
                <h6 class="card-title">Sentiment Analysis</h6>
                <div class="row mb-2">
                    <div class="col-4 text-end fw-bold">Score:</div>
                    <div class="col-8">
                        <span class="indicator-badge ${sentimentClass}">
                            ${typeof sentimentScore === 'number' ? sentimentScore.toFixed(2) : sentimentScore}
                        </span>
                    </div>
                </div>
                
                <div class="row mb-2">
                    <div class="col-4 text-end fw-bold">Prediction:</div>
                    <div class="col-8">
                        <span class="badge ${sentimentPrediction === 'up' ? 'bg-success' : sentimentPrediction === 'down' ? 'bg-danger' : 'bg-secondary'}">
                            ${sentimentPrediction.toUpperCase()}
                        </span>
                        <div class="progress mt-1" style="height: 5px">
                            <div class="progress-bar ${sentimentPrediction === 'up' ? 'bg-success' : 'bg-danger'}" 
                                style="width: ${sentimentConfidence}%"></div>
                        </div>
                        <small class="text-muted">${sentimentConfidence.toFixed(1)}% confidence</small>
                    </div>
                </div>
                
                <div class="row mb-2">
                    <div class="col-12 text-center mt-2">
                        <small class="text-muted">Based on 90-day historical correlation</small>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function initializeDropdowns() {
    console.log("Initializing analysis details buttons");
    
    // Get all detail toggle buttons
    const detailsButtons = document.querySelectorAll('.details-toggle');
    
    // Add click event listener to each button
    detailsButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Get target details row ID from data attribute
            const targetId = this.getAttribute('data-target');
            console.log(`Toggle clicked for ${targetId}`);
            
            // Find the target row
            const detailsRow = document.getElementById(targetId);
            
            if (detailsRow) {
                // Toggle visibility
                const isCurrentlyHidden = detailsRow.classList.contains('d-none');
                
                // Close all other details rows first
                document.querySelectorAll('.analysis-details').forEach(row => {
                    if (row.id !== targetId) {
                        row.classList.add('d-none');
                    }
                });
                
                // Toggle target row
                detailsRow.classList.toggle('d-none');
                
                // If we're showing the details and chart hasn't been loaded yet
                if (isCurrentlyHidden) {
                    // Extract symbol from the target ID
                    const symbol = targetId.replace('details-', '');
                    console.log(`Loading chart for ${symbol}`);
                    
                    // Load the chart
                    loadAnalysisChart(symbol);
                }
            }
        });
    });
    
    // Initialize verify price buttons
    const verifyButtons = document.querySelectorAll('.verify-price');
    verifyButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            
            const symbol = this.getAttribute('data-symbol');
            const dateInput = document.getElementById('analysis_date');
            const date = dateInput ? dateInput.value : getAnalysisDate();
            
            console.log(`Verifying price for ${symbol} on ${date}`);
            verifyStockPrice(symbol, date);
        });
    });
}

function calculatePortfolioProjections() {
    const investmentInput = document.getElementById('investment-amount');
    const total_amount = parseFloat(investmentInput.value);
    
    if (isNaN(total_amount) || total_amount <= 0) {
        console.warn('Invalid investment amount:', investmentInput.value);
            return;
        }
        
    // Get all symbols from the results table
    const symbolCells = document.querySelectorAll('.results-table tbody tr td:first-child');
    const symbols = Array.from(symbolCells).map(cell => cell.textContent.trim());
    
    if (symbols.length === 0) {
        console.warn('No symbols found in results table');
                    return;
                }
                
    // Collect recommendation info from the badge elements
    const recommendations = {};
    document.querySelectorAll('.results-table tbody tr').forEach(row => {
        const symbol = row.querySelector('td:first-child').textContent.trim();
        const recommendationBadge = row.querySelector('td:nth-child(2) .badge');
        
        if (recommendationBadge) {
            recommendations[symbol] = recommendationBadge.textContent.trim();
        }
    });
    
    // Calculate allocation based on recommendations
    const allocation = {};
    const strongBuySymbols = symbols.filter(symbol => recommendations[symbol] === 'STRONG BUY');
    const buySymbols = symbols.filter(symbol => recommendations[symbol] === 'BUY');
    
    const strongBuyAllocation = strongBuySymbols.length > 0 ? total_amount * 0.7 / strongBuySymbols.length : 0;
    const buyAllocation = buySymbols.length > 0 ? total_amount * 0.3 / buySymbols.length : 0;
    
    strongBuySymbols.forEach(symbol => {
        allocation[symbol] = strongBuyAllocation;
    });
    
    buySymbols.forEach(symbol => {
        allocation[symbol] = buyAllocation;
    });
    
    // Get 3-day and 5-day returns from the table
    const returns = {};
    document.querySelectorAll('.results-table tbody tr').forEach(row => {
        const symbol = row.querySelector('td:first-child').textContent.trim();
        const day3ReturnCell = row.querySelector('td:nth-child(5)');
        const day5ReturnCell = row.querySelector('td:nth-child(6)');
        
        if (day3ReturnCell && day5ReturnCell) {
            const day3Return = parseFloat(day3ReturnCell.textContent.trim().replace('%', ''));
            const day5Return = parseFloat(day5ReturnCell.textContent.trim().replace('%', ''));
            
            returns[symbol] = {
                day3Return: isNaN(day3Return) ? 0 : day3Return,
                day5Return: isNaN(day5Return) ? 0 : day5Return
            };
        }
    });
    
    // Calculate projected values
    let total3DayValue = 0;
    let total5DayValue = 0;
    
    Object.keys(allocation).forEach(symbol => {
        const investment = allocation[symbol];
        const symbolReturns = returns[symbol] || { day3Return: 0, day5Return: 0 };
        
        total3DayValue += investment * (1 + symbolReturns.day3Return / 100);
        total5DayValue += investment * (1 + symbolReturns.day5Return / 100);
    });
    
    // Calculate percentage returns
    const day3ReturnPct = total_amount > 0 ? (total3DayValue - total_amount) / total_amount * 100 : 0;
    const day5ReturnPct = total_amount > 0 ? (total5DayValue - total_amount) / total_amount * 100 : 0;
    
    // Update UI
    const totalInvestmentEl = document.querySelector('.dashboard-card:nth-child(1) h3');
    const day3ReturnEl = document.querySelector('.dashboard-card:nth-child(2) h3');
    const day5ReturnEl = document.querySelector('.dashboard-card:nth-child(3) h3');
    
    if (totalInvestmentEl) totalInvestmentEl.textContent = `$${total_amount.toFixed(2)}`;
    if (day3ReturnEl) day3ReturnEl.textContent = `${day3ReturnPct.toFixed(2)}%`;
    if (day5ReturnEl) day5ReturnEl.textContent = `${day5ReturnPct.toFixed(2)}%`;
    
    // Update profit/loss badges
    const day3ProfitLossEl = document.querySelector('.dashboard-card:nth-child(2) .badge');
    const day5ProfitLossEl = document.querySelector('.dashboard-card:nth-child(3) .badge');
    
    const day3ProfitLoss = total3DayValue - total_amount;
    const day5ProfitLoss = total5DayValue - total_amount;
    
    if (day3ProfitLossEl) {
        day3ProfitLossEl.className = `badge ${day3ProfitLoss >= 0 ? 'bg-success' : 'bg-danger'}`;
        day3ProfitLossEl.textContent = `${day3ProfitLoss >= 0 ? '+' : ''}$${day3ProfitLoss.toFixed(2)}`;
    }
    
    if (day5ProfitLossEl) {
        day5ProfitLossEl.className = `badge ${day5ProfitLoss >= 0 ? 'bg-success' : 'bg-danger'}`;
        day5ProfitLossEl.textContent = `${day5ProfitLoss >= 0 ? '+' : ''}$${day5ProfitLoss.toFixed(2)}`;
    }
    
    // Update allocation in the table
    document.querySelectorAll('.results-table tbody tr').forEach(row => {
        const symbol = row.querySelector('td:first-child').textContent.trim();
        const allocationCell = row.querySelector('td:nth-child(4)');
        
        if (allocationCell && allocation[symbol]) {
            allocationCell.textContent = `$${allocation[symbol].toFixed(2)}`;
        } else if (allocationCell) {
            allocationCell.textContent = '$0.00';
        }
    });
}

window.addEventListener('resize', function() {
    // Resize all active charts
    Object.keys(chartInstances).forEach(symbol => {
        if (chartInstances[symbol]) {
            chartInstances[symbol].resize();
        }
    });
});

function getAnalysisDate() {
    // Try to get date from form input first
    const dateInput = document.getElementById('date');
    if (dateInput && dateInput.value) {
        return dateInput.value;
    }
    
    // Fallback to trying to extract from the HTML
    try {
        const returnElements = document.querySelectorAll('td:first-child');
        for (const el of returnElements) {
            if (el.textContent.includes('3-Day Return')) {
                const match = el.textContent.match(/\((\d{4}-\d{2}-\d{2})\)/);
                if (match && match[1]) {
                    return match[1];
                }
            }
        }
    } catch (e) {
        console.error("Error extracting date from HTML:", e);
    }
    
    // Default to current date if no other date is found
    const today = new Date();
    return today.toISOString().split('T')[0];
}

function addDebugInfo() {
    // Create a debug info container
    const debugInfo = document.createElement('div');
    debugInfo.style.display = 'none'; // Hidden by default
    debugInfo.id = 'debug-info';
    debugInfo.classList.add('card', 'mt-3', 'p-3');
    
    const toggleBtn = document.createElement('button');
    toggleBtn.textContent = 'Show Debug Info';
    toggleBtn.classList.add('btn', 'btn-sm', 'btn-outline-secondary', 'mb-2');
    toggleBtn.addEventListener('click', function() {
        const infoPanel = document.getElementById('debug-panel');
        if (infoPanel.style.display === 'none') {
            infoPanel.style.display = 'block';
            this.textContent = 'Hide Debug Info';
        } else {
            infoPanel.style.display = 'none';
            this.textContent = 'Show Debug Info';
        }
    });
    
    const infoPanel = document.createElement('div');
    infoPanel.id = 'debug-panel';
    infoPanel.style.display = 'none';
    
    // Get page info
    const pageInfo = {
        'Page URL': window.location.href,
        'Analysis Date': document.getElementById('date')?.value || 'Not found',
        'Stock Symbols': document.getElementById('symbols')?.value || 'Not found',
        'Investment Amount': document.getElementById('investment')?.value || 'Not found',
        'Projection Elements': {
            '3-Day Element': document.getElementById('projected-3day-return') ? 'Found' : 'Not found',
            '5-Day Element': document.getElementById('projected-5day-return') ? 'Found' : 'Not found'
        },
        'Analysis Rows': document.querySelectorAll('.analysis-row').length,
        'Detail Rows': document.querySelectorAll('tr[id^="details-"]').length
    };
    
    // Create debug table
    const table = document.createElement('table');
    table.classList.add('table', 'table-sm', 'table-bordered');
    
    // Add header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    ['Property', 'Value'].forEach(text => {
        const th = document.createElement('th');
        th.textContent = text;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Add body
    const tbody = document.createElement('tbody');
    
    // Add simple properties
    Object.entries(pageInfo).forEach(([key, value]) => {
        if (typeof value !== 'object') {
            const row = document.createElement('tr');
            const keyCell = document.createElement('td');
            const valueCell = document.createElement('td');
            
            keyCell.textContent = key;
            valueCell.textContent = value;
            
            row.appendChild(keyCell);
            row.appendChild(valueCell);
            tbody.appendChild(row);
        }
    });
    
    // Add nested properties
    Object.entries(pageInfo).forEach(([key, value]) => {
        if (typeof value === 'object') {
            const headerRow = document.createElement('tr');
            const headerCell = document.createElement('td');
            headerCell.colSpan = 2;
            headerCell.textContent = key;
            headerCell.classList.add('table-secondary');
            headerRow.appendChild(headerCell);
            tbody.appendChild(headerRow);
            
            // Add subproperties
            Object.entries(value).forEach(([subKey, subValue]) => {
                const row = document.createElement('tr');
                const keyCell = document.createElement('td');
                const valueCell = document.createElement('td');
                
                keyCell.textContent = '  ' + subKey;
                valueCell.textContent = subValue;
                
                row.appendChild(keyCell);
                row.appendChild(valueCell);
                tbody.appendChild(row);
            });
        }
    });
    
    table.appendChild(tbody);
    infoPanel.appendChild(table);
    
    // Add button and panel to debug info container
    debugInfo.appendChild(toggleBtn);
    debugInfo.appendChild(infoPanel);
    
    // Add to document
    document.querySelector('.container-fluid').appendChild(debugInfo);
}

function debugChartIssue(symbol) {
    console.log('--- Chart Debug Information ---');
    console.log(`Debugging chart for symbol: ${symbol}`);
    
    // Check if containers exist
    const containerExists = !!document.getElementById(`chart-container-${symbol}`);
    console.log(`Chart container exists: ${containerExists}`);
    
    const canvasExists = !!document.getElementById(`chart-${symbol}`);
    console.log(`Chart canvas exists: ${canvasExists}`);
    
    // Check if chart instance exists
    const chartInstanceExists = chartInstances && !!chartInstances[symbol];
    console.log(`Chart instance exists: ${chartInstanceExists}`);
    
    // Fetch API test
    console.log('Testing API endpoint...');
    fetch(`/api/stock-data/${symbol}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.log(`API Error: ${data.error}`);
            } else {
                console.log(`API success: Received ${data.length} data points`);
            }
        })
        .catch(error => {
            console.log(`API fetch error: ${error.message}`);
        });
        
    console.log('--- End Debug Information ---');
    
    // Return the debug info as HTML for display
    return `
        <div class="card mb-3">
            <div class="card-header bg-light">Chart Debug Information</div>
            <div class="card-body">
                <ul>
                    <li>Symbol: ${symbol}</li>
                    <li>Container exists: ${containerExists}</li>
                    <li>Canvas exists: ${canvasExists}</li>
                    <li>Chart instance exists: ${chartInstanceExists}</li>
                </ul>
                <p>Check browser console for complete debug output</p>
                <button class="btn btn-sm btn-primary" onclick="loadAnalysisChart('${symbol}')">
                    Retry Loading Chart
                </button>
            </div>
        </div>
    `;
}

// Add this function to check and display the actual price
function verifyStockPrice(symbol, date) {
    fetch(`/debug/price/${symbol}/${date}`)
        .then(response => response.json())
        .then(data => {
            console.log('Price verification:', data);
            
            // Create or update verification notice
            let noticeDiv = document.getElementById(`price-verify-${symbol}`);
            if (!noticeDiv) {
                noticeDiv = document.createElement('div');
                noticeDiv.id = `price-verify-${symbol}`;
                noticeDiv.className = 'alert alert-info mt-2';
                
                // Find a place to append it
                const detailsRow = document.getElementById(`details-${symbol}`);
                if (detailsRow) {
                    detailsRow.querySelector('.card-body').appendChild(noticeDiv);
                } else {
                    // Fallback location
                    document.querySelector('.container-fluid').appendChild(noticeDiv);
                }
            }
            
            if (data.error) {
                noticeDiv.innerHTML = `<strong>Error:</strong> ${data.error}`;
                return;
            }
            
            // Create a price verification table
            noticeDiv.innerHTML = `
                <h6>Price Verification for ${symbol}</h6>
                <p>Requested date: ${data.requested_date}<br>
                Closest available date: ${data.closest_date}<br>
                Current displayed price: $${data.price.toFixed(2)}</p>
                
                <table class="table table-sm table-bordered">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Price</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.date_range.map(item => `
                            <tr ${item.is_target ? 'class="table-primary"' : ''}>
                                <td>${item.date}</td>
                                <td>$${item.price.toFixed(2)}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
                
                <button class="btn btn-sm btn-primary" onclick="reanalyzeStock('${symbol}', '${date}')">
                    Reanalyze with Verified Price
                </button>
            `;
        })
        .catch(error => {
            console.error('Price verification error:', error);
        });
}

// Add this function to allow reanalysis with the correct price
function reanalyzeStock(symbol, date) {
    // Create a temporary form to submit
    const form = document.createElement('form');
    form.method = 'POST';
    form.action = '/analyze';
    
    // Add the necessary inputs
    const symbolInput = document.createElement('input');
    symbolInput.type = 'hidden';
    symbolInput.name = 'symbols';
    symbolInput.value = symbol;
    
    const dateInput = document.createElement('input');
    dateInput.type = 'hidden';
    dateInput.name = 'date';
    dateInput.value = date;
    
    // Add investment amount from current page
    const investmentInput = document.createElement('input');
    investmentInput.type = 'hidden';
    investmentInput.name = 'investment';
    investmentInput.value = document.getElementById('investment')?.value || '1000';
    
    // Add current stop loss and take profit values
    const stopLossInput = document.createElement('input');
    stopLossInput.type = 'hidden';
    stopLossInput.name = 'stop_loss';
    stopLossInput.value = document.getElementById('stop_loss')?.value || '5';
    
    const takeProfitInput = document.createElement('input');
    takeProfitInput.type = 'hidden';
    takeProfitInput.name = 'take_profit';
    takeProfitInput.value = document.getElementById('take_profit')?.value || '8';
    
    // Add all inputs to form
    form.appendChild(symbolInput);
    form.appendChild(dateInput);
    form.appendChild(investmentInput);
    form.appendChild(stopLossInput);
    form.appendChild(takeProfitInput);
    
    // Add form to document and submit
    document.body.appendChild(form);
    form.submit();
} 