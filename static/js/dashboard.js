document.addEventListener('DOMContentLoaded', function() {
    // Initialize position detail toggles
    document.querySelectorAll('.details-toggle').forEach(btn => {
        btn.addEventListener('click', function() {
            const targetId = this.dataset.target;
            const detailsRow = document.getElementById(targetId);
            
            if (detailsRow) {
                detailsRow.classList.toggle('d-none');
                
                // Toggle icon
                const icon = this.querySelector('i');
                if (icon) {
                    icon.classList.toggle('bi-chevron-down');
                    icon.classList.toggle('bi-chevron-up');
                }
                
                // If showing details and chart isn't loaded yet, load it
                if (!detailsRow.classList.contains('d-none')) {
                    const symbol = targetId.replace('details-', '');
                    if (!this.dataset.chartLoaded) {
                        loadStockChart(symbol);
                        this.dataset.chartLoaded = 'true';
                    }
                }
            }
        });
    });
    
    // Initialize portfolio allocation chart
    initAllocationChart();
    
    // Initialize market overview chart
    initMarketChart();
});

function loadStockChart(symbol) {
    // Fetch stock data from API
    fetch(`/api/stock-data/${symbol}?days=90`)
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                createStockChart(symbol, data.data);
            } else {
                console.error('Error loading stock data:', data.message);
            }
        })
        .catch(error => {
            console.error('Error fetching stock data:', error);
        });
}

function createStockChart(symbol, data) {
    const chartEl = document.getElementById(`chart-${symbol}`);
    if (!chartEl) return;
    
    const ctx = chartEl.getContext('2d');
    
    // Extract data for chart
    const labels = data.map(item => item.date);
    const prices = data.map(item => item.close);
    const sma20 = data.map(item => item.sma20);
    
    // Determine if stock is up or down over the period
    const startPrice = prices[0];
    const endPrice = prices[prices.length - 1];
    const priceChange = endPrice - startPrice;
    const lineColor = priceChange >= 0 ? 'rgba(40, 167, 69, 1)' : 'rgba(220, 53, 69, 1)';
    const fillColor = priceChange >= 0 ? 'rgba(40, 167, 69, 0.1)' : 'rgba(220, 53, 69, 0.1)';
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: `${symbol} Price`,
                    data: prices,
                    borderColor: lineColor,
                    backgroundColor: fillColor,
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1
                },
                {
                    label: 'SMA 20',
                    data: sma20,
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
                    position: 'top',
                    align: 'end'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
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
}

function initAllocationChart() {
    const chartEl = document.getElementById('allocation-chart');
    if (!chartEl) return;
    
    const ctx = chartEl.getContext('2d');
    
    // Collect data from positions table
    const symbols = [];
    const values = [];
    const colors = [
        'rgba(54, 162, 235, 0.8)',
        'rgba(255, 99, 132, 0.8)',
        'rgba(255, 206, 86, 0.8)',
        'rgba(75, 192, 192, 0.8)',
        'rgba(153, 102, 255, 0.8)',
        'rgba(255, 159, 64, 0.8)',
        'rgba(199, 199, 199, 0.8)'
    ];
    
    document.querySelectorAll('.position-row').forEach((row, index) => {
        const symbol = row.dataset.symbol;
        const shares = parseFloat(row.querySelector('td:nth-child(2)').textContent);
        const price = parseFloat(row.querySelector('td:nth-child(4)').textContent.replace('$', ''));
        
        symbols.push(symbol);
        values.push(shares * price);
    });
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: symbols,
            datasets: [{
                data: values,
                backgroundColor: colors.slice(0, symbols.length),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        boxWidth: 15,
                        padding: 15
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.parsed;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = (value / total * 100).toFixed(1);
                            return `${context.label}: ${formatCurrency(value)} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

function initMarketChart() {
    const chartEl = document.getElementById('market-chart');
    if (!chartEl) return;
    
    const ctx = chartEl.getContext('2d');
    
    // Fetch S&P 500 data
    fetch('/api/stock-data/SPY?days=30')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                const marketData = data.data;
                
                // Extract data for chart
                const labels = marketData.map(item => item.date);
                const spyPrices = marketData.map(item => item.close);
                
                new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [
                            {
                                label: 'S&P 500',
                                data: spyPrices,
                                borderColor: 'rgba(54, 162, 235, 1)',
                                backgroundColor: 'rgba(54, 162, 235, 0.1)',
                                borderWidth: 2,
                                fill: true,
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
                                position: 'top',
                                align: 'end'
                            }
                        },
                        scales: {
                            x: {
                                grid: {
                                    display: false
                                },
                                ticks: {
                                    maxRotation: 0,
                                    maxTicksLimit: 5
                                }
                            },
                            y: {
                                grid: {
                                    color: 'rgba(0, 0, 0, 0.05)'
                                },
                                ticks: {
                                    callback: function(value) {
                                        return '$' + value.toFixed(0);
                                    }
                                }
                            }
                        }
                    }
                });
            }
        })
        .catch(error => {
            console.error('Error fetching market data:', error);
        });
} 