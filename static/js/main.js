// Global utility functions for use across all pages

// Format currency values
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
}

// Format percentage values
function formatPercent(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value / 100);
}

// Create a color-coded badge for changes
function createChangeBadge(value) {
    const badgeClass = value >= 0 ? 'bg-success' : 'bg-danger';
    const sign = value >= 0 ? '+' : '';
    return `<span class="badge ${badgeClass}">${sign}${value.toFixed(2)}%</span>`;
}

// Add a "shake" animation to elements
function shakeElement(element) {
    element.classList.add('shake');
    setTimeout(() => {
        element.classList.remove('shake');
    }, 500);
}

// Flash element with green/red based on value change
function flashPriceChange(element, newValue, oldValue) {
    if (newValue > oldValue) {
        element.classList.add('price-change-up');
        setTimeout(() => element.classList.remove('price-change-up'), 2000);
    } else if (newValue < oldValue) {
        element.classList.add('price-change-down');
        setTimeout(() => element.classList.remove('price-change-down'), 2000);
    }
}

// Initialize tooltips
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});

/**
 * Display a notification message
 * @param {string} message - The message to display
 * @param {string} type - The type of message (success, danger, warning, info)
 * @param {number} duration - How long to display the message in ms
 */
function showNotification(message, type = 'info', duration = 5000) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = message;
    
    // Add to document
    document.body.appendChild(notification);
    
    // Remove after specified duration
    setTimeout(() => {
        notification.style.animation = 'fade-out 0.3s forwards';
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, duration);
}

// Check for message in URL parameters when page loads
document.addEventListener('DOMContentLoaded', function() {
    const urlParams = new URLSearchParams(window.location.search);
    const message = urlParams.get('message');
    const messageType = urlParams.get('type') || 'info';
    
    if (message) {
        showNotification(decodeURIComponent(message), messageType);
        
        // Clean up URL (remove message parameters)
        if (window.history && window.history.replaceState) {
            const cleanUrl = window.location.pathname + 
                             (window.location.search.replace(/[?&]message=[^&]+/, '').replace(/[?&]type=[^&]+/, '') || '');
            window.history.replaceState({}, document.title, cleanUrl);
        }
    }
    
    // Add click handler for all execute-trade buttons
    const tradeBtns = document.querySelectorAll('.execute-trade-btn');
    tradeBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const symbol = this.dataset.symbol;
            showNotification(`Processing trade for ${symbol}...`, 'info');
        });
    });
});

// Handle JSON responses for AJAX calls
function handleApiResponse(response, successCallback) {
    response.json().then(data => {
        if (data.status === 'success') {
            showNotification(data.message || 'Operation completed successfully', 'success');
            if (successCallback) successCallback(data);
        } else {
            showNotification(data.message || 'An error occurred', 'danger');
        }
    }).catch(err => {
        showNotification('Failed to process response', 'danger');
    });
} 