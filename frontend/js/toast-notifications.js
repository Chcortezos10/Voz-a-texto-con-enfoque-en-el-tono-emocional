//sistema de notificaicones

class ToastNotifications {
constructor() {
        this.container = this.createContainer();
        document.body.appendChild(this.container);
        this.setupStyles();
    }

    createContainer() {
        const container = document.createElement('div');
        container.id = 'toastContainer';
        container.style.cssText = `
            position: fixed;
            top: 80px;
            right: 20px;
            z-index: 10000;
            display: flex;
            flex-direction: column;
            gap: 10px;
            pointer-events: none;
        `;
        return container;
    }

    setupStyles() {
        if (!document.getElementById('toast-styles')) {
            const style = document.createElement('style');
            style.id = 'toast-styles';
            style.textContent = `
                .toast {
                    pointer-events: auto;
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    min-width: 300px;
                    max-width: 400px;
                    padding: 16px 20px;
                    background: var(--surface);
                    border: 1px solid var(--border);
                    border-radius: 8px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                    backdrop-filter: blur(20px);
                    color: var(--text-main);
                    opacity: 0;
                    transform: translateX(400px);
                    transition: all 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55);
                }
                .toast.show {
                    opacity: 1 !important;
                    transform: translateX(0) !important;
                }
                .toast-close {
                    background: none;
                    border: none;
                    color: var(--text-muted);
                    cursor: pointer;
                    font-size: 18px;
                    padding: 0;
                    margin-left: auto;
                }
            `;
            document.head.appendChild(style);
        }
    }

    show(message, type = 'info', duration = 3000) {
        const toast = this.createToast(message, type);
        this.container.appendChild(toast);

        setTimeout(() => toast.classList.add('show'), 10);

        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, duration);

        return toast;
    }

    createToast(message, type) {
        const icons = {
            success: '<i class="fas fa-check-circle"></i>',
            error: '<i class="fas fa-times-circle"></i>',
            warning: '<i class="fas fa-exclamation-triangle"></i>',
            info: '<i class="fas fa-info-circle"></i>'
        };

        const colors = {
            success: '#10b981',
            error: '#ef4444',
            warning: '#f59e0b',
            info: '#6366f1'
        };

        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.style.borderLeft = `4px solid ${colors[type]}`;

        toast.innerHTML = `
            <div style="color: ${colors[type]}; font-size: 20px;">${icons[type]}</div>
            <div style="flex: 1; font-weight: 500;">${message}</div>
            <button class="toast-close" onclick="this.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        `;

        return toast;
    }

    success(message, duration) { return this.show(message, 'success', duration); }
    error(message, duration) { return this.show(message, 'error', duration); }
    warning(message, duration) { return this.show(message, 'warning', duration); }
    info(message, duration) { return this.show(message, 'info', duration); }
}

const originalAlert = window.alert;
window.alert = function(message) {
    if (!window.toast) return originalAlert(message);
    
    const msg = message.replace(/✅|❌|⚠️/g, '').trim();
    
    if (message.includes('✅') || message.toLowerCase().includes('éxito')) {
        window.toast.success(msg);
    } else if (message.includes('❌') || message.toLowerCase().includes('error')) {
        window.toast.error(msg);
    } else if (message.includes('⚠️')) {
        window.toast.warning(msg);
    } else {
        window.toast.info(msg);
    }
};
