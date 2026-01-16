//gestion de temas

class ThemeManager {
    constructor() {
        this.currentTheme = localStorage.getItem('theme') || 'dark';
        this.init();
    }

    init() {
        this.applyTheme(this.currentTheme);
        this.createToggleButton();
    }

    applyTheme(theme) {
        const root = document.documentElement;
        
        if (theme === 'light') {
            root.style.setProperty('--bg', '#f8fafc');
            root.style.setProperty('--surface', '#ffffff');
            root.style.setProperty('--surface-hover', '#f1f5f9');
            root.style.setProperty('--border', '#e2e8f0');
            root.style.setProperty('--text-main', '#0f172a');
            root.style.setProperty('--text-muted', '#64748b');
        } else {
            root.style.setProperty('--bg', '#0a0a0f');
            root.style.setProperty('--surface', '#16161d');
            root.style.setProperty('--surface-hover', '#1f1f2a');
            root.style.setProperty('--border', '#2a2a3a');
            root.style.setProperty('--text-main', '#f8fafc');
            root.style.setProperty('--text-muted', '#94a3b8');
        }
        
        this.currentTheme = theme;
        localStorage.setItem('theme', theme);
    }

    toggle() {
        const newTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
        this.applyTheme(newTheme);
        this.updateToggleIcon();
    }

    createToggleButton() {
        const button = document.createElement('button');
        button.id = 'themeToggle';
        button.className = 'theme-toggle-btn';
        button.innerHTML = this.currentTheme === 'dark' 
            ? '<i class="fas fa-sun"></i>' 
            : '<i class="fas fa-moon"></i>';
        button.onclick = () => this.toggle();
        button.title = 'Cambiar tema';
        
        const header = document.querySelector('.header');
        if (header) header.appendChild(button);
    }

    updateToggleIcon() {
        const btn = document.getElementById('themeToggle');
        if (btn) {
            btn.innerHTML = this.currentTheme === 'dark' 
                ? '<i class="fas fa-sun"></i>' 
                : '<i class="fas fa-moon"></i>';
        }
    }
}