//bara de progreso

class ProgressBar {
    constructor() {
        this.overlay = document.getElementById('loadingOverlay');
        if (this.overlay) {
            this.enhanceOverlay();
        }
    }

    enhanceOverlay() {
        this.overlay.innerHTML = `
            <div class="progress-container">
                <div class="spinner"></div>
                <div style="font-size: 1.2rem; font-weight: 600; margin-top: 1rem;" id="progressText">
                    Procesando...
                </div>
                <div class="progress-bar-wrapper">
                    <div class="progress-bar-track">
                        <div class="progress-bar-fill" id="progressBarFill"></div>
                    </div>
                    <div class="progress-percentage" id="progressPercentage">0%</div>
                </div>
                <div class="progress-details" id="progressDetails">
                    Iniciando análisis...
                </div>
            </div>
        `;
    }

    show(text = 'Procesando...') {
        if (this.overlay) {
            this.overlay.classList.add('active');
            const textEl = document.getElementById('progressText');
            if (textEl) textEl.textContent = text;
            this.setProgress(0);
        }
    }

    hide() {
        if (this.overlay) {
            this.overlay.classList.remove('active');
        }
    }

    setProgress(percent, details = '') {
        const fill = document.getElementById('progressBarFill');
        const percentageText = document.getElementById('progressPercentage');
        const detailsText = document.getElementById('progressDetails');

        if (fill) fill.style.width = percent + '%';
        if (percentageText) percentageText.textContent = Math.round(percent) + '%';
        if (details && detailsText) detailsText.textContent = details;
    }

    update(percent, text, details) {
        const textEl = document.getElementById('progressText');
        if (text && textEl) textEl.textContent = text;
        this.setProgress(percent, details);
    }

    simulate(duration = 10000) {
        const steps = [
            { p: 10, text: 'Cargando archivo...', details: 'Preparando audio' },
            { p: 30, text: 'Transcribiendo audio...', details: 'Procesando con Whisper' },
            { p: 50, text: 'Detectando hablantes...', details: 'Aplicando diarización' },
            { p: 70, text: 'Analizando emociones...', details: 'Analizando sentimientos' },
            { p: 90, text: 'Generando resultados...', details: 'Finalizando' },
            { p: 100, text: 'Completado', details: 'Listo' }
        ];

        let currentStep = 0;
        const stepDuration = duration / steps.length;

        const interval = setInterval(() => {
            if (currentStep >= steps.length) {
                clearInterval(interval);
                return;
            }

            const step = steps[currentStep];
            this.update(step.p, step.text, step.details);
            currentStep++;
        }, stepDuration);

        return interval;
    }
}