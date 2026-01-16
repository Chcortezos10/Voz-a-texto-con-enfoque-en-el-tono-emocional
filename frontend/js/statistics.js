//reporte y estaditicas

class StatisticsManager {
    constructor() {
        this.historyData = [];
        this.charts = {};
    }

    async loadHistoryForStats() {
        try {
            const data = await API.get('/history/');
            this.historyData = Array.isArray(data) ? data : [];
            return this.historyData;
        } catch (e) {
            console.error('Error cargando historial:', e);
            return [];
        }
    }

    async generateDashboard() {
        await this.loadHistoryForStats();
        
        if (this.historyData.length === 0) {
            if (window.toast) window.toast.warning('No hay suficientes datos para estadísticas');
            return;
        }

        this.showStatsPanel();
        this.renderGlobalStats();
        this.renderEmotionTrends();
        this.renderSpeakerStats();
        this.renderTimelineAnalysis();
    }

    showStatsPanel() {
        let panel = document.getElementById('statsPanel');
        if (!panel) {
            const html = `
                <div id="statsPanel" class="stats-panel">
                    <div class="stats-header">
                        <h2><i class="fas fa-chart-bar"></i> Dashboard de Estadísticas</h2>
                        <div class="stats-actions">
                            <button class="btn btn-primary" onclick="statsManager.exportReport()">
                                <i class="fas fa-file-download"></i> Exportar Reporte
                            </button>
                            <button class="btn btn-secondary" onclick="statsManager.closeStats()">
                                <i class="fas fa-times"></i> Cerrar
                            </button>
                        </div>
                    </div>

                    <div class="global-metrics">
                        <div class="metric-card-large">
                            <i class="fas fa-clock metric-icon"></i>
                            <div class="metric-value" id="totalHours">0</div>
                            <div class="metric-label">Horas Analizadas</div>
                        </div>
                        <div class="metric-card-large">
                            <i class="fas fa-file-audio metric-icon"></i>
                            <div class="metric-value" id="totalAnalyses">0</div>
                            <div class="metric-label">Total Análisis</div>
                        </div>
                        <div class="metric-card-large">
                            <i class="fas fa-users metric-icon"></i>
                            <div class="metric-value" id="totalSpeakers">0</div>
                            <div class="metric-label">Total Segmentos</div>
                        </div>
                        <div class="metric-card-large">
                            <i class="fas fa-smile metric-icon"></i>
                            <div class="metric-value" id="dominantEmotion">-</div>
                            <div class="metric-label">Emoción Dominante</div>
                        </div>
                    </div>

                    <div class="stats-charts-grid">
                        <div class="chart-container-stats">
                            <h3>Distribución Global de Emociones</h3>
                            <canvas id="globalEmotionChart"></canvas>
                        </div>
                        <div class="chart-container-stats">
                            <h3>Análisis por Mes</h3>
                            <canvas id="monthlyTrendChart"></canvas>
                        </div>
                        <div class="chart-container-stats">
                            <h3>Duración Promedio por Emoción</h3>
                            <canvas id="durationByEmotionChart"></canvas>
                        </div>
                        <div class="chart-container-stats">
                            <h3>Evolución Temporal</h3>
                            <canvas id="timelineEvolutionChart"></canvas>
                        </div>
                    </div>

                    <div class="recent-analyses">
                        <h3><i class="fas fa-history"></i> Análisis Recientes</h3>
                        <div class="analysis-table" id="analysisTable"></div>
                    </div>
                </div>
            `;
            
            document.querySelector('.container').insertAdjacentHTML('beforeend', html);
        } else {
            panel.style.display = 'block';
        }
    }

    closeStats() {
        const panel = document.getElementById('statsPanel');
        if (panel) panel.style.display = 'none';
    }

    renderGlobalStats() {
        const totalDuration = this.historyData.reduce((sum, item) => sum + (item.duration || 0), 0);
        const totalHours = (totalDuration / 3600).toFixed(1);
        const totalAnalyses = this.historyData.length;
        const totalSegments = this.historyData.reduce((sum, item) => sum + (item.num_segments || 0), 0);

        const emotions = {};
        this.historyData.forEach(item => {
            const emotion = item.dominant_emotion || 'neutral';
            emotions[emotion] = (emotions[emotion] || 0) + 1;
        });
        
        const dominantEmotion = Object.entries(emotions).sort((a, b) => b[1] - a[1])[0];

        const totalHoursEl = document.getElementById('totalHours');
        const totalAnalysesEl = document.getElementById('totalAnalyses');
        const totalSpeakersEl = document.getElementById('totalSpeakers');
        const dominantEmotionEl = document.getElementById('dominantEmotion');

        if (totalHoursEl) totalHoursEl.textContent = totalHours + 'h';
        if (totalAnalysesEl) totalAnalysesEl.textContent = totalAnalyses;
        if (totalSpeakersEl) totalSpeakersEl.textContent = totalSegments;
        if (dominantEmotionEl && dominantEmotion) {
            const emoji = CONFIG.EMOTIONS[dominantEmotion[0]] || '😐';
            dominantEmotionEl.textContent = `${emoji} ${dominantEmotion[0]}`;
        }
    }

    renderEmotionTrends() {
        const emotions = { 'feliz': 0, 'enojado': 0, 'triste': 0, 'neutral': 0 };
        
        this.historyData.forEach(item => {
            const emotion = item.dominant_emotion || 'neutral';
            emotions[emotion]++;
        });

        const ctx = document.getElementById('globalEmotionChart');
        if (!ctx) return;

        if (this.charts.globalEmotion) this.charts.globalEmotion.destroy();

        this.charts.globalEmotion = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(emotions).map(e => e.charAt(0).toUpperCase() + e.slice(1)),
                datasets: [{
                    data: Object.values(emotions),
                    backgroundColor: Object.keys(emotions).map(e => CONFIG.EMOTION_COLORS[e]),
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { 
                            color: getComputedStyle(document.documentElement).getPropertyValue('--text-main'),
                            padding: 15 
                        }
                    }
                }
            }
        });
    }

    renderSpeakerStats() {
        const emotionDurations = { 'feliz': [], 'enojado': [], 'triste': [], 'neutral': [] };
        
        this.historyData.forEach(item => {
            const emotion = item.dominant_emotion || 'neutral';
            emotionDurations[emotion].push(item.duration || 0);
        });

        const avgDurations = {};
        Object.keys(emotionDurations).forEach(emotion => {
            const durations = emotionDurations[emotion];
            avgDurations[emotion] = durations.length > 0 
                ? durations.reduce((a, b) => a + b, 0) / durations.length 
                : 0;
        });

        const ctx = document.getElementById('durationByEmotionChart');
        if (!ctx) return;

        if (this.charts.durationEmotion) this.charts.durationEmotion.destroy();

        this.charts.durationEmotion = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: Object.keys(avgDurations).map(e => e.charAt(0).toUpperCase() + e.slice(1)),
                datasets: [{
                    label: 'Duración Promedio (segundos)',
                    data: Object.values(avgDurations),
                    backgroundColor: Object.keys(avgDurations).map(e => CONFIG.EMOTION_COLORS[e]),
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-muted') },
                        grid: { color: 'rgba(42, 42, 58, 0.5)' }
                    },
                    x: {
                        ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-muted') },
                        grid: { display: false }
                    }
                }
            }
        });
    }

    renderTimelineAnalysis() {
        const monthlyData = {};
        
        this.historyData.forEach(item => {
            const date = new Date(item.timestamp);
            const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
            
            if (!monthlyData[monthKey]) {
                monthlyData[monthKey] = { count: 0, emotions: {} };
            }
            
            monthlyData[monthKey].count++;
            const emotion = item.dominant_emotion || 'neutral';
            monthlyData[monthKey].emotions[emotion] = (monthlyData[monthKey].emotions[emotion] || 0) + 1;
        });

        const sortedMonths = Object.keys(monthlyData).sort();
        const monthLabels = sortedMonths.map(m => {
            const [year, month] = m.split('-');
            const date = new Date(year, parseInt(month) - 1);
            return date.toLocaleDateString('es-ES', { month: 'short', year: 'numeric' });
        });

        // Gráfico mensual
        const ctx1 = document.getElementById('monthlyTrendChart');
        if (ctx1) {
            if (this.charts.monthly) this.charts.monthly.destroy();

            this.charts.monthly = new Chart(ctx1, {
                type: 'line',
                data: {
                    labels: monthLabels,
                    datasets: [{
                        label: 'Análisis por Mes',
                        data: sortedMonths.map(m => monthlyData[m].count),
                        borderColor: '#6366f1',
                        backgroundColor: 'rgba(99, 102, 241, 0.1)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: { 
                                color: getComputedStyle(document.documentElement).getPropertyValue('--text-muted'),
                                precision: 0 
                            },
                            grid: { color: 'rgba(42, 42, 58, 0.5)' }
                        },
                        x: {
                            ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-muted') },
                            grid: { color: 'rgba(42, 42, 58, 0.5)' }
                        }
                    }
                }
            });
        }

        // Evolución de emociones
        const emotions = ['feliz', 'enojado', 'triste', 'neutral'];
        const ctx2 = document.getElementById('timelineEvolutionChart');
        if (ctx2) {
            if (this.charts.evolution) this.charts.evolution.destroy();

            this.charts.evolution = new Chart(ctx2, {
                type: 'line',
                data: {
                    labels: monthLabels,
                    datasets: emotions.map(emotion => ({
                        label: emotion.charAt(0).toUpperCase() + emotion.slice(1),
                        data: sortedMonths.map(m => monthlyData[m].emotions[emotion] || 0),
                        borderColor: CONFIG.EMOTION_COLORS[emotion],
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        tension: 0.4
                    }))
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-main') },
                            position: 'bottom'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: { 
                                color: getComputedStyle(document.documentElement).getPropertyValue('--text-muted'),
                                precision: 0 
                            },
                            grid: { color: 'rgba(42, 42, 58, 0.5)' }
                        },
                        x: {
                            ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-muted') },
                            grid: { color: 'rgba(42, 42, 58, 0.5)' }
                        }
                    }
                }
            });
        }

        // Tabla de análisis recientes
        const table = document.getElementById('analysisTable');
        if (table) {
            const recent = [...this.historyData]
                .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
                .slice(0, 10);

            table.innerHTML = `
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Archivo</th>
                            <th>Fecha</th>
                            <th>Duración</th>
                            <th>Emoción</th>
                            <th>Segmentos</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${recent.map(item => `
                            <tr>
                                <td><i class="fas fa-file-audio"></i> ${item.filename || 'Sin nombre'}</td>
                                <td>${Utils.formatDate(item.timestamp)}</td>
                                <td>${Utils.formatDuration(item.duration || 0)}</td>
                                <td><span class="emotion-badge ${item.dominant_emotion}">${item.dominant_emotion || 'neutral'}</span></td>
                                <td>${item.num_segments || 0}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
        }
    }

    async exportReport() {
        if (window.toast) window.toast.info('Generando reporte...');

        const reportData = {
            generatedAt: new Date().toISOString(),
            totalAnalyses: this.historyData.length,
            totalDuration: this.historyData.reduce((sum, item) => sum + (item.duration || 0), 0),
            analyses: this.historyData,
            statistics: {
                emotions: this.calculateEmotionStats(),
                monthlyTrend: this.calculateMonthlyTrend()
            }
        };

        Utils.downloadFile(
            JSON.stringify(reportData, null, 2),
            `reporte_estadisticas_${Date.now()}.json`,
            'application/json'
        );

        if (window.toast) window.toast.success('Reporte exportado');
    }

    calculateEmotionStats() {
        const emotions = { 'feliz': 0, 'enojado': 0, 'triste': 0, 'neutral': 0 };
        this.historyData.forEach(item => {
            const emotion = item.dominant_emotion || 'neutral';
            emotions[emotion]++;
        });
        return emotions;
    }

    calculateMonthlyTrend() {
        const monthlyData = {};
        this.historyData.forEach(item => {
            const date = new Date(item.timestamp);
            const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
            monthlyData[monthKey] = (monthlyData[monthKey] || 0) + 1;
        });
        return monthlyData;
    }
}