//graficos avanzados

class AdvancedCharts {
    constructor() {
        this.charts = {};
    }

    renderIntensityTimeline(containerId, data) {
        const ctx = document.getElementById(containerId);
        if (!ctx) return;

        const segments = data.segments || [];
        
        if (this.charts[containerId]) {
            this.charts[containerId].destroy();
        }

        this.charts[containerId] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: segments.map(s => s.start.toFixed(1) + 's'),
                datasets: [{
                    label: 'Intensidad Emocional',
                    data: segments.map(s => (s.intensity || 0) * 100),
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 2,
                    pointHoverRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Intensidad Emocional a lo Largo del Tiempo',
                        color: getComputedStyle(document.documentElement).getPropertyValue('--text-main'),
                        font: { size: 16, weight: 'bold' }
                    },
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            color: getComputedStyle(document.documentElement).getPropertyValue('--text-muted'),
                            callback: (value) => value + '%'
                        },
                        grid: { color: 'rgba(42, 42, 58, 0.5)' }
                    },
                    x: {
                        ticks: { 
                            color: getComputedStyle(document.documentElement).getPropertyValue('--text-muted'),
                            maxTicksLimit: 10
                        },
                        grid: { color: 'rgba(42, 42, 58, 0.5)' }
                    }
                }
            }
        });
    }

    renderSpeakerComparison(containerId, data) {
        const ctx = document.getElementById(containerId);
        if (!ctx) return;

        const segments = data.segments || [];
        const speakers = {};

        segments.forEach(seg => {
            const speaker = seg.speaker_label || 'Hablante 1';
            if (!speakers[speaker]) {
                speakers[speaker] = { feliz: 0, enojado: 0, triste: 0, neutral: 0, count: 0 };
            }
            const emotion = seg.emotion || 'neutral';
            speakers[speaker][emotion]++;
            speakers[speaker].count++;
        });

        Object.keys(speakers).forEach(speaker => {
            const total = speakers[speaker].count;
            ['feliz', 'enojado', 'triste', 'neutral'].forEach(emotion => {
                speakers[speaker][emotion] = (speakers[speaker][emotion] / total) * 100;
            });
        });

        if (this.charts[containerId]) {
            this.charts[containerId].destroy();
        }

        const speakerNames = Object.keys(speakers);
        const emotions = ['feliz', 'enojado', 'triste', 'neutral'];

        this.charts[containerId] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: speakerNames,
                datasets: emotions.map(emotion => ({
                    label: emotion.charAt(0).toUpperCase() + emotion.slice(1),
                    data: speakerNames.map(speaker => speakers[speaker][emotion]),
                    backgroundColor: CONFIG.EMOTION_COLORS[emotion],
                    borderWidth: 0
                }))
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Distribución Emocional por Hablante',
                        color: getComputedStyle(document.documentElement).getPropertyValue('--text-main'),
                        font: { size: 16, weight: 'bold' }
                    },
                    legend: {
                        labels: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-main') }
                    }
                },
                scales: {
                    x: {
                        stacked: true,
                        ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-muted') },
                        grid: { display: false }
                    },
                    y: {
                        stacked: true,
                        ticks: {
                            color: getComputedStyle(document.documentElement).getPropertyValue('--text-muted'),
                            callback: (value) => value + '%'
                        },
                        grid: { color: 'rgba(42, 42, 58, 0.5)' }
                    }
                }
            }
        });
    }

    exportChart(chartId, filename) {
        const canvas = document.getElementById(chartId);
        if (!canvas) {
            if (window.toast) window.toast.error('Gráfico no encontrado');
            return;
        }

        canvas.toBlob((blob) => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename + '.png';
            a.click();
            URL.revokeObjectURL(url);
            if (window.toast) window.toast.success('Gráfico exportado');
        });
    }

    destroyAll() {
        Object.values(this.charts).forEach(chart => chart.destroy());
        this.charts = {};
    }
}