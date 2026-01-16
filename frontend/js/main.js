// ============================================
// MAIN.JS - Funciones principales del dashboard
// ============================================

// Check API - DEFINIR PRIMERO para garantizar que siempre esté disponible
async function checkAPI() {
    try {
        const response = await fetch(`${CONFIG.API_URL}/health`);
        const statusEl = document.getElementById('apiStatus');
        if (statusEl) {
            if (response.ok) {
                statusEl.textContent = 'Online';
                statusEl.classList.add('online');
            } else {
                statusEl.textContent = 'Error';
                statusEl.classList.remove('online');
            }
        }
    } catch (e) {
        const statusEl = document.getElementById('apiStatus');
        if (statusEl) {
            statusEl.textContent = 'Offline';
            statusEl.classList.remove('online');
        }
        console.warn('API no disponible:', e.message);
    }
}

// Ejecutar check INMEDIATAMENTE y cada 5 segundos
checkAPI();
setInterval(checkAPI, 5000);

// Inicializar managers de forma segura (no bloquea si falla alguno)
function initManagers() {
    try { window.themeManager = new ThemeManager(); } catch(e) { console.warn('ThemeManager no disponible'); }
    try { window.toast = new ToastManager(); } catch(e) { console.warn('ToastManager no disponible'); }
    try { window.progressBar = new ProgressBar(); } catch(e) { console.warn('ProgressBar no disponible'); }
    try { window.advancedCharts = new AdvancedCharts(); } catch(e) { console.warn('AdvancedCharts no disponible'); }
    try { window.audioPlayer = new AudioPlayer(); } catch(e) { console.warn('AudioPlayer no disponible'); }
    try { window.searchFilters = new SearchFilters(); } catch(e) { console.warn('SearchFilters no disponible'); }
    try { window.transcriptEditor = new TranscriptEditor(); } catch(e) { console.warn('TranscriptEditor no disponible'); }
    try { window.statsManager = new StatisticsManager(); } catch(e) { console.warn('StatisticsManager no disponible'); }
}

// Inicializar cuando el DOM esté listo
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initManagers);
} else {
    initManagers();
}

// Tabs
function switchTab(tab) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    
    const clickedTab = event.target;
    clickedTab.classList.add('active');
    
    const tabContent = document.getElementById(`tab-${tab}`);
    if (tabContent) tabContent.classList.add('active');
    
    if (tab === 'history') loadHistory();
}

// Upload handling
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');

if (uploadArea && fileInput) {
    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('active');
    });
    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('active'));
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('active');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
}

function handleFile(file) {
    if (!file) return;
    selectedFile = file;
    const fileNameEl = document.getElementById('fileName');
    if (fileNameEl) {
        fileNameEl.textContent = `✅ ${file.name}`;
    }
}

// Model selector
document.querySelectorAll('.model-option').forEach(opt => {
    opt.addEventListener('click', function() {
        document.querySelectorAll('.model-option').forEach(o => o.classList.remove('active'));
        this.classList.add('active');
        const apiKeySection = document.getElementById('apiKeySection');
        if (apiKeySection) {
            apiKeySection.style.display = this.dataset.model === 'cloud' ? 'block' : 'none';
        }
    });
});

// Sliders
const numSpeakersSlider = document.getElementById('numSpeakers');
const audioWeightSlider = document.getElementById('audioWeight');

if (numSpeakersSlider) {
    numSpeakersSlider.addEventListener('input', function() {
        const valueEl = document.getElementById('numSpeakersValue');
        if (valueEl) {
            valueEl.textContent = this.value == 0 ? 'Auto' : this.value;
        }
    });
}

if (audioWeightSlider) {
    audioWeightSlider.addEventListener('input', function() {
        const valueEl = document.getElementById('audioWeightValue');
        if (valueEl) {
            valueEl.textContent = this.value + '%';
        }
    });
}

// API Key management
function saveApiKey() {
    const key = document.getElementById('openaiApiKey')?.value || 
                document.getElementById('settingsApiKey')?.value;
    if (key) {
        localStorage.setItem('openai_api_key', key);
        toast.success('API Key guardada');
    }
}

// Cargar API key guardada
window.addEventListener('load', () => {
    const key = localStorage.getItem('openai_api_key');
    if (key) {
        const openaiInput = document.getElementById('openaiApiKey');
        const settingsInput = document.getElementById('settingsApiKey');
        if (openaiInput) openaiInput.value = key;
        if (settingsInput) settingsInput.value = key;
    }
});

// Analyze button
const analyzeBtn = document.getElementById('analyzeBtn');
if (analyzeBtn) {
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) {
            toast.warning('Selecciona un archivo primero');
            return;
        }

        progressBar.show('Iniciando análisis...');
        const simulationInterval = progressBar.simulate(30000);

        try {
            const model = document.querySelector('.model-option.active')?.dataset.model || 'local';
            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('lite_mode', document.getElementById('liteMode')?.checked || false);
            formData.append('audio_weight', parseInt(document.getElementById('audioWeight')?.value || 40) / 100);
            formData.append('enable_diarization', document.getElementById('enableDiarization')?.checked || false);
            
            const numSpeakers = parseInt(document.getElementById('numSpeakers')?.value || 0);
            if (numSpeakers > 0) formData.append('num_speakers', numSpeakers);

            let endpoint = CONFIG.ENDPOINTS.TRANSCRIBE_LOCAL;
            if (model === 'cloud') {
                const apiKey = localStorage.getItem('openai_api_key');
                if (!apiKey) {
                    toast.warning('Configura tu API Key de OpenAI primero');
                    clearInterval(simulationInterval);
                    progressBar.hide();
                    return;
                }
                formData.append('api_key', apiKey);
                endpoint = CONFIG.ENDPOINTS.TRANSCRIBE_CLOUD;
            }

            const res = await fetch(CONFIG.API_URL + endpoint, { 
                method: 'POST', 
                body: formData 
            });
            
            if (!res.ok) throw new Error('Error en API');
            const data = await res.json();

            clearInterval(simulationInterval);
            progressBar.setProgress(100, 'Completado');
            
            setTimeout(() => {
                currentResults = data;
                saveToHistory(data, selectedFile.name);
                displayResults(data);
                
                const resultsSection = document.getElementById('resultsSection');
                if (resultsSection) resultsSection.style.display = 'block';
                
                progressBar.hide();
                toast.success('Análisis completado exitosamente');

                // Cargar audio si es archivo de audio
                if (selectedFile.type.startsWith('audio/')) {
                    audioPlayer.loadAudio(selectedFile, data.segments || []);
                }
            }, 500);

        } catch (e) {
            clearInterval(simulationInterval);
            progressBar.hide();
            toast.error('Error en el análisis: ' + e.message);
        }
    });
}

// Display results
function displayResults(data) {
    const emotions = CONFIG.EMOTIONS;
    const top = data.global_emotions?.top_emotion || 'neutral';
    
    const metricEmotion = document.getElementById('metricEmotion');
    const metricIntensity = document.getElementById('metricIntensity');
    const metricSpeakers = document.getElementById('metricSpeakers');
    const metricDuration = document.getElementById('metricDuration');

    if (metricEmotion) metricEmotion.textContent = emotions[top] + ' ' + top.toUpperCase();
    if (metricIntensity) metricIntensity.textContent = Math.round((data.global_emotions?.top_score || 0) * 100) + '%';
    if (metricSpeakers) metricSpeakers.textContent = data.diarization?.num_speakers || 1;
    if (metricDuration) metricDuration.textContent = Math.round(data.metadata?.total_duration || 0) + 's';

    renderCharts(data);
    renderTranscript(data.segments || []);

    // Cargar gráficos avanzados
    advancedCharts.renderIntensityTimeline('intensityChart', data);
    advancedCharts.renderSpeakerComparison('speakerChart', data);

    // Cargar filtros
    searchFilters.loadSegments(data.segments || []);

    // Cargar editor
    transcriptEditor.loadSegments(data.segments || []);
}

function renderCharts(data) {
    const timeline = data.emotion_timeline || [];
    const emotions = [...new Set(timeline.map(t => t.emotion))];
    
    const timelineChart = document.getElementById('timelineChart');
    if (timelineChart) {
        new Chart(timelineChart, {
            type: 'line',
            data: {
                labels: timeline.map(t => t.time.toFixed(1) + 's'),
                datasets: emotions.map(e => ({
                    label: e.toUpperCase(),
                    data: timeline.map(t => t.all_emotions?.[e] || 0),
                    borderColor: CONFIG.EMOTION_COLORS[e],
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    tension: 0.4
                }))
            },
            options: { 
                responsive: true, 
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-main') } }
                },
                scales: {
                    y: { ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-muted') } },
                    x: { ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-muted') } }
                }
            }
        });
    }

    const dist = data.global_emotions?.emotion_distribution || {};
    const distributionChart = document.getElementById('distributionChart');
    if (distributionChart) {
        new Chart(distributionChart, {
            type: 'doughnut',
            data: {
                labels: Object.keys(dist).map(k => k.toUpperCase()),
                datasets: [{
                    data: Object.values(dist),
                    backgroundColor: Object.keys(dist).map(k => CONFIG.EMOTION_COLORS[k]),
                    borderWidth: 0
                }]
            },
            options: { 
                responsive: true, 
                maintainAspectRatio: false,
                plugins: {
                    legend: { 
                        position: 'bottom',
                        labels: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-main') } 
                    }
                }
            }
        });
    }
}

function renderTranscript(segments) {
    const box = document.getElementById('transcriptBox');
    if (!box) return;

    box.innerHTML = segments.map((seg, idx) => `
        <div class="transcript-segment" onclick="audioPlayer.seekToTime(${seg.start})" data-index="${idx}">
            <strong>[${seg.start.toFixed(1)}s] ${seg.speaker_label || 'Hablante 1'}</strong> (${seg.emotion || 'neutral'})
            <br>
            ${seg.text_es || seg.text}
        </div>
    `).join('');
}

// History
async function saveToHistory(data, filename) {
    try {
        await API.post('/history/save', {
            filename: filename,
            timestamp: new Date().toISOString(),
            duration: data.metadata?.total_duration || 0,
            num_segments: (data.segments || []).length,
            dominant_emotion: data.global_emotions?.top_emotion || 'neutral',
            global_emotions: data.global_emotions,
            segments: data.segments,
            transcription: data.transcription
        });
    } catch (e) {
        console.error('Error guardando historial:', e);
    }
}

async function loadHistory() {
    try {
        const history = await API.get('/history/');
        const grid = document.getElementById('historyGrid');
        if (!grid) return;
        
        if (!Array.isArray(history) || history.length === 0) {
            grid.innerHTML = '<p style="text-align: center; color: var(--text-muted); padding: 3rem;">No hay transcripciones</p>';
            return;
        }

        grid.innerHTML = history.map(item => `
            <div class="history-item" onclick="loadHistoryItem('${item.id}')">
                <strong>🎙️ ${item.filename || 'Sin nombre'}</strong><br>
                <small>${Utils.formatDate(item.timestamp)}</small><br>
                <small>${CONFIG.EMOTIONS[item.dominant_emotion] || '😐'} ${item.dominant_emotion || 'neutral'} · 
                       ⏱ ${Utils.formatDuration(item.duration || 0)} · 
                       📝 ${item.num_segments || 0} segmentos</small>
            </div>
        `).join('');
    } catch (e) {
        console.error('Error cargando historial:', e);
    }
}

async function clearHistory() {
    if (!confirm('¿Borrar todo el historial?')) return;
    try {
        await API.delete('/history/clear');
        loadHistory();
        toast.success('Historial borrado');
    } catch (e) {
        toast.error('Error borrando historial');
    }
}

// Export
async function exportData(format) {
    if (!currentResults) {
        toast.warning('No hay resultados para exportar');
        return;
    }

    try {
        const res = await fetch(`${CONFIG.API_URL}/export/${format}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentResults)
        });

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcripcion_${Date.now()}.${format}`;
        a.click();
        URL.revokeObjectURL(url);
        
        toast.success(`Exportado a ${format.toUpperCase()}`);
    } catch (e) {
        toast.error('Error exportando: ' + e.message);
    }
}
