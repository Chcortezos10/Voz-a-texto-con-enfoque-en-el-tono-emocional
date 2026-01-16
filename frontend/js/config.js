//frontend/js/config.js
//configuracion centralizada para el proyecto

const CONFIG = {
    //URL del backend
    API_URL: "http://127.0.0.1:8000",
    
    //Otras configuraciones
    MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
    ALLOWED_FILE_TYPES: ['audio/*','video/*'],

    //endpoints 
    ENDPOINTS: {
        HEALTH: '/health',
        TRANSCRIBE_LOCAL: '/transcribe/full-analysis',
        TRANSCRIBE_CLOUD: '/transcribe/cloud-whisper',
        HISTORY_LIST: '/history/',
        HISTORY_SAVE: '/history/save',
        HISTORY_CLEAR: '/history/clear',
        EXPORT: '/export',
    },
    
    EMOTIONS: {
        'feliz': '😊',
        'enojado': '😠',
        'triste': '😢',
        'neutral': '😐'
    },
    
    EMOTION_COLORS: {
        'feliz': '#10b981',
        'enojado': '#ef4444',
        'triste': '#3b82f6',
        'neutral': '#a78bfa'
    }
};

// Auto-detectar entorno
if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    CONFIG.API_URL = 'http://127.0.0.1:8000';
}