//CORE: variables globales y utilidades 

//variables globales
let selectedFile = null;
let currentResult = null;

//utilidades
const Utils = {
    formatTime(seconds){
        const mins = Math.floor(seconds/60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    },

    formatDate(dateString){
        const date = new Date(dateString);
        return date.toLocaleString('es-EC', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    },

    formatDuration(seconds){
        if(seconds<60)return '${seconds}s';
        if(seconds<3600)return '${Math.floor(seconds/60)}m ${Math.floor(seconds%60)}s';
        return '${Math.floor(seconds/3600)}h ${Math.floor((seconds%3600)/60)}m';
    },

    hexToRgba(hex, alpha){
        const r = parseInt(hex.slice(1,3),16);
        const g = parseInt(hex.slice(3,5),16);
        const b = parseInt(hex.slice(5,7),16);
        return `rgba(${r},${g},${b},${alpha})`;
    },

    debounce(func, wait){
        let timeout;
        return function(...args){
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), wait);
        };
    },

    downloadFile(filename, content, mimeType){
        const blob = new Blob([content], {type: mimeType});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    },
  
}

// Api helper

const API = {
        async get(endpoint) {
        const response = await fetch(`${CONFIG.API_URL}${endpoint}`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        return await response.json();
    },

    async post(endpoint, data) {
        const response = await fetch(`${CONFIG.API_URL}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        return await response.json();
    },

    async postFormData(endpoint, formData) {
        const response = await fetch(`${CONFIG.API_URL}${endpoint}`, {
            method: 'POST',
            body: formData
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        return await response.json();
    },

    async delete(endpoint) {
        const response = await fetch(`${CONFIG.API_URL}${endpoint}`, {
            method: 'DELETE'
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        return await response.json();
    }
}

