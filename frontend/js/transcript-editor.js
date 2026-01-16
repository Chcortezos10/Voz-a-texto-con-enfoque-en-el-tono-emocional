// editor de transcripciones 

class TranscriptEditor {
    constructor() {
        this.segments = [];
        this.editedSegments = new Set();
        this.sessionId = null;
        this.isEditMode = false;
    }

    loadSegments(segments, sessionId = null) {
        this.segments = segments || [];
        this.sessionId = sessionId;
        this.editedSegments.clear();
    }

    enableEditMode() {
        this.isEditMode = true;
        this.renderEditableTranscript();
    }

    renderEditableTranscript() {
        const box = document.getElementById('transcriptBox');
        if (!box) return;

        box.innerHTML = this.segments.map((seg, idx) => `
            <div class="transcript-segment editable-segment" data-index="${idx}">
                <div class="segment-header">
                    <div class="segment-time" onclick="audioPlayer.seekToTime(${seg.start})">
                        <i class="fas fa-play-circle"></i> [${seg.start.toFixed(1)}s]
                    </div>
                    <input type="text" 
                           class="speaker-input ${seg._speaker_edited ? 'edited' : ''}" 
                           value="${seg.speaker_label || 'Hablante 1'}" 
                           data-index="${idx}"
                           title="Editar nombre de hablante" />
                    <select class="emotion-select ${seg._emotion_edited ? 'edited' : ''}" 
                            data-index="${idx}">
                        <option value="feliz" ${seg.emotion === 'feliz' ? 'selected' : ''}>😊 Feliz</option>
                        <option value="enojado" ${seg.emotion === 'enojado' ? 'selected' : ''}>😠 Enojado</option>
                        <option value="triste" ${seg.emotion === 'triste' ? 'selected' : ''}>😢 Triste</option>
                        <option value="neutral" ${seg.emotion === 'neutral' ? 'selected' : ''}>😐 Neutral</option>
                    </select>
                    <div class="segment-actions">
                        <button class="action-btn" data-index="${idx}" data-action="split" 
                                title="Dividir segmento">
                            <i class="fas fa-cut"></i>
                        </button>
                        <button class="action-btn" data-index="${idx}" data-action="delete" 
                                title="Eliminar segmento">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
                <textarea class="segment-text ${seg._edited ? 'edited' : ''}" 
                          data-index="${idx}"
                          rows="2">${seg.text_es || seg.text}</textarea>
                ${seg._edited || seg._speaker_edited || seg._emotion_edited ? 
                    '<div class="edited-badge"><i class="fas fa-pencil-alt"></i> Editado</div>' : ''}
            </div>
        `).join('');

        box.insertAdjacentHTML('beforeend', `
            <div class="editor-actions">
                <button class="btn btn-primary" id="saveChangesBtn">
                    <i class="fas fa-save"></i> Guardar Cambios
                </button>
                <button class="btn btn-secondary" id="exportEditedBtn">
                    <i class="fas fa-download"></i> Exportar Editado
                </button>
                <button class="btn btn-secondary" id="undoAllBtn">
                    <i class="fas fa-undo"></i> Deshacer Todo
                </button>
                <div class="edit-counter">
                    <i class="fas fa-pencil-alt"></i> ${this.editedSegments.size} segmentos editados
                </div>
            </div>
        `);

        this.bindEditEvents();
    }

    bindEditEvents() {
        document.querySelectorAll('.speaker-input').forEach(input => {
            input.onchange = (e) => {
                const idx = parseInt(e.target.dataset.index);
                this.updateSpeaker(idx, e.target.value);
            };
        });

        document.querySelectorAll('.emotion-select').forEach(select => {
            select.onchange = (e) => {
                const idx = parseInt(e.target.dataset.index);
                this.updateEmotion(idx, e.target.value);
            };
        });

        document.querySelectorAll('.segment-text').forEach(textarea => {
            textarea.onchange = (e) => {
                const idx = parseInt(e.target.dataset.index);
                this.updateText(idx, e.target.value);
            };
        });

        document.querySelectorAll('.action-btn').forEach(btn => {
            btn.onclick = (e) => {
                const idx = parseInt(e.currentTarget.dataset.index);
                const action = e.currentTarget.dataset.action;
                
                if (action === 'split') {
                    this.splitSegment(idx);
                } else if (action === 'delete') {
                    this.deleteSegment(idx);
                }
            };
        });

        const saveBtn = document.getElementById('saveChangesBtn');
        const exportBtn = document.getElementById('exportEditedBtn');
        const undoBtn = document.getElementById('undoAllBtn');

        if (saveBtn) saveBtn.onclick = () => this.saveChanges();
        if (exportBtn) exportBtn.onclick = () => this.exportEdited();
        if (undoBtn) undoBtn.onclick = () => this.undoAll();
    }

    updateText(index, newText) {
        this.segments[index].text_es = newText;
        this.segments[index].text = newText;
        this.segments[index]._edited = true;
        this.editedSegments.add(index);
        this.updateEditCounter();
        if (window.toast) window.toast.info('Texto actualizado');
    }

    updateSpeaker(index, newSpeaker) {
        this.segments[index].speaker_label = newSpeaker;
        this.segments[index]._speaker_edited = true;
        this.editedSegments.add(index);
        this.updateEditCounter();
        if (window.toast) window.toast.info('Hablante actualizado');
    }

    updateEmotion(index, newEmotion) {
        this.segments[index].emotion = newEmotion;
        this.segments[index]._emotion_edited = true;
        this.editedSegments.add(index);
        this.updateEditCounter();
        if (window.toast) window.toast.info('Emoción actualizada');
    }

    splitSegment(index) {
        const segment = this.segments[index];
        const midpoint = (segment.start + segment.end) / 2;
        
        const firstHalf = { ...segment, end: midpoint };
        const secondHalf = { ...segment, start: midpoint };
        
        this.segments.splice(index, 1, firstHalf, secondHalf);
        this.renderEditableTranscript();
        if (window.toast) window.toast.success('Segmento dividido');
    }

    deleteSegment(index) {
        if (!confirm('¿Eliminar este segmento?')) return;
        
        this.segments.splice(index, 1);
        this.renderEditableTranscript();
        if (window.toast) window.toast.success('Segmento eliminado');
    }

    async saveChanges() {
        if (this.editedSegments.size === 0) {
            if (window.toast) window.toast.warning('No hay cambios para guardar');
            return;
        }

        try {
            if (this.sessionId) {
                await API.post(`/sessions/${this.sessionId}`, {
                    data: JSON.stringify({ segments: this.segments })
                });
            }

            if (window.currentResults) {
                currentResults.segments = this.segments;
            }
            
            this.editedSegments.clear();
            this.updateEditCounter();
            
            if (window.toast) window.toast.success('Cambios guardados correctamente');
        } catch (e) {
            if (window.toast) window.toast.error('Error guardando: ' + e.message);
        }
    }

    exportEdited() {
        const data = {
            ...currentResults,
            segments: this.segments,
            _metadata: {
                edited: true,
                editedAt: new Date().toISOString(),
                editedSegments: Array.from(this.editedSegments)
            }
        };

        Utils.downloadFile(
            JSON.stringify(data, null, 2),
            `transcripcion_editada_${Date.now()}.json`,
            'application/json'
        );
        
        if (window.toast) window.toast.success('Transcripción editada exportada');
    }

    undoAll() {
        if (!confirm('¿Deshacer todos los cambios?')) return;
        
        this.segments.forEach(seg => {
            delete seg._edited;
            delete seg._speaker_edited;
            delete seg._emotion_edited;
        });
        
        this.editedSegments.clear();
        this.renderEditableTranscript();
        if (window.toast) window.toast.info('Cambios deshechos');
    }

    updateEditCounter() {
        const counter = document.querySelector('.edit-counter');
        if (counter) {
            counter.innerHTML = `<i class="fas fa-pencil-alt"></i> ${this.editedSegments.size} segmentos editados`;
        }
    }
}