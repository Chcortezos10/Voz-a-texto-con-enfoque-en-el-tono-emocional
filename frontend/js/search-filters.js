//busqueda avanzada y filtros

class SearchFilters {
    constructor() {
        this.allResults = [];
        this.currentSegments = [];
    }

    createFilterUI() {
        const filterHTML = `
            <div id="searchFiltersPanel" class="filters-panel" style="display: none;">
                <div class="filters-header">
                    <h3><i class="fas fa-filter"></i> Filtros y Búsqueda</h3>
                    <button class="btn btn-secondary btn-sm" onclick="searchFilters.clearFilters()">
                        <i class="fas fa-times"></i> Limpiar
                    </button>
                </div>
                
                <div class="filters-content">
                    <div class="filter-group">
                        <label><i class="fas fa-search"></i> Buscar en transcripción:</label>
                        <input type="text" id="searchText" placeholder="Buscar palabras..." 
                               class="filter-input" />
                    </div>

                    <div class="filter-group">
                        <label><i class="fas fa-smile"></i> Emoción:</label>
                        <div class="filter-chips">
                            <button class="filter-chip active" data-emotion="all">Todas</button>
                            <button class="filter-chip" data-emotion="feliz">😊 Feliz</button>
                            <button class="filter-chip" data-emotion="enojado">😠 Enojado</button>
                            <button class="filter-chip" data-emotion="triste">😢 Triste</button>
                            <button class="filter-chip" data-emotion="neutral">😐 Neutral</button>
                        </div>
                    </div>

                    <div class="filter-group">
                        <label><i class="fas fa-user"></i> Hablante:</label>
                        <select id="speakerFilter" class="filter-input">
                            <option value="all">Todos</option>
                        </select>
                    </div>

                    <div class="filter-group">
                        <label><i class="fas fa-clock"></i> Rango de tiempo:</label>
                        <div style="display: flex; gap: 0.5rem; align-items: center;">
                            <input type="number" id="timeFrom" placeholder="Desde (s)" 
                                   class="filter-input" style="width: 100px;" />
                            <span>-</span>
                            <input type="number" id="timeTo" placeholder="Hasta (s)" 
                                   class="filter-input" style="width: 100px;" />
                        </div>
                    </div>

                    <div class="filter-group">
                        <label><i class="fas fa-chart-line"></i> Intensidad mínima: <span id="intensityValue">0%</span></label>
                        <input type="range" id="intensityFilter" min="0" max="100" value="0" />
                    </div>

                    <div class="filter-results">
                        <span id="filterResultsCount">0 segmentos</span>
                    </div>
                </div>
            </div>
        `;

        const resultsSection = document.getElementById('resultsSection');
        if (resultsSection) {
            resultsSection.insertAdjacentHTML('beforebegin', filterHTML);
            this.bindEvents();
        }
    }

    bindEvents() {
        const searchText = document.getElementById('searchText');
        const speakerFilter = document.getElementById('speakerFilter');
        const timeFrom = document.getElementById('timeFrom');
        const timeTo = document.getElementById('timeTo');
        const intensityFilter = document.getElementById('intensityFilter');

        if (searchText) searchText.oninput = Utils.debounce(() => this.applyFilters(), 300);
        if (speakerFilter) speakerFilter.onchange = () => this.applyFilters();
        if (timeFrom) timeFrom.oninput = Utils.debounce(() => this.applyFilters(), 500);
        if (timeTo) timeTo.oninput = Utils.debounce(() => this.applyFilters(), 500);
        if (intensityFilter) {
            intensityFilter.oninput = (e) => {
                document.getElementById('intensityValue').textContent = e.target.value + '%';
                this.applyFilters();
            };
        }

        document.querySelectorAll('.filter-chip').forEach(chip => {
            chip.onclick = () => this.filterByEmotion(chip.dataset.emotion);
        });
    }

    loadSegments(segments) {
        if (!document.getElementById('searchFiltersPanel')) {
            this.createFilterUI();
        }

        this.currentSegments = segments || [];
        this.allResults = segments || [];
        
        const speakers = [...new Set(segments.map(s => s.speaker_label))];
        const speakerSelect = document.getElementById('speakerFilter');
        if (speakerSelect) {
            speakerSelect.innerHTML = '<option value="all">Todos</option>' +
                speakers.map(s => `<option value="${s}">${s}</option>`).join('');
        }
        
        const panel = document.getElementById('searchFiltersPanel');
        if (panel) panel.style.display = 'block';
        
        this.clearFilters();
    }

    applyFilters() {
        let filtered = [...this.allResults];

        const searchText = document.getElementById('searchText')?.value.toLowerCase();
        if (searchText) {
            filtered = filtered.filter(seg => 
                (seg.text_es || seg.text || '').toLowerCase().includes(searchText)
            );
        }

        const selectedEmotion = document.querySelector('.filter-chip.active')?.dataset.emotion;
        if (selectedEmotion && selectedEmotion !== 'all') {
            filtered = filtered.filter(seg => seg.emotion === selectedEmotion);
        }

        const speaker = document.getElementById('speakerFilter')?.value;
        if (speaker && speaker !== 'all') {
            filtered = filtered.filter(seg => seg.speaker_label === speaker);
        }

        const timeFrom = parseFloat(document.getElementById('timeFrom')?.value);
        const timeTo = parseFloat(document.getElementById('timeTo')?.value);
        if (!isNaN(timeFrom)) {
            filtered = filtered.filter(seg => seg.start >= timeFrom);
        }
        if (!isNaN(timeTo)) {
            filtered = filtered.filter(seg => seg.end <= timeTo);
        }

        const minIntensity = parseFloat(document.getElementById('intensityFilter')?.value || 0) / 100;
        filtered = filtered.filter(seg => (seg.intensity || 0) >= minIntensity);

        this.currentSegments = filtered;
        if (window.renderTranscript) {
            renderTranscript(filtered);
        }
        
        const counter = document.getElementById('filterResultsCount');
        if (counter) {
            counter.textContent = `${filtered.length} de ${this.allResults.length} segmentos`;
        }

        if (searchText) {
            this.highlightSearchResults(searchText);
        }
    }

    filterByEmotion(emotion) {
        document.querySelectorAll('.filter-chip').forEach(chip => {
            chip.classList.remove('active');
        });
        const chip = document.querySelector(`[data-emotion="${emotion}"]`);
        if (chip) chip.classList.add('active');
        
        this.applyFilters();
    }

    highlightSearchResults(searchText) {
        document.querySelectorAll('.transcript-segment').forEach(segment => {
            let html = segment.innerHTML;
            const regex = new RegExp(`(${searchText})`, 'gi');
            html = html.replace(/<mark>(.*?)<\/mark>/gi, '$1');
            segment.innerHTML = html.replace(regex, '<mark>$1</mark>');
        });
    }

    clearFilters() {
        const searchText = document.getElementById('searchText');
        const speakerFilter = document.getElementById('speakerFilter');
        const timeFrom = document.getElementById('timeFrom');
        const timeTo = document.getElementById('timeTo');
        const intensityFilter = document.getElementById('intensityFilter');
        const intensityValue = document.getElementById('intensityValue');

        if (searchText) searchText.value = '';
        if (speakerFilter) speakerFilter.value = 'all';
        if (timeFrom) timeFrom.value = '';
        if (timeTo) timeTo.value = '';
        if (intensityFilter) intensityFilter.value = '0';
        if (intensityValue) intensityValue.textContent = '0%';
        
        document.querySelectorAll('.filter-chip').forEach(chip => chip.classList.remove('active'));
        const allChip = document.querySelector('[data-emotion="all"]');
        if (allChip) allChip.classList.add('active');
        
        this.currentSegments = this.allResults;
        if (window.renderTranscript) {
            renderTranscript(this.allResults);
        }
        
        const counter = document.getElementById('filterResultsCount');
        if (counter) {
            counter.textContent = `${this.allResults.length} segmentos`;
        }
    }
}