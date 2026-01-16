// reproductor de audio integrado 

class AudioPlayer {
    constructor() {
        this.audio = null;
        this.currentSegmentIndex = -1;
        this.segments = [];
        this.isPlaying = false;
    }

    createPlayerUI() {
        const playerHTML = `
            <div id="audioPlayer" class="audio-player" style="display: none;">
                <div class="player-controls">
                    <button id="playPauseBtn" class="player-btn" title="Reproducir/Pausar">
                        <i class="fas fa-play"></i>
                    </button>
                    <button id="rewindBtn" class="player-btn" title="Retroceder 5s">
                        <i class="fas fa-backward"></i>
                    </button>
                    <button id="forwardBtn" class="player-btn" title="Adelantar 5s">
                        <i class="fas fa-forward"></i>
                    </button>
                    <div class="time-display">
                        <span id="currentTime">0:00</span>
                        <span>/</span>
                        <span id="totalTime">0:00</span>
                    </div>
                </div>
                <div class="player-progress">
                    <input type="range" id="seekBar" min="0" max="100" value="0" step="0.1" />
                </div>
                <div class="player-extras">
                    <div class="speed-control">
                        <label><i class="fas fa-tachometer-alt"></i> Velocidad:</label>
                        <select id="playbackSpeed">
                            <option value="0.5">0.5x</option>
                            <option value="0.75">0.75x</option>
                            <option value="1" selected>1x</option>
                            <option value="1.25">1.25x</option>
                            <option value="1.5">1.5x</option>
                            <option value="2">2x</option>
                        </select>
                    </div>
                    <div class="volume-control">
                        <i class="fas fa-volume-up"></i>
                        <input type="range" id="volumeSlider" min="0" max="100" value="100" />
                    </div>
                    <div id="currentSegmentInfo" class="segment-info">
                        <span class="segment-speaker"></span>
                        <span class="segment-emotion"></span>
                    </div>
                </div>
            </div>
        `;

        const resultsSection = document.getElementById('resultsSection');
        if (resultsSection) {
            resultsSection.insertAdjacentHTML('afterbegin', playerHTML);
            this.bindEvents();
        }
    }

    bindEvents() {
        const playPauseBtn = document.getElementById('playPauseBtn');
        const rewindBtn = document.getElementById('rewindBtn');
        const forwardBtn = document.getElementById('forwardBtn');
        const seekBar = document.getElementById('seekBar');
        const playbackSpeed = document.getElementById('playbackSpeed');
        const volumeSlider = document.getElementById('volumeSlider');

        if (playPauseBtn) playPauseBtn.onclick = () => this.togglePlay();
        if (rewindBtn) rewindBtn.onclick = () => this.skip(-5);
        if (forwardBtn) forwardBtn.onclick = () => this.skip(5);
        if (seekBar) seekBar.oninput = (e) => this.seek(e.target.value);
        if (playbackSpeed) playbackSpeed.onchange = (e) => this.setSpeed(e.target.value);
        if (volumeSlider) volumeSlider.oninput = (e) => this.setVolume(e.target.value);
    }

    loadAudio(audioFile, segments) {
        if (!document.getElementById('audioPlayer')) {
            this.createPlayerUI();
        }

        if (this.audio) {
            this.audio.pause();
            this.audio = null;
        }

        this.segments = segments || [];
        this.audio = new Audio();
        
        const url = URL.createObjectURL(audioFile);
        this.audio.src = url;

        this.audio.onloadedmetadata = () => {
            const totalTime = document.getElementById('totalTime');
            const seekBar = document.getElementById('seekBar');
            if (totalTime) totalTime.textContent = Utils.formatTime(this.audio.duration);
            if (seekBar) seekBar.max = this.audio.duration;
        };

        this.audio.ontimeupdate = () => {
            const current = this.audio.currentTime;
            const currentTime = document.getElementById('currentTime');
            const seekBar = document.getElementById('seekBar');
            
            if (currentTime) currentTime.textContent = Utils.formatTime(current);
            if (seekBar) seekBar.value = current;
            
            this.updateCurrentSegment(current);
        };

        this.audio.onended = () => {
            this.isPlaying = false;
            const btn = document.getElementById('playPauseBtn');
            if (btn) btn.innerHTML = '<i class="fas fa-play"></i>';
        };

        const player = document.getElementById('audioPlayer');
        if (player) player.style.display = 'block';
        
        if (window.toast) {
            window.toast.success('Audio cargado. Haz clic en los segmentos para saltar.');
        }
    }

    togglePlay() {
        if (!this.audio) return;

        const btn = document.getElementById('playPauseBtn');
        
        if (this.isPlaying) {
            this.audio.pause();
            if (btn) btn.innerHTML = '<i class="fas fa-play"></i>';
        } else {
            this.audio.play();
            if (btn) btn.innerHTML = '<i class="fas fa-pause"></i>';
        }
        this.isPlaying = !this.isPlaying;
    }

    skip(seconds) {
        if (!this.audio) return;
        this.audio.currentTime = Math.max(0, Math.min(this.audio.duration, this.audio.currentTime + seconds));
    }

    seek(time) {
        if (!this.audio) return;
        this.audio.currentTime = parseFloat(time);
    }

    seekToTime(seconds) {
        if (!this.audio) return;
        this.audio.currentTime = seconds;
        if (!this.isPlaying) {
            this.togglePlay();
        }
    }

    setSpeed(speed) {
        if (!this.audio) return;
        this.audio.playbackRate = parseFloat(speed);
        if (window.toast) {
            window.toast.info(`Velocidad: ${speed}x`);
        }
    }

    setVolume(volume) {
        if (!this.audio) return;
        this.audio.volume = volume / 100;
    }

    updateCurrentSegment(currentTime) {
        const segmentIndex = this.segments.findIndex(seg => 
            currentTime >= seg.start && currentTime <= seg.end
        );

        if (segmentIndex !== this.currentSegmentIndex) {
            this.currentSegmentIndex = segmentIndex;
            
            document.querySelectorAll('.transcript-segment').forEach(seg => {
                seg.classList.remove('active-segment');
            });

            if (segmentIndex >= 0) {
                const segment = this.segments[segmentIndex];
                
                const segmentElement = document.querySelectorAll('.transcript-segment')[segmentIndex];
                if (segmentElement) {
                    segmentElement.classList.add('active-segment');
                    segmentElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }

                const speakerEl = document.querySelector('.segment-speaker');
                const emotionEl = document.querySelector('.segment-emotion');
                
                if (speakerEl) speakerEl.textContent = segment.speaker_label || 'Hablante 1';
                if (emotionEl) {
                    const emoji = CONFIG.EMOTIONS[segment.emotion] || '😐';
                    emotionEl.textContent = `${emoji} ${segment.emotion || 'neutral'}`;
                }
            }
        }
    }
}