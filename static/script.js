let audioContext;
let processor;
let input;
let audioSamples = [];
let audioBlob;
let myChart;

async function toggleRecording() {
    const btn = document.getElementById('record-btn');
    const status = document.getElementById('status');

    if (btn.innerText.includes('Record')) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 22050 });
            input = audioContext.createMediaStreamSource(stream);
            processor = audioContext.createScriptProcessor(4096, 1, 1);

            audioSamples = [];

            processor.onaudioprocess = (e) => {
                const channelData = e.inputBuffer.getChannelData(0);
                audioSamples.push(new Float32Array(channelData));
            };

            input.connect(processor);
            processor.connect(audioContext.destination);

            btn.classList.add('recording');
            btn.classList.replace('btn-danger', 'btn-warning');
            btn.innerHTML = '<span>Stop Analysis</span>';
            status.innerText = "Listening... maintain consistent distance.";
        } catch (err) {
            console.error('Error starting recording:', err);
            status.innerText = "Mic access denied. Please use the upload option.";
        }
    } else {
        stopAndEncode();
        btn.classList.remove('recording');
        btn.classList.replace('btn-warning', 'btn-danger');
        btn.innerHTML = '<span>Record Live Cough</span>';
    }
}

function stopAndEncode() {
    if (processor) {
        processor.disconnect();
        input.disconnect();
        audioContext.close();
    }

    // Flatten samples
    let totalLength = audioSamples.reduce((acc, curr) => acc + curr.length, 0);
    let result = new Float32Array(totalLength);
    let offset = 0;
    for (let sample of audioSamples) {
        result.set(sample, offset);
        offset += sample.length;
    }

    // Encode to WAV
    audioBlob = encodeWAV(result, 22050);
    const audioUrl = URL.createObjectURL(audioBlob);
    const audioPlayback = document.getElementById('audio-playback');
    audioPlayback.src = audioUrl;
    audioPlayback.classList.remove('d-none');
    document.getElementById('analyze-btn').classList.remove('d-none');
    document.getElementById('status').innerText = "Recording finished. Analysis ready.";
}

function encodeWAV(samples, sampleRate) {
    let buffer = new ArrayBuffer(44 + samples.length * 2);
    let view = new DataView(buffer);

    /* RIFF identifier */
    writeString(view, 0, 'RIFF');
    /* file length */
    view.setUint32(4, 36 + samples.length * 2, true);
    /* RIFF type */
    writeString(view, 8, 'WAVE');
    /* format chunk identifier */
    writeString(view, 12, 'fmt ');
    /* format chunk length */
    view.setUint32(16, 16, true);
    /* sample format (raw) */
    view.setUint16(20, 1, true);
    /* channel count */
    view.setUint16(22, 1, true);
    /* sample rate */
    view.setUint32(24, sampleRate, true);
    /* byte rate (sample rate * block align) */
    view.setUint32(28, sampleRate * 2, true);
    /* block align (channel count * bytes per sample) */
    view.setUint16(32, 2, true);
    /* bits per sample */
    view.setUint16(34, 16, true);
    /* data chunk identifier */
    writeString(view, 36, 'data');
    /* data chunk length */
    view.setUint32(40, samples.length * 2, true);

    floatTo16BitPCM(view, 44, samples);

    return new Blob([view], { type: 'audio/wav' });
}

function floatTo16BitPCM(output, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, input[i]));
        output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

function loadFile(event) {
    const file = event.target.files[0];
    if (file) {
        audioBlob = file;
        const audioUrl = URL.createObjectURL(file);
        const audioPlayback = document.getElementById('audio-playback');
        audioPlayback.src = audioUrl;
        audioPlayback.classList.remove('d-none');
        document.getElementById('analyze-btn').classList.remove('d-none');
        document.getElementById('status').innerText = "File uploaded: " + file.name;
    }
}

async function analyzeAudio() {
    if (!audioBlob) return;
    document.getElementById('loading-overlay').classList.remove('d-none');

    const formData = new FormData();
    formData.append('audio', audioBlob, 'upload.wav');

    const timeoutId = setTimeout(() => {
        document.getElementById('loading-overlay').classList.add('d-none');
        alert("The scan is taking longer than expected. Please check your network.");
    }, 20000);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        clearTimeout(timeoutId);

        if (!response.ok) throw new Error(`Server responded with ${response.status}`);

        const data = await response.json();
        if (data.error) {
            alert("Analysis failed: " + data.error);
        } else {
            showResult(data);
        }
    } catch (err) {
        console.error('Analysis error:', err);
        alert("Server error occurred. Details: " + err.message);
    } finally {
        document.getElementById('loading-overlay').classList.add('d-none');
    }
}

function showResult(data) {
    document.getElementById('result-section').classList.remove('d-none');
    const labelEl = document.getElementById('prediction-label');
    labelEl.innerText = data.label;
    labelEl.style.transform = "scale(0.5)";
    setTimeout(() => {
        labelEl.style.transition = "transform 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275)";
        labelEl.style.transform = "scale(1)";
    }, 50);

    if (data.probabilities) updateChart(data.probabilities);
    document.getElementById('result-section').scrollIntoView({ behavior: 'smooth' });
}

function updateChart(probs) {
    const labels = Object.keys(probs);
    const values = Object.values(probs).map(v => v * 100);
    const ctx = document.getElementById('analysis-chart').getContext('2d');
    if (myChart) myChart.destroy();
    myChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Confidence (%)',
                data: values,
                backgroundColor: ['#6366f199', '#10b98199', '#f43f5e99', '#f59e0b99', '#8b5cf699', '#94a3b899'],
                borderColor: ['#6366f1', '#10b981', '#f43f5e', '#f59e0b', '#8b5cf6', '#94a3b8'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: { y: { beginAtZero: true, max: 100 } },
            plugins: { legend: { display: false } }
        }
    });
}
