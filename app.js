const fashionClasses = [
	'T-shirt/top',
	'Trouser',
	'Pullover',
	'Dress',
	'Coat',
	'Sandal',
	'Shirt',
	'Sneaker',
	'Bag',
	'Ankle boot',
];
const shapeClasses = [
	'Cercle',
	'Heptagone',
	'Hexagone',
	'Nonagone',
	'Octogone',
	'Pentagone',
	'Carré',
	'Étoile',
	'Triangle',
];

let mnistSession = null;
let fashionSession = null;
let shapeSession = null;

document.getElementById('mnist-result').textContent = 'Modèle en cours de chargement...';
document.getElementById('fashion-result').textContent = 'Modèle en cours de chargement...';
document.getElementById('shape-result').textContent = 'Modèle en cours de chargement...';
document.getElementById('draw-result').textContent = 'Modèle en cours de chargement...';

async function loadModels() {
	try {
		mnistSession = await ort.InferenceSession.create('mnistcnn.onnx');
		document.getElementById('mnist-result').textContent = 'Modèle MNIST chargé !';
	} catch {
		document.getElementById('mnist-result').textContent = 'Erreur chargement modèle MNIST';
	}
	try {
		fashionSession = await ort.InferenceSession.create('fashioncnn.onnx');
		document.getElementById('fashion-result').textContent = 'Modèle FashionMNIST chargé !';
	} catch {
		document.getElementById('fashion-result').textContent = 'Erreur chargement modèle FashionMNIST';
	}
	try {
		shapeSession = await ort.InferenceSession.create('shapescnn.onnx');
		document.getElementById('shape-result').textContent = 'Modèle Formes chargé !';
		document.getElementById('draw-result').textContent = 'Modèle Formes chargé !';
	} catch {
		document.getElementById('shape-result').textContent = 'Erreur chargement modèle Formes';
		document.getElementById('draw-result').textContent = 'Erreur chargement modèle Formes';
	}
}
loadModels();

const mnistCanvas = document.getElementById('mnist-canvas');
const mnistCtx = mnistCanvas.getContext('2d');
let drawing = false;
mnistCtx.fillStyle = '#000';
mnistCtx.fillRect(0, 0, mnistCanvas.width, mnistCanvas.height);

mnistCanvas.addEventListener('mousedown', (e) => {
	drawing = true;
	mnistCtx.lineWidth = 24;
	mnistCtx.lineCap = 'round';
	mnistCtx.strokeStyle = '#fff';
	const rect = mnistCanvas.getBoundingClientRect();
	const x = e.clientX - rect.left;
	const y = e.clientY - rect.top;
	mnistCtx.beginPath();
	mnistCtx.moveTo(x, y);
});
mnistCanvas.addEventListener('mouseup', () => {
	drawing = false;
	mnistCtx.beginPath();
});
mnistCanvas.addEventListener('mouseleave', () => {
	drawing = false;
	mnistCtx.beginPath();
});
mnistCanvas.addEventListener('mousemove', function (e) {
	if (!drawing) return;
	const rect = mnistCanvas.getBoundingClientRect();
	const x = e.clientX - rect.left;
	const y = e.clientY - rect.top;
	mnistCtx.lineTo(x, y);
	mnistCtx.stroke();
	mnistCtx.beginPath();
	mnistCtx.moveTo(x, y);
});
document.getElementById('mnist-clear-btn').addEventListener('click', () => {
	mnistCtx.fillStyle = '#000';
	mnistCtx.fillRect(0, 0, mnistCanvas.width, mnistCanvas.height);
	document.getElementById('mnist-result').textContent = '';
});
document.getElementById('mnist-predict-btn').addEventListener('click', async function () {
	const resultDiv = document.getElementById('mnist-result');
	if (!mnistSession) {
		resultDiv.textContent = 'Modèle non chargé...';
		resultDiv.style.color = '#e53e3e';
		return;
	}
	resultDiv.textContent = 'Prédiction en cours...';
	const smallCanvas = document.createElement('canvas');
	smallCanvas.width = 28;
	smallCanvas.height = 28;
	const smallCtx = smallCanvas.getContext('2d');
	smallCtx.drawImage(mnistCanvas, 0, 0, 28, 28);
	let imageData = smallCtx.getImageData(0, 0, 28, 28).data;
	let input = new Float32Array(1 * 1 * 28 * 28);
	for (let i = 0; i < 28 * 28; ++i) {
		let pixel = imageData[i * 4];
		input[i] = (pixel / 255 - 0.5) / 0.5;
	}
	const inputTensor = new ort.Tensor('float32', input, [1, 1, 28, 28]);
	try {
		const feeds = { input: inputTensor };
		const results = await mnistSession.run(feeds);
		const output = results.output.data;
		const maxIdx = output.indexOf(Math.max(...output));
		resultDiv.textContent = `Chiffre prédit : ${maxIdx}`;
		resultDiv.style.color = '#4299e1';
	} catch (err) {
		resultDiv.textContent = 'Erreur lors de la prédiction : ' + err;
		resultDiv.style.color = '#e53e3e';
	}
});

document.getElementById('fashion-image-input').addEventListener('change', function (e) {
	const file = e.target.files[0];
	if (!file) return;
	const reader = new FileReader();
	reader.onload = function (ev) {
		const img = new window.Image();
		img.onload = function () {
			const canvas = document.getElementById('fashion-canvas');
			const ctx = canvas.getContext('2d', { willReadFrequently: true });
			ctx.clearRect(0, 0, 28, 28);
			ctx.drawImage(img, 0, 0, 28, 28);
			let imageData = ctx.getImageData(0, 0, 28, 28);
			for (let i = 0; i < imageData.data.length; i += 4) {
				let avg = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
				let inverted = 255 - avg;
				let bw = inverted > 128 ? 255 : 0;
				imageData.data[i] = imageData.data[i + 1] = imageData.data[i + 2] = bw;
			}
			ctx.putImageData(imageData, 0, 0);
		};
		img.src = ev.target.result;
	};
	reader.readAsDataURL(file);
});
document.getElementById('fashion-predict-btn').addEventListener('click', async function () {
	const resultDiv = document.getElementById('fashion-result');
	if (!fashionSession) {
		resultDiv.textContent = 'Modèle non chargé...';
		resultDiv.style.color = '#e53e3e';
		return;
	}
	resultDiv.textContent = 'Prédiction en cours...';
	const ctx = document
		.getElementById('fashion-canvas')
		.getContext('2d', { willReadFrequently: true });
	const imageData = ctx.getImageData(0, 0, 28, 28).data;
	let input = new Float32Array(1 * 1 * 28 * 28);
	for (let i = 0; i < 28 * 28; ++i) {
		let pixel = imageData[i * 4];
		input[i] = (pixel / 255 - 0.5) / 0.5;
	}
	const inputTensor = new ort.Tensor('float32', input, [1, 1, 28, 28]);
	try {
		const feeds = { input: inputTensor };
		const results = await fashionSession.run(feeds);
		const output = results.output.data;
		const maxIdx = output.indexOf(Math.max(...output));
		resultDiv.textContent = `Classe prédite : ${fashionClasses[maxIdx]}`;
		resultDiv.style.color = '#4299e1';
	} catch (err) {
		resultDiv.textContent = 'Erreur lors de la prédiction : ' + err;
		resultDiv.style.color = '#e53e3e';
	}
});

document.getElementById('shape-image-input').addEventListener('change', function (e) {
	const file = e.target.files[0];
	if (!file) return;
	const reader = new FileReader();
	reader.onload = function (ev) {
		const img = new window.Image();
		img.onload = function () {
			const canvas = document.getElementById('shape-canvas');
			const ctx = canvas.getContext('2d', { willReadFrequently: true });
			ctx.clearRect(0, 0, 64, 64);
			ctx.drawImage(img, 0, 0, 64, 64);
		};
		img.src = ev.target.result;
	};
	reader.readAsDataURL(file);
});
document.getElementById('shape-predict-btn').addEventListener('click', async function () {
	const resultDiv = document.getElementById('shape-result');
	if (!shapeSession) {
		resultDiv.textContent = 'Modèle non chargé...';
		resultDiv.style.color = '#e53e3e';
		return;
	}
	resultDiv.textContent = 'Prédiction en cours...';
	const ctx = document
		.getElementById('shape-canvas')
		.getContext('2d', { willReadFrequently: true });
	const imageData = ctx.getImageData(0, 0, 64, 64).data;
	let input = new Float32Array(1 * 3 * 64 * 64);
	for (let i = 0; i < 64 * 64; ++i) {
		input[i] = (imageData[i * 4] / 255 - 0.5) / 0.5;
		input[64 * 64 + i] = (imageData[i * 4 + 1] / 255 - 0.5) / 0.5;
		input[2 * 64 * 64 + i] = (imageData[i * 4 + 2] / 255 - 0.5) / 0.5;
	}
	const inputTensor = new ort.Tensor('float32', input, [1, 3, 64, 64]);
	try {
		const feeds = { input: inputTensor };
		const results = await shapeSession.run(feeds);
		const output = results.output.data;
		const maxIdx = output.indexOf(Math.max(...output));
		resultDiv.textContent = `Forme prédite : ${shapeClasses[maxIdx]}`;
		resultDiv.style.color = '#4299e1';
	} catch (err) {
		resultDiv.textContent = 'Erreur lors de la prédiction : ' + err;
		resultDiv.style.color = '#e53e3e';
	}
});

const drawCanvas = document.getElementById('draw-canvas');
const drawCtx = drawCanvas.getContext('2d');
let drawingShape = false;
drawCanvas.addEventListener('mousedown', (e) => {
	drawingShape = true;
	drawCtx.lineWidth = 24;
	drawCtx.lineCap = 'round';
	drawCtx.strokeStyle = '#fff';
	const rect = drawCanvas.getBoundingClientRect();
	const x = e.clientX - rect.left;
	const y = e.clientY - rect.top;
	drawCtx.beginPath();
	drawCtx.moveTo(x, y);
});
drawCanvas.addEventListener('mouseup', () => {
	drawingShape = false;
	drawCtx.beginPath();
});
drawCanvas.addEventListener('mouseleave', () => {
	drawingShape = false;
	drawCtx.beginPath();
});
drawCanvas.addEventListener('mousemove', function (e) {
	if (!drawingShape) return;
	const rect = drawCanvas.getBoundingClientRect();
	const x = e.clientX - rect.left;
	const y = e.clientY - rect.top;
	drawCtx.lineTo(x, y);
	drawCtx.stroke();
	drawCtx.beginPath();
	drawCtx.moveTo(x, y);
});
document.getElementById('draw-clear-btn').addEventListener('click', () => {
	drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
	document.getElementById('draw-result').textContent = '';
});
document.getElementById('draw-predict-btn').addEventListener('click', async function () {
	const resultDiv = document.getElementById('draw-result');
	if (!shapeSession) {
		resultDiv.textContent = 'Modèle non chargé...';
		resultDiv.style.color = '#e53e3e';
		return;
	}
	resultDiv.textContent = 'Prédiction en cours...';
	const smallCanvas = document.createElement('canvas');
	smallCanvas.width = 64;
	smallCanvas.height = 64;
	const smallCtx = smallCanvas.getContext('2d');
	smallCtx.drawImage(drawCanvas, 0, 0, 64, 64);
	let imageData = smallCtx.getImageData(0, 0, 64, 64).data;
	let input = new Float32Array(1 * 3 * 64 * 64);
	for (let i = 0; i < 64 * 64; ++i) {
		input[i] = (imageData[i * 4] / 255 - 0.5) / 0.5;
		input[64 * 64 + i] = (imageData[i * 4 + 1] / 255 - 0.5) / 0.5;
		input[2 * 64 * 64 + i] = (imageData[i * 4 + 2] / 255 - 0.5) / 0.5;
	}
	const inputTensor = new ort.Tensor('float32', input, [1, 3, 64, 64]);
	try {
		const feeds = { input: inputTensor };
		const results = await shapeSession.run(feeds);
		const output = results.output.data;
		const maxIdx = output.indexOf(Math.max(...output));
		resultDiv.textContent = `Forme prédite : ${shapeClasses[maxIdx]}`;
		resultDiv.style.color = '#4299e1';
	} catch (err) {
		resultDiv.textContent = 'Erreur lors de la prédiction : ' + err;
		resultDiv.style.color = '#e53e3e';
	}
});

const themeToggle = document.getElementById('theme-toggle');
const themeIcon = document.getElementById('theme-icon');
function setTheme(dark) {
	document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
	themeIcon.textContent = dark ? '☀️' : '🌙';
	localStorage.setItem('theme', dark ? 'dark' : 'light');
}
themeToggle.addEventListener('click', () => {
	const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
	setTheme(!isDark);
});
(function () {
	const stored = localStorage.getItem('theme');
	const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
	setTheme(stored ? stored === 'dark' : prefersDark);
})();

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

let newsSession = null;
let newsVectorizer = null;

async function loadNewsModel() {
    try {
        newsSession = await ort.InferenceSession.create('news_classifier.onnx');
        const resp = await fetch('news_vectorizer.json');
        newsVectorizer = await resp.json();
        document.getElementById('news-result').textContent = 'Modèle News chargé !';
    } catch (err) {
        document.getElementById('news-result').textContent = 'Erreur chargement modèle News';
        console.error(err);
    }
}
loadNewsModel();

function vectorize(text, vectorizer) {
    text = text.toLowerCase().replace(/[^a-z0-9 ]/g, ' ');
    const words = text.split(/\s+/);
    const features = vectorizer.vocabulary;
    const vec = new Float32Array(features.length);
    for (let i = 0; i < features.length; ++i) {
        const word = features[i];
        vec[i] = words.filter(w => w === word).length * vectorizer.idf[i];
    }
    return vec;
}

const newsLabels = ['World', 'Sports', 'Business', 'Sci/Tech'];

document.getElementById('news-predict-btn').addEventListener('click', async function () {
    const input = document.getElementById('news-input').value.trim();
    const resultDiv = document.getElementById('news-result');
    if (!input) {
        resultDiv.textContent = "Veuillez entrer un titre d'article.";
        return;
    }
    if (!newsSession || !newsVectorizer) {
        resultDiv.textContent = 'Modèle non chargé...';
        return;
    }
    resultDiv.textContent = 'Prédiction en cours...';
    const inputVec = vectorize(input, newsVectorizer);
    const inputTensor = new ort.Tensor('float32', inputVec, [1, inputVec.length]);
    try {
        const feeds = { input: inputTensor };
        const results = await newsSession.run(feeds);
        const output = results.output.data;
        const maxIdx = output.indexOf(Math.max(...output));
        resultDiv.textContent = `Catégorie : ${newsLabels[maxIdx]}`;
        resultDiv.style.color = '#4299e1';
    } catch (err) {
        resultDiv.textContent = 'Erreur lors de la prédiction : ' + err;
        resultDiv.style.color = '#e53e3e';
    }
});

document.querySelectorAll('.example-news-btn').forEach((btn) => {
    btn.addEventListener('click', function () {
        document.getElementById('news-input').value = this.textContent;
    });
});

