const MATERIAL_LABELS_KOR = {
  glass: "유리",
  paper: "종이",
  can: "캔",
  vinyl: "비닐",
  styrofoam: "스티로폼",
  plastic: "플라스틱",
  unknown: "판정불가"
};

const CONTAMINATION_LABELS_KOR = {
  clean: "깨끗함",
  dirty: "오염됨",
  uncertain: "판정불가"
};

const MATERIAL_KEYWORDS = {
  glass: ["유리", "glass"],
  paper: ["종이", "paper"],
  can: ["캔", "can", "알루미늄", "aluminum"],
  vinyl: ["비닐", "vinyl"],
  styrofoam: ["스티로폼", "styrofoam", "스티로폼박스"],
  plastic: ["플라스틱", "plastic", "pet", "페트"]
};

const CONTAMINATION_KEYWORDS = {
  dirty: [
    "오염o",
    "오염 o",
    "오염O",
    "오염 O",
    "오염0",
    "오염 0",
    "오염됨",
    "dirty"
  ],
  clean: [
    "오염x",
    "오염 x",
    "오염X",
    "오염 X",
    "깨끗",
    "clean"
  ]
};

const DEFAULT_ZIP_FILES = [
  "비닐 오염o.zip",
  "비닐 오염x.zip",
  "스티로폼 오염0.zip",
  "스티로폼 오염x.zip",
  "유리 오염0.zip",
  "유리 오염x.zip",
  "종이 오염0.zip",
  "종이 오염x.zip",
  "캔 오염o.zip",
  "캔 오염0.zip",
  "캔 오염x.zip"
];

const IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"];

const MODEL_KEY = "indexeddb://recycle-classifier-fast-v1";
const CLASS_KEY = "recycle-class-names-fast-v1";

const MAX_IMAGES_PER_ZIP = 20;
const TRAIN_EPOCHS = 6;
const DENSE_UNITS = 64;

let mobilenetModel = null;
let classifierModel = null;
let classNames = [];
let uploadImage = null;
let cameraImage = null;
let cameraStream = null;

const $ = id => document.getElementById(id);

const els = {
  navItems: document.querySelectorAll(".nav-item"),
  views: document.querySelectorAll(".view"),

  statusChip: $("statusChip"),
  modelStatusText: $("modelStatusText"),
  statusDesc: $("statusDesc"),
  statusProgressBar: $("statusProgressBar"),
  statusProgressText: $("statusProgressText"),

  progressBar: $("progressBar"),
  progressText: $("progressText"),
  trainProgress: $("trainProgress"),
  trainLog: $("trainLog"),

  imageInput: $("imageInput"),
  previewCanvas: $("previewCanvas"),
  imageEmpty: $("imageEmpty"),
  predictBtn: $("predictBtn"),
  result: $("result"),

  cameraVideo: $("cameraVideo"),
  cameraCanvas: $("cameraCanvas"),
  cameraEmpty: $("cameraEmpty"),
  startCameraBtn: $("startCameraBtn"),
  captureBtn: $("captureBtn"),
  cameraPredictBtn: $("cameraPredictBtn"),
  cameraResult: $("cameraResult"),

  trainManifestBtn: $("trainManifestBtn"),
  clearModelBtn: $("clearModelBtn"),
  zipInput: $("zipInput"),
  trainSelectedBtn: $("trainSelectedBtn")
};

function log(message) {
  if (!els.trainLog) return;

  if (els.trainLog.textContent === "학습 로그가 여기에 표시됩니다.") {
    els.trainLog.textContent = "";
  }

  els.trainLog.textContent += `${message}\n`;
  els.trainLog.scrollTop = els.trainLog.scrollHeight;
}

function setProgress(value) {
  const safe = Math.max(0, Math.min(100, Math.round(value)));

  if (els.progressBar) {
    els.progressBar.style.width = `${safe}%`;
  }

  if (els.progressText) {
    els.progressText.textContent = `${safe}%`;
  }

  if (els.trainProgress) {
    els.trainProgress.value = safe;
  }

  if (els.statusProgressBar) {
    els.statusProgressBar.style.width = `${safe}%`;
  }

  if (els.statusProgressText) {
    els.statusProgressText.textContent = `${safe}%`;
  }
}

function setStatus(text, desc = "", type = "loading") {
  if (els.modelStatusText) {
    els.modelStatusText.textContent = text;
  }

  if (els.statusDesc) {
    els.statusDesc.textContent = desc;
  }

  if (els.statusChip) {
    els.statusChip.className = `status-chip ${type}`;

    if (type === "ready") {
      els.statusChip.textContent = "준비 완료";
    } else if (type === "error") {
      els.statusChip.textContent = "오류";
    } else {
      els.statusChip.textContent = "진행 중";
    }
  }
}

function normalizeName(text) {
  return String(text || "")
    .normalize("NFC")
    .toLowerCase()
    .replace(/[()]/g, "")
    .replace(/[＿_\-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function detectClass(text, mapping, fallback) {
  const normalized = normalizeName(text);

  for (const [className, keywords] of Object.entries(mapping)) {
    for (const keyword of keywords) {
      if (normalized.includes(normalizeName(keyword))) {
        return className;
      }
    }
  }

  return fallback;
}

function isImageFile(name) {
  const lower = String(name || "").toLowerCase();
  return IMAGE_EXTS.some(ext => lower.endsWith(ext));
}

function encodePath(path) {
  return String(path)
    .split("/")
    .map(part => encodeURIComponent(part))
    .join("/");
}

function makeZipNameVariants(name) {
  const variants = new Set();

  const clean = String(name || "").trim();
  if (!clean) return [];

  variants.add(clean);

  if (clean.includes("오염0")) {
    variants.add(clean.replaceAll("오염0", "오염o"));
    variants.add(clean.replaceAll("오염0", "오염O"));
  }

  if (clean.includes("오염o")) {
    variants.add(clean.replaceAll("오염o", "오염0"));
    variants.add(clean.replaceAll("오염o", "오염O"));
  }

  if (clean.includes("오염O")) {
    variants.add(clean.replaceAll("오염O", "오염0"));
    variants.add(clean.replaceAll("오염O", "오염o"));
  }

  if (clean.includes("오염 0")) {
    variants.add(clean.replaceAll("오염 0", "오염 o"));
    variants.add(clean.replaceAll("오염 0", "오염 O"));
  }

  if (clean.includes("오염 o")) {
    variants.add(clean.replaceAll("오염 o", "오염 0"));
    variants.add(clean.replaceAll("오염 o", "오염 O"));
  }

  if (clean.includes("오염 O")) {
    variants.add(clean.replaceAll("오염 O", "오염 0"));
    variants.add(clean.replaceAll("오염 O", "오염 o"));
  }

  return [...variants];
}

async function fetchFirstAvailable(paths) {
  let lastError = null;

  for (const path of paths) {
    try {
      const res = await fetch(encodePath(path), { cache: "no-store" });

      if (res.ok) {
        return { res, path };
      }

      lastError = new Error(`${path} 불러오기 실패: ${res.status}`);
    } catch (err) {
      lastError = err;
    }
  }

  throw lastError || new Error("파일을 불러올 수 없습니다.");
}

async function getZipNames() {
  try {
    const manifestResult = await fetchFirstAvailable([
      "zips/manifest.json",
      "manifest.json"
    ]);

    const manifest = await manifestResult.res.json();

    if (Array.isArray(manifest.files) && manifest.files.length) {
      log(`manifest 사용: ${manifestResult.path}`);
      return manifest.files;
    }
  } catch (err) {
    log("manifest.json 없음. 기본 ZIP 목록으로 학습합니다.");
  }

  return DEFAULT_ZIP_FILES;
}

async function loadZipFilesFromRepo() {
  const zipNames = await getZipNames();
  const files = [];
  const skipped = [];

  for (let i = 0; i < zipNames.length; i++) {
    const rawName = zipNames[i];
    const name = String(rawName).trim();

    if (!name) continue;

    const cleanName = name.replace(/^zips\//, "");
    const nameVariants = makeZipNameVariants(cleanName);

    const candidatePaths = [];

    for (const variant of nameVariants) {
      candidatePaths.push(`zips/${variant}`);
      candidatePaths.push(variant);
    }

    candidatePaths.push(name);

    setProgress(8 + Math.round((i / zipNames.length) * 12));
    setStatus("ZIP 데이터 확인 중...", `${cleanName} 파일을 확인하고 있습니다.`, "loading");
    log(`ZIP 확인 중: ${cleanName}`);

    try {
      const result = await fetchFirstAvailable(candidatePaths);
      const blob = await result.res.blob();

      files.push(new File([blob], cleanName, { type: "application/zip" }));

      log(`  불러옴: ${result.path}`);
    } catch (err) {
      skipped.push(cleanName);
      log(`  건너뜀: ${cleanName} 파일을 찾을 수 없습니다.`);
    }

    await tf.nextFrame();
  }

  if (skipped.length > 0) {
    log("");
    log("누락되어 건너뛴 ZIP 파일:");
    skipped.forEach(name => log(`  - ${name}`));
    log("");
  }

  if (!files.length) {
    throw new Error("불러올 수 있는 ZIP 파일이 없습니다. 저장소에 학습용 ZIP 파일이 있는지 확인하세요.");
  }

  log(`사용 가능한 ZIP ${files.length}개로 학습을 진행합니다.`);

  return files;
}

async function loadBaseModel() {
  if (!mobilenetModel) {
    setProgress(20);
    setStatus("AI 엔진 준비 중...", "처음 한 번만 시간이 걸립니다.", "loading");
    log("MobileNet 로딩 중...");

    mobilenetModel = await mobilenet.load({
      version: 2,
      alpha: 0.5
    });

    log("MobileNet 로딩 완료");
    setProgress(28);
  }
}

async function fileToImage(fileOrBlob) {
  const url = URL.createObjectURL(fileOrBlob);

  return new Promise((resolve, reject) => {
    const img = new Image();

    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };

    img.onerror = err => {
      URL.revokeObjectURL(url);
      reject(err);
    };

    img.src = url;
  });
}

function getEmbedding(image) {
  return tf.tidy(() => {
    const tensor = tf.browser
      .fromPixels(image)
      .resizeBilinear([224, 224])
      .toFloat()
      .div(255)
      .expandDims(0);

    return mobilenetModel.infer(tensor, true).squeeze();
  });
}

async function readZipFile(file, sourceName) {
  const zip = await JSZip.loadAsync(file);

  const materialClass = detectClass(sourceName, MATERIAL_KEYWORDS, "unknown");
  const contaminationClass = detectClass(sourceName, CONTAMINATION_KEYWORDS, "uncertain");

  let entries = Object.values(zip.files).filter(entry => {
    return !entry.dir && isImageFile(entry.name);
  });

  if (entries.length > MAX_IMAGES_PER_ZIP) {
    const sampled = [];
    const step = entries.length / MAX_IMAGES_PER_ZIP;

    for (let i = 0; i < MAX_IMAGES_PER_ZIP; i++) {
      sampled.push(entries[Math.floor(i * step)]);
    }

    entries = sampled;
  }

  const samples = [];

  for (const entry of entries) {
    const blob = await entry.async("blob");

    samples.push({
      blob,
      materialClass,
      contaminationClass,
      source: sourceName,
      entry: entry.name
    });
  }

  return samples;
}

function makeClassName(materialClass, contaminationClass) {
  return `${materialClass}__${contaminationClass}`;
}

function parseClassName(className) {
  const [materialClass, contaminationClass] = String(className).split("__");

  return {
    materialClass: materialClass || "unknown",
    contaminationClass: contaminationClass || "uncertain"
  };
}

async function buildTrainingData(zipFiles) {
  const samples = [];

  for (let i = 0; i < zipFiles.length; i++) {
    const file = zipFiles[i];

    setProgress(30 + Math.round((i / zipFiles.length) * 15));
    setStatus("ZIP 압축 해제 중...", `${file.name} 이미지를 읽고 있습니다.`, "loading");

    const zipSamples = await readZipFile(file, file.name);

    const materialClass = detectClass(file.name, MATERIAL_KEYWORDS, "unknown");
    const contaminationClass = detectClass(file.name, CONTAMINATION_KEYWORDS, "uncertain");

    log(`${file.name}`);
    log(`  이미지: ${zipSamples.length}장`);
    log(`  품목: ${MATERIAL_LABELS_KOR[materialClass] || materialClass}`);
    log(`  오염도: ${CONTAMINATION_LABELS_KOR[contaminationClass] || contaminationClass}`);

    samples.push(...zipSamples);
    await tf.nextFrame();
  }

  return samples.filter(sample => {
    return sample.materialClass !== "unknown" && sample.contaminationClass !== "uncertain";
  });
}

async function trainFromZipFiles(zipFiles) {
  setProgress(0);

  if (els.trainLog) {
    els.trainLog.textContent = "";
  }

  setStatus("학습 준비 중...", "데이터와 AI 엔진을 준비하고 있습니다.", "loading");

  await loadBaseModel();

  const samples = await buildTrainingData(zipFiles);

  if (samples.length < 4) {
    throw new Error("학습 가능한 이미지가 너무 적습니다.");
  }

  const uniqueClasses = [
    ...new Set(samples.map(sample => makeClassName(sample.materialClass, sample.contaminationClass)))
  ].sort();

  if (uniqueClasses.length < 2) {
    throw new Error("최소 2개 이상의 클래스가 필요합니다.");
  }

  classNames = uniqueClasses;
  log(`학습 클래스: ${classNames.join(", ")}`);

  const xs = [];
  const ys = [];

  setStatus("이미지 특징 추출 중...", `총 ${samples.length}장의 이미지를 분석하고 있습니다.`, "loading");

  for (let i = 0; i < samples.length; i++) {
    const img = await fileToImage(samples[i].blob);
    const embedding = getEmbedding(img);

    xs.push(await embedding.array());
    embedding.dispose();

    const classIndex = classNames.indexOf(
      makeClassName(samples[i].materialClass, samples[i].contaminationClass)
    );

    ys.push(classIndex);

    if (i % 3 === 0) {
      setProgress(45 + Math.round((i / samples.length) * 30));
      setStatus("이미지 특징 추출 중...", `${i + 1} / ${samples.length}장 처리 중입니다.`, "loading");
      await tf.nextFrame();
    }
  }

  const xTensor = tf.tensor2d(xs);
  const yTensor = tf.oneHot(tf.tensor1d(ys, "int32"), classNames.length);

  classifierModel = tf.sequential();

  classifierModel.add(tf.layers.dense({
    inputShape: [xTensor.shape[1]],
    units: DENSE_UNITS,
    activation: "relu"
  }));

  classifierModel.add(tf.layers.dropout({
    rate: 0.2
  }));

  classifierModel.add(tf.layers.dense({
    units: classNames.length,
    activation: "softmax"
  }));

  classifierModel.compile({
    optimizer: tf.train.adam(0.001),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });

  setStatus("모델 학습 중...", "분류 모델을 빠르게 학습하고 있습니다.", "loading");
  log("모델 학습 시작");

  await classifierModel.fit(xTensor, yTensor, {
    epochs: TRAIN_EPOCHS,
    batchSize: Math.min(16, samples.length),
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        const acc = logs.acc ?? logs.accuracy ?? 0;

        log(`epoch ${epoch + 1}/${TRAIN_EPOCHS} - loss ${logs.loss.toFixed(4)} - acc ${(acc * 100).toFixed(1)}%`);

        setProgress(76 + Math.round(((epoch + 1) / TRAIN_EPOCHS) * 18));
        setStatus("모델 학습 중...", `${epoch + 1} / ${TRAIN_EPOCHS}회 학습 중입니다.`, "loading");

        await tf.nextFrame();
      }
    }
  });

  xTensor.dispose();
  yTensor.dispose();

  setProgress(96);
  setStatus("모델 저장 중...", "학습 결과를 브라우저에 저장하고 있습니다.", "loading");

  await classifierModel.save(MODEL_KEY);
  localStorage.setItem(CLASS_KEY, JSON.stringify(classNames));

  setProgress(100);
  setStatus("분석 준비 완료", "이미지 업로드 또는 카메라 분석을 사용할 수 있습니다.", "ready");
  log("학습 완료. 모델이 브라우저에 저장되었습니다.");
}

async function loadSavedModelFast() {
  try {
    setProgress(3);
    setStatus("모델 확인 중...", "저장된 모델이 있는지 확인합니다.", "loading");

    classNames = JSON.parse(localStorage.getItem(CLASS_KEY) || "[]");

    if (!classNames.length) {
      return false;
    }

    classifierModel = await tf.loadLayersModel(MODEL_KEY + "/model.json");

    setProgress(100);
    setStatus("분석 준비 완료", "저장된 모델을 불러왔습니다.", "ready");
    log("저장된 모델을 불러왔습니다.");

    return true;
  } catch (err) {
    return false;
  }
}

async function autoTrain() {
  try {
    setProgress(6);
    setStatus("자동 학습 시작", "사용 가능한 ZIP 데이터를 불러오는 중입니다.", "loading");

    const files = await loadZipFilesFromRepo();

    if (!files.length) {
      throw new Error("불러올 ZIP 파일이 없습니다.");
    }

    await trainFromZipFiles(files);
  } catch (err) {
    setStatus("자동 학습 실패", err.message, "error");
    log(`오류: ${err.message}`);
    alert(`자동 학습에 실패했습니다.\n\n${err.message}`);
  }
}

function drawImageToCanvas(img, canvas, emptyEl) {
  const maxWidth = 900;
  const scale = Math.min(1, maxWidth / img.width);

  canvas.width = Math.round(img.width * scale);
  canvas.height = Math.round(img.height * scale);

  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  canvas.style.display = "block";

  if (emptyEl) {
    emptyEl.style.display = "none";
  }
}

async function predictImage(imageElement, resultEl) {
  if (!classifierModel || !classNames.length) {
    alert("아직 모델이 준비되지 않았습니다. 모델 상태를 확인해주세요.");
    return;
  }

  if (!imageElement) {
    alert("분석할 이미지를 먼저 선택하거나 촬영하세요.");
    return;
  }

  await loadBaseModel();

  const embedding = getEmbedding(imageElement);
  const pred = classifierModel.predict(embedding.expandDims(0));
  const probs = await pred.data();

  embedding.dispose();
  pred.dispose();

  let topIndex = 0;

  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[topIndex]) {
      topIndex = i;
    }
  }

  const confidence = probs[topIndex];
  const { materialClass, contaminationClass } = parseClassName(classNames[topIndex]);

  const materialKor = MATERIAL_LABELS_KOR[materialClass] || materialClass;
  const contaminationKor = CONTAMINATION_LABELS_KOR[contaminationClass] || contaminationClass;

  const disposal =
    contaminationClass === "dirty"
      ? `세척 필요 / ${materialKor}류`
      : `정상 배출 / ${materialKor}류`;

  const advice =
    contaminationClass === "dirty"
      ? `⚠️ ${materialKor}에 오염이 감지되었습니다. 내용물을 비우고 물로 헹군 뒤 배출하세요.`
      : `✅ 깨끗한 ${materialKor}로 판단됩니다. 알맞은 수거함에 분리배출하세요.`;

  resultEl.innerHTML = `
    <div class="result-summary">
      <div class="result-metric">
        <span>재질</span>
        <strong>${materialKor}</strong>
        <em>${(confidence * 100).toFixed(1)}%</em>
      </div>

      <div class="result-metric">
        <span>오염도</span>
        <strong>${contaminationKor}</strong>
        <em>AI 분석 기준</em>
      </div>
    </div>

    <table class="result-table">
      <thead>
        <tr>
          <th>번호</th>
          <th>탐지명</th>
          <th>재질</th>
          <th>신뢰도</th>
          <th>오염도</th>
          <th>배출 방법</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>1</td>
          <td>${materialClass}</td>
          <td>${materialKor}</td>
          <td>${confidence.toFixed(3)}</td>
          <td>${contaminationKor}</td>
          <td>${disposal}</td>
        </tr>
      </tbody>
    </table>

    <div class="advice-box ${contaminationClass === "dirty" ? "dirty" : ""}">
      ${advice}
    </div>
  `;
}

els.navItems.forEach(btn => {
  btn.addEventListener("click", () => {
    els.navItems.forEach(item => item.classList.remove("active"));
    els.views.forEach(view => view.classList.remove("active"));

    btn.classList.add("active");

    const target = $(btn.dataset.view);

    if (target) {
      target.classList.add("active");
    }
  });
});

if (els.imageInput) {
  els.imageInput.addEventListener("change", async event => {
    const file = event.target.files[0];

    if (!file) return;

    uploadImage = await fileToImage(file);
    drawImageToCanvas(uploadImage, els.previewCanvas, els.imageEmpty);
  });
}

if (els.predictBtn) {
  els.predictBtn.addEventListener("click", () => {
    predictImage(uploadImage, els.result);
  });
}

if (els.startCameraBtn) {
  els.startCameraBtn.addEventListener("click", async () => {
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false
      });

      els.cameraVideo.srcObject = cameraStream;
    } catch (err) {
      alert("카메라를 켤 수 없습니다. 브라우저 권한을 확인하세요.");
    }
  });
}

if (els.captureBtn) {
  els.captureBtn.addEventListener("click", () => {
    const video = els.cameraVideo;

    if (!video.videoWidth) {
      alert("카메라를 먼저 켜세요.");
      return;
    }

    const canvas = els.cameraCanvas;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    canvas.style.display = "block";

    if (els.cameraEmpty) {
      els.cameraEmpty.style.display = "none";
    }

    const img = new Image();

    img.onload = () => {
      cameraImage = img;
    };

    img.src = canvas.toDataURL("image/jpeg");
  });
}

if (els.cameraPredictBtn) {
  els.cameraPredictBtn.addEventListener("click", () => {
    predictImage(cameraImage, els.cameraResult);
  });
}

if (els.trainManifestBtn) {
  els.trainManifestBtn.addEventListener("click", async () => {
    await autoTrain();
  });
}

if (els.clearModelBtn) {
  els.clearModelBtn.addEventListener("click", async () => {
    try {
      await tf.io.removeModel(MODEL_KEY);
    } catch (_) {}

    localStorage.removeItem(CLASS_KEY);

    classifierModel = null;
    classNames = [];

    setProgress(0);
    setStatus("모델 초기화 완료", "다시 학습 버튼을 누르면 새로 학습합니다.", "loading");
    log("저장 모델을 초기화했습니다.");
  });
}

function timeout(ms) {
  return new Promise(resolve => {
    setTimeout(() => resolve(false), ms);
  });
}

async function startApp() {
  setProgress(3);
  setStatus("모델 확인 중...", "저장된 모델이 있는지 빠르게 확인합니다.", "loading");

  const hasModel = await Promise.race([
    loadSavedModelFast(),
    timeout(3500)
  ]);

  if (hasModel) {
    setProgress(100);
    setStatus("분석 준비 완료", "저장된 모델을 불러왔습니다.", "ready");
    return;
  }

  setProgress(8);
  setStatus("자동 학습 시작", "저장된 모델이 없어 빠른 학습을 시작합니다.", "loading");

  await autoTrain();
}

startApp();
