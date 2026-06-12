const APP_VERSION = "v24-object-detect";

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

/*
  v24 변경점
  - 기존 저장 모델과 충돌하지 않도록 키 변경
  - 학습 데이터 수와 epoch를 늘려 정확도 회복
  - 중앙 박스만 보는 방식 대신, 배경과 다른 실제 객체 후보 영역을 먼저 찾음
*/
const MODEL_KEY = "indexeddb://recycle-classifier-fast-v24-object-detect";
const CLASS_KEY = "recycle-class-names-fast-v24-object-detect";

const MAX_IMAGES_PER_ZIP = 80;
const MAX_IMAGES_PER_CLASS = 60;
const TRAIN_EPOCHS = 18;
const DENSE_UNITS = 192;

const DETECTION_CANVAS_SIZE = 192;
const MIN_OBJECT_AREA_RATIO = 0.015;
const MAX_OBJECT_AREA_RATIO = 0.88;
const MIN_BOX_SIDE_RATIO = 0.08;
const MAX_CANDIDATE_BOXES = 9;

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

function injectRuntimeStyles() {
  if (document.getElementById("runtimeDetectionStyles")) return;

  const style = document.createElement("style");
  style.id = "runtimeDetectionStyles";
  style.textContent = `
    .detected-canvas-wrap{
      margin: 0 0 18px 0;
      border-radius: 18px;
      overflow: hidden;
      background: rgba(8,12,20,.75);
      border: 1px solid rgba(255,255,255,.08);
      box-shadow: 0 10px 30px rgba(0,0,0,.22);
    }
    .detected-canvas{
      display:block;
      width:100%;
      height:auto;
      background:#09111c;
    }
    .result-summary{
      display:grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }
    .result-metric{
      background: rgba(255,255,255,.04);
      border: 1px solid rgba(255,255,255,.08);
      border-radius: 16px;
      padding: 16px;
    }
    .result-metric span{
      display:block;
      font-size: 13px;
      opacity:.75;
      margin-bottom:8px;
    }
    .result-metric strong{
      display:block;
      font-size: 22px;
      line-height:1.25;
      margin-bottom:6px;
    }
    .result-metric em{
      font-style:normal;
      font-size: 13px;
      opacity:.7;
    }
    .result-table{
      width:100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 14px;
      background: rgba(255,255,255,.03);
      border: 1px solid rgba(255,255,255,.08);
      margin-bottom: 16px;
    }
    .result-table th,
    .result-table td{
      padding: 12px 14px;
      border-bottom: 1px solid rgba(255,255,255,.06);
      text-align: left;
      font-size: 14px;
    }
    .result-table th{
      opacity:.8;
      font-weight:600;
      background: rgba(255,255,255,.03);
    }
    .advice-box{
      padding: 14px 16px;
      border-radius: 14px;
      background: rgba(95, 207, 128, .12);
      border: 1px solid rgba(95, 207, 128, .25);
      line-height:1.6;
      font-size:14px;
    }
    .advice-box.dirty{
      background: rgba(255, 180, 72, .12);
      border: 1px solid rgba(255, 180, 72, .25);
    }
    .result-caption{
      font-size: 13px;
      opacity: .75;
      margin: 0 0 12px;
    }
    .empty-note{
      padding: 18px;
      border-radius: 16px;
      background: rgba(255,255,255,.04);
      border: 1px solid rgba(255,255,255,.08);
      opacity: .85;
      line-height: 1.6;
    }
  `;
  document.head.appendChild(style);
}

function applyVersionLabel() {
  document.title = `스마트 재활용 분류 시스템 ${APP_VERSION}`;

  const brandText = document.querySelector(".brand p");
  if (brandText) {
    brandText.textContent = `스마트 재활용 분류 · ${APP_VERSION}`;
  }

  const eyebrow = document.querySelector(".eyebrow");
  if (eyebrow) {
    eyebrow.textContent = `AI WASTE SORTING · ${APP_VERSION}`;
  }
}

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

  if (els.progressBar) els.progressBar.style.width = `${safe}%`;
  if (els.progressText) els.progressText.textContent = `${safe}%`;
  if (els.trainProgress) els.trainProgress.value = safe;
  if (els.statusProgressBar) els.statusProgressBar.style.width = `${safe}%`;
  if (els.statusProgressText) els.statusProgressText.textContent = `${safe}%`;
}

function setStatus(text, desc = "", type = "loading") {
  if (els.modelStatusText) els.modelStatusText.textContent = text;
  if (els.statusDesc) els.statusDesc.textContent = desc;

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

function looksLikeZip(arrayBuffer) {
  if (!arrayBuffer || arrayBuffer.byteLength < 4) return false;

  const bytes = new Uint8Array(arrayBuffer.slice(0, 4));

  return (
    bytes[0] === 0x50 &&
    bytes[1] === 0x4b &&
    (
      (bytes[2] === 0x03 && bytes[3] === 0x04) ||
      (bytes[2] === 0x05 && bytes[3] === 0x06) ||
      (bytes[2] === 0x07 && bytes[3] === 0x08)
    )
  );
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
      const buffer = await result.res.arrayBuffer();

      if (!looksLikeZip(buffer)) {
        skipped.push(cleanName);
        log(`  건너뜀: ${result.path} 파일이 정상 ZIP 형식이 아닙니다.`);
        await tf.nextFrame();
        continue;
      }

      const blob = new Blob([buffer], { type: "application/zip" });
      files.push(new File([blob], cleanName, { type: "application/zip" }));
      log(`  불러옴: ${result.path}`);
    } catch (err) {
      skipped.push(cleanName);
      log(`  건너뜀: ${cleanName} 파일을 찾을 수 없거나 불러올 수 없습니다.`);
    }

    await tf.nextFrame();
  }

  if (skipped.length > 0) {
    log("");
    log("누락 또는 손상으로 건너뛴 ZIP 파일:");
    skipped.forEach(name => log(`  - ${name}`));
    log("");
  }

  if (!files.length) {
    throw new Error("불러올 수 있는 정상 ZIP 파일이 없습니다. 저장소에 학습용 ZIP 파일이 있는지 확인하세요.");
  }

  log(`사용 가능한 정상 ZIP ${files.length}개로 학습을 진행합니다.`);
  return files;
}

async function prepareBackend() {
  try {
    await tf.ready();

    if (tf.getBackend() !== "webgl" && tf.findBackend && tf.findBackend("webgl")) {
      await tf.setBackend("webgl");
      await tf.ready();
    }
  } catch (err) {
    console.warn("TF backend 설정 실패:", err);
  }
}

async function loadBaseModel() {
  if (!mobilenetModel) {
    setProgress(20);
    setStatus("AI 엔진 준비 중...", "처음 한 번만 시간이 걸립니다.", "loading");
    log("MobileNet 로딩 중...");

    mobilenetModel = await mobilenet.load({
      version: 2,
      alpha: 0.75
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

function getEmbedding(imageSource) {
  return tf.tidy(() => {
    const tensor = tf.browser
      .fromPixels(imageSource)
      .resizeBilinear([224, 224])
      .toFloat()
      .div(255)
      .expandDims(0);

    return mobilenetModel.infer(tensor, true).squeeze();
  });
}

async function readZipFile(file, sourceName) {
  let zip;

  try {
    zip = await JSZip.loadAsync(file);
  } catch (err) {
    log(`  건너뜀: ${sourceName} 압축을 해제할 수 없습니다.`);
    return [];
  }

  const materialClass = detectClass(sourceName, MATERIAL_KEYWORDS, "unknown");
  const contaminationClass = detectClass(sourceName, CONTAMINATION_KEYWORDS, "uncertain");

  let entries = Object.values(zip.files).filter(entry => !entry.dir && isImageFile(entry.name));

  if (!entries.length) {
    log(`  건너뜀: ${sourceName} 안에 이미지가 없습니다.`);
    return [];
  }

  if (entries.length > MAX_IMAGES_PER_ZIP) {
    const sampled = [];
    const shuffled = [...entries].sort(() => Math.random() - 0.5);

    for (let i = 0; i < MAX_IMAGES_PER_ZIP; i++) {
      sampled.push(shuffled[i % shuffled.length]);
    }

    entries = sampled;
  }

  const samples = [];

  for (const entry of entries) {
    try {
      const blob = await entry.async("blob");

      samples.push({
        blob,
        materialClass,
        contaminationClass,
        source: sourceName,
        entry: entry.name
      });
    } catch (err) {
      log(`  이미지 건너뜀: ${entry.name}`);
    }
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

function balanceSamples(samples) {
  const groups = new Map();

  for (const sample of samples) {
    const key = makeClassName(sample.materialClass, sample.contaminationClass);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(sample);
  }

  const balanced = [];

  for (const [key, arr] of groups.entries()) {
    const shuffled = [...arr].sort(() => Math.random() - 0.5);
    const picked = shuffled.slice(0, MAX_IMAGES_PER_CLASS);

    balanced.push(...picked);
    log(`샘플 사용: ${key} → ${picked.length}장 사용`);
  }

  return balanced.sort(() => Math.random() - 0.5);
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

  const filtered = samples.filter(sample => {
    return sample.materialClass !== "unknown" && sample.contaminationClass !== "uncertain";
  });

  return balanceSamples(filtered);
}

async function trainFromZipFiles(zipFiles) {
  setProgress(0);

  if (els.trainLog) {
    els.trainLog.textContent = "";
  }

  setStatus("학습 준비 중...", "데이터와 AI 엔진을 준비하고 있습니다.", "loading");

  await prepareBackend();
  await loadBaseModel();

  const samples = await buildTrainingData(zipFiles);

  if (samples.length < 8) {
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
    try {
      const img = await fileToImage(samples[i].blob);
      const embedding = getEmbedding(img);

      xs.push(await embedding.array());
      embedding.dispose();

      const classIndex = classNames.indexOf(
        makeClassName(samples[i].materialClass, samples[i].contaminationClass)
      );

      ys.push(classIndex);
    } catch (err) {
      log(`  이미지 특징 추출 실패. 1장 건너뜀.`);
    }

    if (i % 3 === 0) {
      setProgress(45 + Math.round((i / samples.length) * 30));
      setStatus("이미지 특징 추출 중...", `${i + 1} / ${samples.length}장 처리 중입니다.`, "loading");
      await tf.nextFrame();
    }
  }

  if (xs.length < 8 || ys.length < 8) {
    throw new Error("학습에 사용할 수 있는 정상 이미지가 너무 적습니다.");
  }

  const xTensor = tf.tensor2d(xs);
  const yTensor = tf.oneHot(tf.tensor1d(ys, "int32"), classNames.length);

  classifierModel = tf.sequential();

  classifierModel.add(tf.layers.dense({
    inputShape: [xTensor.shape[1]],
    units: DENSE_UNITS,
    activation: "relu",
    kernelRegularizer: tf.regularizers.l2({ l2: 0.0008 })
  }));

  classifierModel.add(tf.layers.dropout({
    rate: 0.28
  }));

  classifierModel.add(tf.layers.dense({
    units: Math.max(48, Math.round(DENSE_UNITS / 2)),
    activation: "relu",
    kernelRegularizer: tf.regularizers.l2({ l2: 0.0008 })
  }));

  classifierModel.add(tf.layers.dropout({
    rate: 0.18
  }));

  classifierModel.add(tf.layers.dense({
    units: classNames.length,
    activation: "softmax"
  }));

  classifierModel.compile({
    optimizer: tf.train.adam(0.0008),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });

  setStatus("모델 학습 중...", "분류 모델을 학습하고 있습니다.", "loading");
  log("모델 학습 시작");

  await classifierModel.fit(xTensor, yTensor, {
    epochs: TRAIN_EPOCHS,
    batchSize: Math.min(16, xs.length),
    shuffle: true,
    validationSplit: xs.length >= 30 ? 0.15 : 0,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        const acc = logs.acc ?? logs.accuracy ?? 0;
        const valAcc = logs.val_acc ?? logs.val_accuracy;

        let line = `epoch ${epoch + 1}/${TRAIN_EPOCHS} - loss ${logs.loss.toFixed(4)} - acc ${(acc * 100).toFixed(1)}%`;

        if (typeof valAcc === "number") {
          line += ` - val_acc ${(valAcc * 100).toFixed(1)}%`;
        }

        log(line);

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
  setStatus("분석 준비 완료", `분석 모델 준비가 완료되었습니다. ${APP_VERSION}`, "ready");
  log("학습 완료. 모델이 브라우저에 저장되었습니다.");
}

async function loadSavedModelFast() {
  try {
    setProgress(3);
    setStatus("모델 확인 중...", "저장된 모델이 있는지 확인합니다.", "loading");

    classNames = JSON.parse(localStorage.getItem(CLASS_KEY) || "[]");
    if (!classNames.length) return false;

    classifierModel = await tf.loadLayersModel(MODEL_KEY + "/model.json");

    setProgress(100);
    setStatus("분석 준비 완료", `저장된 모델을 불러왔습니다. ${APP_VERSION}`, "ready");
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
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  canvas.style.display = "block";

  if (emptyEl) {
    emptyEl.style.display = "none";
  }
}

function getSourceSize(source) {
  return {
    width: source.naturalWidth || source.videoWidth || source.width,
    height: source.naturalHeight || source.videoHeight || source.height
  };
}

function makeCropCanvas(source, box) {
  const c = document.createElement("canvas");
  c.width = Math.max(8, Math.round(box.w));
  c.height = Math.max(8, Math.round(box.h));

  const ctx = c.getContext("2d");
  ctx.drawImage(
    source,
    box.x, box.y, box.w, box.h,
    0, 0, c.width, c.height
  );

  return c;
}

function clampBox(box, width, height) {
  const x = Math.max(0, Math.min(width - 1, Math.round(box.x)));
  const y = Math.max(0, Math.min(height - 1, Math.round(box.y)));
  const w = Math.max(24, Math.min(width - x, Math.round(box.w)));
  const h = Math.max(24, Math.min(height - y, Math.round(box.h)));

  return { x, y, w, h };
}

function expandBox(box, width, height, padRatio = 0.08) {
  const padX = box.w * padRatio;
  const padY = box.h * padRatio;

  return clampBox({
    x: box.x - padX,
    y: box.y - padY,
    w: box.w + padX * 2,
    h: box.h + padY * 2
  }, width, height);
}

function getImageDataForDetection(source) {
  const { width, height } = getSourceSize(source);

  const scale = Math.min(
    1,
    DETECTION_CANVAS_SIZE / Math.max(width, height)
  );

  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(width * scale));
  canvas.height = Math.max(1, Math.round(height * scale));

  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);

  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  return {
    canvas,
    data: imageData.data,
    smallWidth: canvas.width,
    smallHeight: canvas.height,
    scaleX: width / canvas.width,
    scaleY: height / canvas.height,
    sourceWidth: width,
    sourceHeight: height
  };
}

function getPixel(data, width, x, y) {
  const idx = (y * width + x) * 4;

  return {
    r: data[idx],
    g: data[idx + 1],
    b: data[idx + 2],
    a: data[idx + 3]
  };
}

function luminance(pixel) {
  return pixel.r * 0.299 + pixel.g * 0.587 + pixel.b * 0.114;
}

function colorDistance(a, b) {
  const dr = a.r - b.r;
  const dg = a.g - b.g;
  const db = a.b - b.b;

  return Math.sqrt(dr * dr + dg * dg + db * db);
}

function median(values) {
  if (!values.length) return 0;

  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);

  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }

  return sorted[mid];
}

function estimateBackgroundColor(data, width, height) {
  const borderPixels = [];
  const step = Math.max(1, Math.floor(Math.min(width, height) / 32));

  for (let x = 0; x < width; x += step) {
    borderPixels.push(getPixel(data, width, x, 0));
    borderPixels.push(getPixel(data, width, x, height - 1));
  }

  for (let y = 0; y < height; y += step) {
    borderPixels.push(getPixel(data, width, 0, y));
    borderPixels.push(getPixel(data, width, width - 1, y));
  }

  return {
    r: median(borderPixels.map(p => p.r)),
    g: median(borderPixels.map(p => p.g)),
    b: median(borderPixels.map(p => p.b)),
    a: 255
  };
}

function buildObjectMask(info) {
  const {
    data,
    smallWidth: width,
    smallHeight: height
  } = info;

  const bg = estimateBackgroundColor(data, width, height);
  const bgLum = luminance(bg);
  const diffValues = [];

  for (let y = 0; y < height; y += 3) {
    for (let x = 0; x < width; x += 3) {
      const p = getPixel(data, width, x, y);
      const diff = colorDistance(p, bg);
      const lumDiff = Math.abs(luminance(p) - bgLum);
      diffValues.push(diff * 0.75 + lumDiff * 0.25);
    }
  }

  const baseThreshold = Math.max(24, median(diffValues) * 1.25);
  const mask = new Uint8Array(width * height);

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const p = getPixel(data, width, x, y);
      const currentLum = luminance(p);

      const right = getPixel(data, width, x + 1, y);
      const down = getPixel(data, width, x, y + 1);

      const bgDiff = colorDistance(p, bg);
      const lumDiff = Math.abs(currentLum - bgLum);
      const edge =
        Math.abs(currentLum - luminance(right)) +
        Math.abs(currentLum - luminance(down));

      const score = bgDiff * 0.72 + lumDiff * 0.22 + edge * 0.18;

      if (score > baseThreshold) {
        mask[y * width + x] = 1;
      }
    }
  }

  return smoothMask(mask, width, height);
}

function smoothMask(mask, width, height) {
  const opened = new Uint8Array(width * height);
  const closed = new Uint8Array(width * height);

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let count = 0;

      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          count += mask[(y + dy) * width + (x + dx)];
        }
      }

      if (count >= 3) {
        opened[y * width + x] = 1;
      }
    }
  }

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let count = 0;

      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          count += opened[(y + dy) * width + (x + dx)];
        }
      }

      if (count >= 2) {
        closed[y * width + x] = 1;
      }
    }
  }

  return closed;
}

function findConnectedComponents(mask, width, height) {
  const visited = new Uint8Array(width * height);
  const components = [];
  const queue = [];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const startIndex = y * width + x;

      if (!mask[startIndex] || visited[startIndex]) continue;

      let minX = x;
      let maxX = x;
      let minY = y;
      let maxY = y;
      let area = 0;

      queue.length = 0;
      queue.push(startIndex);
      visited[startIndex] = 1;

      while (queue.length) {
        const index = queue.shift();
        const cx = index % width;
        const cy = Math.floor(index / width);

        area++;
        if (cx < minX) minX = cx;
        if (cx > maxX) maxX = cx;
        if (cy < minY) minY = cy;
        if (cy > maxY) maxY = cy;

        const neighbors = [
          index - 1,
          index + 1,
          index - width,
          index + width
        ];

        for (const next of neighbors) {
          if (next < 0 || next >= mask.length) continue;

          const nx = next % width;
          const ny = Math.floor(next / width);

          if (Math.abs(nx - cx) + Math.abs(ny - cy) !== 1) continue;
          if (visited[next] || !mask[next]) continue;

          visited[next] = 1;
          queue.push(next);
        }
      }

      components.push({
        x: minX,
        y: minY,
        w: maxX - minX + 1,
        h: maxY - minY + 1,
        area
      });
    }
  }

  return components;
}

function boxIou(a, b) {
  const ax2 = a.x + a.w;
  const ay2 = a.y + a.h;
  const bx2 = b.x + b.w;
  const by2 = b.y + b.h;

  const ix1 = Math.max(a.x, b.x);
  const iy1 = Math.max(a.y, b.y);
  const ix2 = Math.min(ax2, bx2);
  const iy2 = Math.min(ay2, by2);

  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;

  const union = a.w * a.h + b.w * b.h - inter;

  return union <= 0 ? 0 : inter / union;
}

function nonMaxSuppression(boxes, limit = MAX_CANDIDATE_BOXES) {
  const sorted = [...boxes].sort((a, b) => b.objectness - a.objectness);
  const picked = [];

  for (const box of sorted) {
    const duplicate = picked.some(prev => boxIou(prev, box) > 0.55);

    if (!duplicate) {
      picked.push(box);
    }

    if (picked.length >= limit) break;
  }

  return picked;
}

function makeFallbackCenterBoxes(width, height) {
  const boxes = [];

  const add = box => {
    const b = clampBox(box, width, height);

    const duplicated = boxes.some(prev => {
      const dx = Math.abs(prev.x - b.x);
      const dy = Math.abs(prev.y - b.y);
      const dw = Math.abs(prev.w - b.w);
      const dh = Math.abs(prev.h - b.h);
      return dx + dy + dw + dh < 35;
    });

    if (!duplicated) {
      boxes.push({
        ...b,
        objectness: 0.35,
        source: "fallback"
      });
    }
  };

  add({
    x: width * 0.20,
    y: height * 0.12,
    w: width * 0.60,
    h: height * 0.76
  });

  add({
    x: width * 0.14,
    y: height * 0.08,
    w: width * 0.72,
    h: height * 0.84
  });

  add({
    x: width * 0.26,
    y: height * 0.18,
    w: width * 0.48,
    h: height * 0.64
  });

  add({
    x: width * 0.08,
    y: height * 0.08,
    w: width * 0.84,
    h: height * 0.84
  });

  return boxes;
}

function generateCandidateBoxes(source) {
  const info = getImageDataForDetection(source);
  const {
    smallWidth,
    smallHeight,
    scaleX,
    scaleY,
    sourceWidth,
    sourceHeight
  } = info;

  const mask = buildObjectMask(info);
  const components = findConnectedComponents(mask, smallWidth, smallHeight);

  const imageArea = sourceWidth * sourceHeight;
  const candidates = [];

  for (const component of components) {
    const smallBoxArea = component.w * component.h;
    const smallImageArea = smallWidth * smallHeight;
    const smallBoxAreaRatio = smallBoxArea / smallImageArea;

    if (smallBoxAreaRatio < MIN_OBJECT_AREA_RATIO) continue;
    if (smallBoxAreaRatio > MAX_OBJECT_AREA_RATIO) continue;

    const box = {
      x: component.x * scaleX,
      y: component.y * scaleY,
      w: component.w * scaleX,
      h: component.h * scaleY
    };

    const expanded = expandBox(box, sourceWidth, sourceHeight, 0.12);
    const areaRatio = (expanded.w * expanded.h) / imageArea;
    const sideRatio = Math.min(expanded.w / sourceWidth, expanded.h / sourceHeight);
    const aspect = expanded.w / Math.max(1, expanded.h);
    const centerX = (expanded.x + expanded.w / 2) / sourceWidth;
    const centerY = (expanded.y + expanded.h / 2) / sourceHeight;
    const centerDist = Math.hypot(centerX - 0.5, centerY - 0.5);

    if (sideRatio < MIN_BOX_SIDE_RATIO) continue;
    if (aspect > 4.5 || aspect < 0.22) continue;

    const fillRatio = component.area / Math.max(1, smallBoxArea);
    const sizeScore = 1 - Math.min(1, Math.abs(areaRatio - 0.38) / 0.38);
    const centerScore = 1 - Math.min(1, centerDist / 0.72);

    const objectness =
      fillRatio * 0.48 +
      sizeScore * 0.32 +
      centerScore * 0.20;

    candidates.push({
      ...expanded,
      objectness,
      source: "object-mask"
    });
  }

  const selected = nonMaxSuppression(candidates, MAX_CANDIDATE_BOXES);

  if (selected.length < 3) {
    selected.push(...makeFallbackCenterBoxes(sourceWidth, sourceHeight));
  }

  const fullBox = {
    x: 0,
    y: 0,
    w: sourceWidth,
    h: sourceHeight,
    objectness: 0.18,
    source: "full-image"
  };

  selected.push(fullBox);

  return nonMaxSuppression(selected, MAX_CANDIDATE_BOXES);
}

async function predictSource(sourceCanvasOrImage) {
  const embedding = getEmbedding(sourceCanvasOrImage);
  const pred = classifierModel.predict(embedding.expandDims(0));
  const probs = await pred.data();

  embedding.dispose();
  pred.dispose();

  let topIndex = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[topIndex]) topIndex = i;
  }

  const confidence = probs[topIndex];
  const parsed = parseClassName(classNames[topIndex] || "unknown__uncertain");

  return {
    className: classNames[topIndex] || "unknown__uncertain",
    confidence,
    materialClass: parsed.materialClass,
    contaminationClass: parsed.contaminationClass
  };
}

async function findBestRegionPrediction(sourceImage) {
  const { width, height } = getSourceSize(sourceImage);
  const boxes = generateCandidateBoxes(sourceImage);

  const predictions = [];
  let best = null;

  for (let i = 0; i < boxes.length; i++) {
    const box = boxes[i];
    const crop = makeCropCanvas(sourceImage, box);
    const pred = await predictSource(crop);

    const areaRatio = (box.w * box.h) / (width * height);
    const centerX = (box.x + box.w / 2) / width;
    const centerY = (box.y + box.h / 2) / height;
    const centerDist = Math.hypot(centerX - 0.5, centerY - 0.5);

    /*
      v24 점수 방식
      - 분류 신뢰도만 믿지 않음
      - 객체처럼 보이는 정도(objectness)를 크게 반영
      - 화면 전체 박스나 배경으로 보이는 박스는 점수를 낮춤
    */
    const objectness = typeof box.objectness === "number" ? box.objectness : 0.3;
    const fullImagePenalty = box.source === "full-image" ? 0.18 : 0;
    const badSizePenalty = areaRatio > 0.82 ? 0.12 : 0;

    const score =
      pred.confidence * 0.58 +
      objectness * 0.34 -
      centerDist * 0.04 -
      Math.abs(areaRatio - 0.38) * 0.04 -
      fullImagePenalty -
      badSizePenalty;

    const item = {
      ...pred,
      box,
      score,
      areaRatio,
      objectness,
      source: box.source || "unknown"
    };

    predictions.push(item);

    if (!best || score > best.score) {
      best = item;
    }

    if (i % 2 === 0) {
      await tf.nextFrame();
    }
  }

  if (!predictions.length || !best) {
    return {
      materialClass: "unknown",
      contaminationClass: "uncertain",
      confidence: 0,
      box: { x: width * 0.15, y: height * 0.10, w: width * 0.70, h: height * 0.80 },
      score: 0,
      objectness: 0,
      lowConfidence: true
    };
  }

  const reliablePredictions = predictions.filter(pred => {
    return pred.objectness >= 0.28 && pred.source !== "full-image";
  });

  const votingPredictions = reliablePredictions.length
    ? reliablePredictions
    : predictions;

  const materialScores = {};
  const contaminationScores = {};

  for (const pred of votingPredictions) {
    if (!materialScores[pred.materialClass]) {
      materialScores[pred.materialClass] = 0;
    }

    if (!contaminationScores[pred.contaminationClass]) {
      contaminationScores[pred.contaminationClass] = 0;
    }

    const voteWeight =
      Math.max(0.04, pred.confidence) *
      Math.max(0.25, pred.objectness);

    materialScores[pred.materialClass] += voteWeight;
    contaminationScores[pred.contaminationClass] += voteWeight;
  }

  const materialClass = Object.entries(materialScores)
    .sort((a, b) => b[1] - a[1])[0][0];

  const contaminationClass = Object.entries(contaminationScores)
    .sort((a, b) => b[1] - a[1])[0][0];

  const sameMaterialPreds = votingPredictions.filter(pred => pred.materialClass === materialClass);
  const representative = sameMaterialPreds.sort((a, b) => b.score - a.score)[0] || best;

  const confidenceValues = sameMaterialPreds.map(pred => pred.confidence);
  const avgConfidence =
    confidenceValues.reduce((sum, value) => sum + value, 0) / Math.max(1, confidenceValues.length);

  const finalConfidence = Math.max(representative.confidence, avgConfidence);

  return {
    ...representative,
    materialClass,
    contaminationClass,
    confidence: finalConfidence,
    lowConfidence:
      finalConfidence < 0.32 ||
      representative.objectness < 0.25 ||
      representative.source === "full-image"
  };
}

function drawAnnotatedCanvas(sourceImage, prediction) {
  const { width, height } = getSourceSize(sourceImage);

  const canvas = document.createElement("canvas");
  const maxWidth = 900;
  const scale = Math.min(1, maxWidth / width);

  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);

  const ctx = canvas.getContext("2d");
  ctx.drawImage(sourceImage, 0, 0, canvas.width, canvas.height);

  const sx = canvas.width / width;
  const sy = canvas.height / height;

  const box = prediction.box || { x: 0, y: 0, w: width, h: height };
  const x = box.x * sx;
  const y = box.y * sy;
  const w = box.w * sx;
  const h = box.h * sy;

  const isWeak = prediction.lowConfidence || prediction.source === "full-image";

  ctx.strokeStyle = isWeak ? "#ffb448" : "#59f16d";
  ctx.lineWidth = Math.max(3, canvas.width * 0.004);
  ctx.shadowColor = isWeak ? "rgba(255,180,72,.35)" : "rgba(89,241,109,.35)";
  ctx.shadowBlur = 10;
  ctx.strokeRect(x, y, w, h);
  ctx.shadowBlur = 0;

  const materialKor = MATERIAL_LABELS_KOR[prediction.materialClass] || "판정불가";
  const contaminationKor = CONTAMINATION_LABELS_KOR[prediction.contaminationClass] || "판정불가";
  const label = `${materialKor} / ${contaminationKor} ${(prediction.confidence * 100).toFixed(1)}%`;

  ctx.font = `bold ${Math.max(14, Math.round(canvas.width * 0.022))}px sans-serif`;
  const paddingX = 10;
  const textWidth = ctx.measureText(label).width;
  const labelW = textWidth + paddingX * 2;
  const labelH = Math.max(28, Math.round(canvas.width * 0.04));
  const labelX = Math.min(x, canvas.width - labelW - 4);
  const labelY = Math.max(0, y - labelH - 6);

  ctx.fillStyle = "rgba(13, 24, 18, .88)";
  ctx.fillRect(labelX, labelY, labelW, labelH);

  ctx.strokeStyle = isWeak ? "rgba(255,180,72,.95)" : "rgba(89,241,109,.95)";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(labelX, labelY, labelW, labelH);

  ctx.fillStyle = isWeak ? "#ffb448" : "#59f16d";
  ctx.textBaseline = "middle";
  ctx.fillText(label, labelX + paddingX, labelY + labelH / 2);

  canvas.className = "detected-canvas";
  return canvas;
}

function renderPredictionResult(resultEl, sourceImage, prediction, sourceType = "이미지") {
  const materialKor = MATERIAL_LABELS_KOR[prediction.materialClass] || "판정불가";
  const contaminationKor = CONTAMINATION_LABELS_KOR[prediction.contaminationClass] || "판정불가";

  const disposal =
    prediction.contaminationClass === "dirty"
      ? `세척 필요 / ${materialKor}류`
      : prediction.materialClass === "unknown"
        ? "재확인 필요"
        : `정상 배출 / ${materialKor}류`;

  let advice = `📌 박스는 AI가 배경과 다른 물체 후보 영역을 먼저 찾은 뒤 분석한 영역입니다.`;

  if (prediction.lowConfidence) {
    advice += ` 신뢰도가 낮거나 객체 영역이 불안정합니다. 재활용품을 화면 중앙에 더 크게 놓고, 단색 배경에서 촬영하면 정확도가 올라갑니다.`;
  } else if (prediction.contaminationClass === "dirty") {
    advice += ` ${materialKor}에 오염이 감지되었습니다. 내용물을 비우고 세척한 뒤 분리배출하세요.`;
  } else {
    advice += ` 깨끗한 ${materialKor}로 판단됩니다. 해당 수거함에 분리배출하세요.`;
  }

  const objectnessText =
    typeof prediction.objectness === "number"
      ? `${(prediction.objectness * 100).toFixed(1)}%`
      : "확인 불가";

  const wrap = document.createElement("div");
  const canvasWrap = document.createElement("div");
  canvasWrap.className = "detected-canvas-wrap";
  canvasWrap.appendChild(drawAnnotatedCanvas(sourceImage, prediction));

  const resultSummaryHtml = `
    <p class="result-caption">${sourceType}에서 배경과 다른 객체 후보 영역을 찾고, 그 영역을 중심으로 분류했습니다.</p>

    <div class="result-summary">
      <div class="result-metric">
        <span>재질</span>
        <strong>${materialKor}</strong>
        <em>${(prediction.confidence * 100).toFixed(1)}% 신뢰도</em>
      </div>

      <div class="result-metric">
        <span>오염도</span>
        <strong>${contaminationKor}</strong>
        <em>${prediction.lowConfidence ? "신뢰도 낮음" : "AI 분석 결과"}</em>
      </div>

      <div class="result-metric">
        <span>객체 영역 점수</span>
        <strong>${objectnessText}</strong>
        <em>${prediction.source === "full-image" ? "전체 이미지 fallback" : "객체 후보 탐지"}</em>
      </div>
    </div>

    <table class="result-table">
      <thead>
        <tr>
          <th>번호</th>
          <th>분석 방식</th>
          <th>재질</th>
          <th>신뢰도</th>
          <th>객체 점수</th>
          <th>오염도</th>
          <th>배출 방법</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>1</td>
          <td>${prediction.source === "object-mask" ? "객체 후보 영역 탐지" : "보조 후보 영역"}</td>
          <td>${materialKor}</td>
          <td>${prediction.confidence.toFixed(3)}</td>
          <td>${typeof prediction.objectness === "number" ? prediction.objectness.toFixed(3) : "-"}</td>
          <td>${contaminationKor}</td>
          <td>${disposal}</td>
        </tr>
      </tbody>
    </table>

    <div class="advice-box ${prediction.contaminationClass === "dirty" || prediction.lowConfidence ? "dirty" : ""}">
      ${advice}
    </div>
  `;

  wrap.appendChild(canvasWrap);

  const body = document.createElement("div");
  body.innerHTML = resultSummaryHtml;
  wrap.appendChild(body);

  resultEl.innerHTML = "";
  resultEl.appendChild(wrap);
}

function renderEmptyResult(resultEl, message) {
  resultEl.innerHTML = `<div class="empty-note">${message}</div>`;
}

async function predictImage(imageElement, resultEl, sourceType = "이미지") {
  if (!classifierModel || !classNames.length) {
    alert("아직 모델이 준비되지 않았습니다. 모델 상태를 확인해주세요.");
    return;
  }

  if (!imageElement) {
    alert("분석할 이미지를 먼저 선택하거나 촬영하세요.");
    return;
  }

  renderEmptyResult(resultEl, "객체 후보 영역을 찾고 분석 중입니다. 잠시만 기다려주세요...");
  await prepareBackend();
  await loadBaseModel();

  try {
    const prediction = await findBestRegionPrediction(imageElement);
    renderPredictionResult(resultEl, imageElement, prediction, sourceType);
  } catch (err) {
    console.error(err);
    renderEmptyResult(resultEl, "분석 중 오류가 발생했습니다. 다른 이미지를 시도해보세요.");
  }
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

    if (els.result) {
      renderEmptyResult(els.result, "이미지를 선택했습니다. 이제 분석 버튼을 눌러주세요.");
    }
  });
}

if (els.predictBtn) {
  els.predictBtn.addEventListener("click", () => {
    predictImage(uploadImage, els.result, "업로드 이미지");
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
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0);

    canvas.style.display = "block";

    if (els.cameraEmpty) {
      els.cameraEmpty.style.display = "none";
    }

    const img = new Image();
    img.onload = () => {
      cameraImage = img;
      if (els.cameraResult) {
        renderEmptyResult(els.cameraResult, "촬영이 완료되었습니다. 이제 분석 버튼을 눌러주세요.");
      }
    };
    img.src = canvas.toDataURL("image/jpeg");
  });
}

if (els.cameraPredictBtn) {
  els.cameraPredictBtn.addEventListener("click", () => {
    predictImage(cameraImage, els.cameraResult, "카메라 촬영 이미지");
  });
}

if (els.trainManifestBtn) {
  els.trainManifestBtn.addEventListener("click", async () => {
    await autoTrain();
  });
}

if (els.trainSelectedBtn && els.zipInput) {
  els.trainSelectedBtn.addEventListener("click", async () => {
    const files = Array.from(els.zipInput.files || []).filter(file =>
      String(file.name || "").toLowerCase().endsWith(".zip")
    );

    if (!files.length) {
      alert("학습할 ZIP 파일을 먼저 선택하세요.");
      return;
    }

    try {
      await trainFromZipFiles(files);
    } catch (err) {
      setStatus("선택 ZIP 학습 실패", err.message, "error");
      log(`오류: ${err.message}`);
      alert(`선택한 ZIP 학습에 실패했습니다.\n\n${err.message}`);
    }
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
  injectRuntimeStyles();
  applyVersionLabel();
  await prepareBackend();

  if (els.result) {
    renderEmptyResult(els.result, "이미지를 업로드한 뒤 분석을 시작하세요.");
  }

  if (els.cameraResult) {
    renderEmptyResult(els.cameraResult, "카메라 촬영 후 분석 버튼을 누르세요.");
  }

  setProgress(3);
  setStatus("모델 확인 중...", `저장된 모델이 있는지 빠르게 확인합니다. ${APP_VERSION}`, "loading");

  const hasModel = await Promise.race([
    loadSavedModelFast(),
    timeout(3500)
  ]);

  if (hasModel) {
    setProgress(100);
    setStatus("분석 준비 완료", `저장된 모델을 불러왔습니다. ${APP_VERSION}`, "ready");
    return;
  }

  setProgress(8);
  setStatus("자동 학습 시작", `저장된 모델이 없어 학습을 시작합니다. ${APP_VERSION}`, "loading");

  await autoTrain();
}

startApp();
