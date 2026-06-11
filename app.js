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
  can: ["캔", "can", "aluminum", "알루미늄"],
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

const IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"];
const MODEL_KEY = "indexeddb://recycle-classifier-v1";

let mobilenetModel = null;
let classifierModel = null;
let classNames = [];
let currentImageElement = null;
let cameraStream = null;

const els = {
  zipInput: document.getElementById("zipInput"),
  trainSelectedBtn: document.getElementById("trainSelectedBtn"),
  trainManifestBtn: document.getElementById("trainManifestBtn"),
  clearModelBtn: document.getElementById("clearModelBtn"),
  trainLog: document.getElementById("trainLog"),
  trainProgress: document.getElementById("trainProgress"),
  imageInput: document.getElementById("imageInput"),
  previewCanvas: document.getElementById("previewCanvas"),
  predictBtn: document.getElementById("predictBtn"),
  result: document.getElementById("result"),
  cameraVideo: document.getElementById("cameraVideo"),
  startCameraBtn: document.getElementById("startCameraBtn"),
  captureBtn: document.getElementById("captureBtn")
};

function log(message) {
  if (!els.trainLog) return;
  els.trainLog.textContent += `${message}\n`;
  els.trainLog.scrollTop = els.trainLog.scrollHeight;
}

function setProgress(value) {
  if (!els.trainProgress) return;
  els.trainProgress.value = Math.max(0, Math.min(100, value));
}

function normalizeName(text) {
  return String(text || "")
    .toLowerCase()
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

async function loadBaseModel() {
  if (!mobilenetModel) {
    log("MobileNet 기본 모델 로딩 중...");
    mobilenetModel = await mobilenet.load({ version: 2, alpha: 1.0 });
    log("MobileNet 로딩 완료");
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

  const samples = [];
  const entries = Object.values(zip.files).filter(entry => {
    return !entry.dir && isImageFile(entry.name);
  });

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

  for (const file of zipFiles) {
    log(`ZIP 읽는 중: ${file.name}`);

    const zipSamples = await readZipFile(file, file.name);

    if (!zipSamples.length) {
      log(`이미지가 없어 건너뜀: ${file.name}`);
      continue;
    }

    const materialClass = detectClass(file.name, MATERIAL_KEYWORDS, "unknown");
    const contaminationClass = detectClass(file.name, CONTAMINATION_KEYWORDS, "uncertain");

    log(`  이미지 ${zipSamples.length}장 감지`);
    log(`  품목: ${MATERIAL_LABELS_KOR[materialClass] || materialClass}`);
    log(`  오염도: ${CONTAMINATION_LABELS_KOR[contaminationClass] || contaminationClass}`);

    samples.push(...zipSamples);
  }

  return samples.filter(sample => {
    return sample.materialClass !== "unknown" && sample.contaminationClass !== "uncertain";
  });
}

async function loadZipFilesFromManifest() {
  const manifestResult = await fetchFirstAvailable([
    "zips/manifest.json",
    "manifest.json"
  ]);

  log(`manifest 위치: ${manifestResult.path}`);

  const manifest = await manifestResult.res.json();

  if (!manifest.files || !Array.isArray(manifest.files)) {
    throw new Error('manifest.json 형식이 올바르지 않습니다. { "files": [...] } 형식이어야 합니다.');
  }

  const files = [];

  for (const rawName of manifest.files) {
    const name = String(rawName).trim();

    if (!name) continue;

    const cleanName = name.replace(/^zips\//, "");

    const candidatePaths = [
      `zips/${cleanName}`,
      cleanName,
      name
    ];

    log(`ZIP 불러오는 중: ${name}`);

    const zipResult = await fetchFirstAvailable(candidatePaths);
    const blob = await zipResult.res.blob();

    log(`  불러옴: ${zipResult.path}`);

    files.push(new File([blob], cleanName, { type: "application/zip" }));
  }

  return files;
}

async function trainFromZipFiles(zipFiles) {
  els.trainLog.textContent = "";
  setProgress(0);

  await loadBaseModel();

  const samples = await buildTrainingData(zipFiles);

  if (samples.length < 4) {
    throw new Error(
      "학습 가능한 이미지가 너무 적습니다. ZIP 파일명에 품목명과 오염o/오염x/오염0이 들어있는지 확인하세요."
    );
  }

  const uniqueClasses = [
    ...new Set(
      samples.map(sample => {
        return makeClassName(sample.materialClass, sample.contaminationClass);
      })
    )
  ].sort();

  if (uniqueClasses.length < 2) {
    throw new Error("최소 2개 이상의 학습 클래스가 필요합니다. 예: 종이 오염o, 종이 오염x");
  }

  classNames = uniqueClasses;

  log(`학습 클래스: ${classNames.join(", ")}`);

  const xs = [];
  const ys = [];

  for (let i = 0; i < samples.length; i++) {
    const sample = samples[i];
    const img = await fileToImage(sample.blob);
    const embedding = getEmbedding(img);

    xs.push(await embedding.array());
    embedding.dispose();

    const classIndex = classNames.indexOf(
      makeClassName(sample.materialClass, sample.contaminationClass)
    );

    ys.push(classIndex);

    if (i % 5 === 0) {
      setProgress(10 + Math.round((i / samples.length) * 45));
    }

    await tf.nextFrame();
  }

  const xTensor = tf.tensor2d(xs);
  const yTensor = tf.oneHot(tf.tensor1d(ys, "int32"), classNames.length);

  classifierModel = tf.sequential();

  classifierModel.add(
    tf.layers.dense({
      inputShape: [xTensor.shape[1]],
      units: 128,
      activation: "relu"
    })
  );

  classifierModel.add(
    tf.layers.dropout({
      rate: 0.2
    })
  );

  classifierModel.add(
    tf.layers.dense({
      units: classNames.length,
      activation: "softmax"
    })
  );

  classifierModel.compile({
    optimizer: tf.train.adam(0.001),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });

  log("브라우저 내 학습 시작...");

  await classifierModel.fit(xTensor, yTensor, {
    epochs: 25,
    batchSize: Math.min(16, samples.length),
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        const acc = logs.acc ?? logs.accuracy ?? 0;

        log(
          `epoch ${epoch + 1}/25 - loss ${logs.loss.toFixed(4)} - acc ${(acc * 100).toFixed(1)}%`
        );

        setProgress(60 + Math.round(((epoch + 1) / 25) * 30));

        await tf.nextFrame();
      }
    }
  });

  xTensor.dispose();
  yTensor.dispose();

  classifierModel.classNames = classNames;

  await classifierModel.save(MODEL_KEY);
  localStorage.setItem("recycle-class-names", JSON.stringify(classNames));

  setProgress(100);

  log("학습 완료. 모델이 브라우저에 저장되었습니다.");
}

async function loadSavedModel() {
  try {
    await loadBaseModel();

    classifierModel = await tf.loadLayersModel(MODEL_KEY + "/model.json");
    classNames = JSON.parse(localStorage.getItem("recycle-class-names") || "[]");

    if (classNames.length) {
      log("저장된 학습 모델을 불러왔습니다.");
    }
  } catch (err) {
    log("저장된 모델이 없습니다. 먼저 ZIP 데이터셋으로 학습하세요.");
  }
}

function drawImageToCanvas(img) {
  const canvas = els.previewCanvas;
  const maxW = 900;
  const scale = Math.min(1, maxW / img.width);

  canvas.width = Math.round(img.width * scale);
  canvas.height = Math.round(img.height * scale);

  const ctx = canvas.getContext("2d");

  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  canvas.style.display = "block";
}

async function predictCurrentImage() {
  if (!classifierModel || !classNames.length) {
    alert("먼저 ZIP 데이터셋으로 학습하세요.");
    return;
  }

  if (!currentImageElement) {
    alert("분석할 이미지를 먼저 선택하거나 촬영하세요.");
    return;
  }

  await loadBaseModel();

  const embedding = getEmbedding(currentImageElement);
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

  const advice =
    contaminationClass === "dirty"
      ? `⚠️ ${materialKor}에 오염이 감지되었습니다. 물로 깨끗이 씻거나 이물질을 제거한 뒤 배출하세요.`
      : `✅ 깨끗한 ${materialKor}로 판단됩니다. 알맞은 수거함에 분리배출하세요.`;

  els.result.className = "result";

  els.result.innerHTML = `
    <div class="metric">
      <div class="label">분류 품목</div>
      <div class="value">${materialKor}</div>
      <div class="conf">신뢰도 ${(confidence * 100).toFixed(1)}%</div>
    </div>

    <div class="metric">
      <div class="label">위생 상태</div>
      <div class="value">${contaminationKor}</div>
      <div class="conf">브라우저 학습 모델 기준</div>
    </div>

    <div class="advice">${advice}</div>
  `;
}

if (els.trainSelectedBtn) {
  els.trainSelectedBtn.addEventListener("click", async () => {
    try {
      const files = [...els.zipInput.files];

      if (!files.length) {
        alert("ZIP 파일을 먼저 선택하세요.");
        return;
      }

      await trainFromZipFiles(files);
    } catch (err) {
      log(`오류: ${err.message}`);
      alert(err.message);
    }
  });
}

if (els.trainManifestBtn) {
  els.trainManifestBtn.addEventListener("click", async () => {
    try {
      els.trainLog.textContent = "";

      log("manifest.json에서 ZIP 목록을 불러오는 중...");

      const files = await loadZipFilesFromManifest();

      await trainFromZipFiles(files);
    } catch (err) {
      log(`오류: ${err.message}`);
      alert(err.message);
    }
  });
}

if (els.clearModelBtn) {
  els.clearModelBtn.addEventListener("click", async () => {
    try {
      await tf.io.removeModel(MODEL_KEY);
    } catch (_) {}

    localStorage.removeItem("recycle-class-names");

    classifierModel = null;
    classNames = [];

    log("저장 모델을 초기화했습니다.");
  });
}

if (els.imageInput) {
  els.imageInput.addEventListener("change", async event => {
    const file = event.target.files[0];

    if (!file) return;

    currentImageElement = await fileToImage(file);

    drawImageToCanvas(currentImageElement);
  });
}

if (els.predictBtn) {
  els.predictBtn.addEventListener("click", predictCurrentImage);
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
  els.captureBtn.addEventListener("click", async () => {
    const video = els.cameraVideo;

    if (!video.videoWidth) {
      alert("카메라를 먼저 켜세요.");
      return;
    }

    const canvas = els.previewCanvas;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");

    ctx.drawImage(video, 0, 0);

    canvas.style.display = "block";

    const img = new Image();

    img.onload = () => {
      currentImageElement = img;
    };

    img.src = canvas.toDataURL("image/jpeg");
  });
}

document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));

    btn.classList.add("active");

    const panel = document.getElementById(btn.dataset.tab);

    if (panel) {
      panel.classList.add("active");
    }
  });
});

loadSavedModel();
