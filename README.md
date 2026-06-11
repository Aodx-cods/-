<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>스마트 재활용 분류 시스템</title>
  <link rel="stylesheet" href="style.css" />
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.18.0/dist/tf.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@2.1.1/dist/mobilenet.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js"></script>
</head>
<body>
  <div class="shell">
    <aside class="left">
      <div class="brand">
        <div class="mark">♻</div>
        <div>
          <h1>Recycle Vision</h1>
          <p>스마트 재활용 분류</p>
        </div>
      </div>

      <div class="mode-tabs">
        <button class="mode active" data-view="uploadView">이미지 업로드</button>
        <button class="mode" data-view="cameraView">카메라 촬영</button>
      </div>

      <div class="status-card">
        <span id="statusChip" class="chip loading">준비 중</span>
        <h2 id="modelStatusText">모델을 준비하고 있습니다</h2>
        <p id="statusDesc">처음 접속하면 ZIP 데이터를 자동으로 압축 해제하고 학습합니다.</p>
        <div class="progress-line"><span id="progressBar"></span></div>
        <strong id="progressText">0%</strong>
      </div>

      <div class="quick-actions">
        <button id="retrainBtn">데이터 다시 학습</button>
        <button id="clearModelBtn">저장 모델 초기화</button>
      </div>

      <details class="train-details">
        <summary>학습 로그 보기</summary>
        <pre id="trainLog">학습 로그가 여기에 표시됩니다.</pre>
      </details>
    </aside>

    <main class="main">
      <section class="hero">
        <div>
          <p class="kicker">AI Recycling Classifier</p>
          <h2>사진 한 장으로 재질과 오염도를 확인하세요</h2>
          <p>업로드 또는 카메라 촬영을 선택하면 학습된 모델이 재활용 품목과 세척 필요 여부를 분석합니다.</p>
        </div>
        <div class="stats">
          <div><span>분류 대상</span><strong>종이 · 캔 · 비닐 · 유리 · 스티로폼</strong></div>
          <div><span>상태 판정</span><strong>오염 / 깨끗함</strong></div>
        </div>
      </section>

      <section id="uploadView" class="view active">
        <div class="grid two">
          <article class="panel">
            <div class="panel-head">
              <div>
                <span>Input</span>
                <h3>원본 이미지</h3>
              </div>
              <label class="upload-btn">
                이미지 선택
                <input id="imageInput" type="file" accept="image/*" />
              </label>
            </div>
            <div class="frame image-frame">
              <canvas id="previewCanvas"></canvas>
              <div id="uploadEmpty" class="empty"><b>이미지를 선택하세요</b><small>JPG, PNG 파일을 사용할 수 있습니다.</small></div>
            </div>
            <button id="predictBtn" class="analyze">분석 시작</button>
          </article>

          <article class="panel result-panel">
            <div class="panel-head">
              <div>
                <span>Output</span>
                <h3>분석 결과</h3>
              </div>
            </div>
            <div id="result" class="result-box empty-result">
              <b>아직 분석 결과가 없습니다</b>
              <small>이미지를 선택한 뒤 분석을 시작하세요.</small>
            </div>
          </article>
        </div>
      </section>

      <section id="cameraView" class="view">
        <div class="grid two">
          <article class="panel">
            <div class="panel-head">
              <div>
                <span>Camera</span>
                <h3>카메라 화면</h3>
              </div>
              <button id="startCameraBtn" class="sub-btn">카메라 켜기</button>
            </div>
            <div class="frame camera-frame">
              <video id="cameraVideo" autoplay playsinline></video>
              <div class="focus-box"></div>
            </div>
            <div class="row-actions">
              <button id="captureBtn" class="sub-btn">촬영</button>
              <button id="cameraPredictBtn" class="analyze inline">촬영 이미지 분석</button>
            </div>
          </article>

          <article class="panel result-panel">
            <div class="panel-head">
              <div>
                <span>Captured</span>
                <h3>촬영 이미지와 결과</h3>
              </div>
            </div>
            <div class="frame small-frame">
              <canvas id="cameraCanvas"></canvas>
              <div id="cameraEmpty" class="empty"><b>촬영된 이미지가 없습니다</b><small>촬영 버튼을 먼저 누르세요.</small></div>
            </div>
            <div id="cameraResult" class="result-box empty-result camera-result">
              <b>아직 분석 결과가 없습니다</b>
              <small>촬영 후 분석을 시작하세요.</small>
            </div>
          </article>
        </div>
      </section>
    </main>
  </div>

  <input id="zipInput" type="file" multiple accept=".zip" hidden />
  <button id="trainSelectedBtn" hidden></button>
  <button id="trainManifestBtn" hidden></button>
  <progress id="trainProgress" value="0" max="100" hidden></progress>

  <script src="app.js"></script>
</body>
</html>
