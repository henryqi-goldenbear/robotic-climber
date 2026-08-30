// Drag-and-drop on the predict page.
(function () {
  const input = document.getElementById('imageInput');
  if (!input) return;

  ['dragenter', 'dragover'].forEach(eventName => {
    document.body.addEventListener(eventName, event => {
      event.preventDefault();
      document.body.classList.add('border', 'border-warning');
    });
  });
  ['dragleave', 'drop'].forEach(eventName => {
    document.body.addEventListener(eventName, event => {
      event.preventDefault();
      document.body.classList.remove('border', 'border-warning');
    });
  });
  document.body.addEventListener('drop', event => {
    if (event.dataTransfer.files.length) {
      input.files = event.dataTransfer.files;
      input.dispatchEvent(new Event('change'));
    }
  });
})();

// Mission-control subsystem status.
(function () {
  const livekitState = document.getElementById('livekitState');
  const visionState = document.getElementById('visionState');
  const terrainState = document.getElementById('terrainState');
  if (!livekitState || !visionState || !terrainState) return;

  fetch('/api/demo-status')
    .then(response => {
      if (!response.ok) throw new Error(`Status request failed: ${response.status}`);
      return response.json();
    })
    .then(status => {
      livekitState.textContent = status.livekit_online ? 'Online' : 'Standby';
      livekitState.className = status.livekit_online ? 'green' : 'amber';
      const visionReady = status.detector_ready && status.classifier_ready;
      visionState.textContent = visionReady ? 'Models ready' : 'Setup required';
      visionState.className = visionReady ? 'green' : 'amber';
      terrainState.textContent = status.avalanche_ready ? 'Model ready' : 'Setup required';
      terrainState.className = status.avalanche_ready ? 'green' : 'amber';
    })
    .catch(error => {
      livekitState.textContent = 'Unavailable';
      visionState.textContent = 'Unavailable';
      terrainState.textContent = 'Unavailable';
      console.error(error);
    });
})();

// A short, hands-free walkthrough for judge presentations.
(function () {
  const trigger = document.getElementById('guidedDemo');
  const toast = document.getElementById('demoToast');
  if (!trigger || !toast) return;

  const cards = Array.from(document.querySelectorAll('[data-demo-step]'));
  const step = document.getElementById('demoStep');
  const title = document.getElementById('demoTitle');
  const copy = document.getElementById('demoCopy');
  const stop = document.getElementById('stopDemo');
  const scenes = [
    ['01', 'Knot verification', 'The robot checks load-bearing rope geometry before movement.'],
    ['02', 'Terrain intelligence', 'Weather and snowpack history add a second layer of decision support.'],
    ['03', 'Resilient connection', 'LiveKit shares evidence while the safety loop stays on the edge.'],
  ];
  let timer = null;
  let sceneIndex = 0;

  function showScene(index) {
    cards.forEach((card, cardIndex) => card.classList.toggle('demo-active', cardIndex === index));
    step.textContent = scenes[index][0];
    title.textContent = scenes[index][1];
    copy.textContent = scenes[index][2];
    cards[index].scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  function endDemo() {
    if (timer) window.clearInterval(timer);
    timer = null;
    toast.classList.remove('visible');
    cards.forEach(card => card.classList.remove('demo-active'));
  }

  trigger.addEventListener('click', () => {
    endDemo();
    sceneIndex = 0;
    toast.classList.add('visible');
    showScene(sceneIndex);
    timer = window.setInterval(() => {
      sceneIndex += 1;
      if (sceneIndex >= scenes.length) {
        endDemo();
        document.getElementById('proof').scrollIntoView({ behavior: 'smooth' });
        return;
      }
      showScene(sceneIndex);
    }, 5200);
  });
  stop.addEventListener('click', endDemo);
})();
