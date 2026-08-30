// Drag-and-drop on the predict page
(function () {
  const input = document.getElementById('imageInput');
  const body = document.body;
  if (!input) return;

  ['dragenter', 'dragover'].forEach(e => {
    body.addEventListener(e, ev => { ev.preventDefault(); body.classList.add('border', 'border-warning'); });
  });
  ['dragleave', 'drop'].forEach(e => {
    body.addEventListener(e, ev => { ev.preventDefault(); body.classList.remove('border', 'border-warning'); });
  });
  body.addEventListener('drop', ev => {
    const files = ev.dataTransfer.files;
    if (files.length) {
      input.files = files;
      input.dispatchEvent(new Event('change'));
    }
  });
})();
