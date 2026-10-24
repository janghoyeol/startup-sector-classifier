const form = document.getElementById("ask-form");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const topPickEl = document.getElementById("top-pick");
const rankingEl = document.getElementById("ranking");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = document.getElementById("text").value.trim();
  if (!text) return;

  statusEl.textContent = "예측 중… (무료 플랜이라 첫 요청은 서버가 깨어나는 데 최대 1분 걸릴 수 있습니다)";
  resultEl.hidden = true;

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error(`서버 오류 (${res.status})`);
    const data = await res.json();
    render(data);
    statusEl.textContent = "";
  } catch (err) {
    statusEl.textContent = `오류: ${err.message}`;
  }
});

function render(data) {
  const preds = data.predictions || [];
  if (preds.length === 0) {
    statusEl.textContent = "텍스트를 더 입력해주세요.";
    return;
  }

  const top = preds[0];
  topPickEl.innerHTML = `<span class="label">${escapeHtml(top.label)}</span> <span class="prob">${(top.probability * 100).toFixed(1)}%</span>`;

  rankingEl.innerHTML = "";
  for (const p of preds) {
    const li = document.createElement("li");
    const pct = (p.probability * 100).toFixed(1);
    li.innerHTML = `
      <span class="rank-label">${escapeHtml(p.label)}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${pct}%"></span></span>
      <span class="rank-pct">${pct}%</span>
    `;
    rankingEl.appendChild(li);
  }

  resultEl.hidden = false;
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}
