const state = {
  socket: null,
  rawSegments: [],
  cleanSegments: [],
  stopped: false,
  summarizing: false,
  summaryText: "",
};

const el = {
  statusLine: document.querySelector("#statusLine"),
  refreshButton: document.querySelector("#refreshButton"),
  startButton: document.querySelector("#startButton"),
  stopButton: document.querySelector("#stopButton"),
  sourceSelect: document.querySelector("#sourceSelect"),
  micDevice: document.querySelector("#micDevice"),
  systemDevice: document.querySelector("#systemDevice"),
  cleanupToggle: document.querySelector("#cleanupToggle"),
  rawOutput: document.querySelector("#rawOutput"),
  cleanOutput: document.querySelector("#cleanOutput"),
  sessionOutput: document.querySelector("#sessionOutput"),
  unifiedOutput: document.querySelector("#unifiedOutput"),
  summaryOutput: document.querySelector("#summaryOutput"),
  summarizeButton: document.querySelector("#summarizeButton"),
};

function setStatus(message) {
  el.statusLine.textContent = message;
}

function option(device) {
  const opt = document.createElement("option");
  opt.value = String(device.id);
  opt.textContent = `${device.name} (${device.default_sample_rate}Hz)`;
  if (device.is_default_input) opt.dataset.default = "true";
  if (device.is_blackhole) opt.dataset.blackhole = "true";
  return opt;
}

async function refresh() {
  const [statusResponse, devicesResponse] = await Promise.all([
    fetch("/api/status"),
    fetch("/api/devices"),
  ]);
  const status = await statusResponse.json();
  const devices = await devicesResponse.json();

  el.micDevice.replaceChildren();
  el.systemDevice.replaceChildren();

  for (const device of devices.devices || []) {
    el.micDevice.append(option(device));
    el.systemDevice.append(option(device));
  }

  const defaultMic = [...el.micDevice.options].find((item) => item.dataset.default === "true");
  if (defaultMic) defaultMic.selected = true;
  const blackhole = [...el.systemDevice.options].find((item) => item.dataset.blackhole === "true");
  if (blackhole) blackhole.selected = true;

  const parts = [];
  parts.push(status.ffmpeg ? "ffmpeg آماده" : "ffmpeg پیدا نشد");
  parts.push(status.mlxWhisper ? "Whisper آماده" : "Whisper نصب نشده");
  parts.push(status.blackhole.available ? "BlackHole آماده" : "BlackHole نصب/انتخاب نشده");
  parts.push(
    status.ollama.modelAvailable
      ? `Ollama آماده: ${status.ollama.model}`
      : "Ollama cleanup در دسترس نیست",
  );
  if (devices.error) parts.push(`خطای صدا: ${devices.error}`);
  setStatus(parts.join(" · "));
}

function appendSegment(target, segment) {
  const item = document.createElement("span");
  item.className = "segment";
  item.dataset.index = String(segment.index);
  const stamp =
    segment.startLabel && segment.endLabel
      ? `<span class="stamp">${segment.startLabel} → ${segment.endLabel}</span>`
      : "";
  item.innerHTML = `${stamp}${escapeHtml(segment.text)}`;

  const existing = target.querySelector(`[data-index="${segment.index}"]`);
  if (existing) {
    existing.replaceWith(item);
  } else {
    target.append(item);
  }
  target.scrollTop = target.scrollHeight;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function selectedValue(select) {
  return select.value ? Number(select.value) : null;
}

function normalizeSegmentText(value) {
  return String(value || "").replace(/\s+/g, " ").trim();
}

function sessionSegments() {
  const length = Math.max(state.rawSegments.length, state.cleanSegments.length);
  const segments = [];

  for (let index = 0; index < length; index += 1) {
    const segment = state.cleanSegments[index] || state.rawSegments[index];
    const text = normalizeSegmentText(segment?.text);
    if (text) segments.push({ ...segment, index, text });
  }

  return segments;
}

function unifiedSessionText() {
  return sessionSegments().map((segment) => segment.text).join("\n\n");
}

function setSessionActionState() {
  const unifiedText = unifiedSessionText();
  const hasUnifiedText = Boolean(unifiedText);
  const hasSummary = Boolean(state.summaryText.trim());
  const canUseSessionOutput = state.stopped && hasUnifiedText;

  el.sessionOutput.hidden = !canUseSessionOutput;
  document
    .querySelectorAll('[data-session-copy="unified"], [data-session-download="unified"]')
    .forEach((button) => {
      button.disabled = !canUseSessionOutput;
    });

  el.summarizeButton.disabled = !canUseSessionOutput || state.summarizing;
  document
    .querySelectorAll('[data-session-copy="summary"], [data-session-download="summary"]')
    .forEach((button) => {
      button.disabled = !canUseSessionOutput || !hasSummary || state.summarizing;
    });
}

function renderSessionOutput() {
  el.unifiedOutput.textContent = unifiedSessionText();
  setSessionActionState();
}

function resetSessionOutput() {
  state.stopped = false;
  state.summarizing = false;
  state.summaryText = "";
  el.unifiedOutput.textContent = "";
  el.summaryOutput.textContent = "";
  setSessionActionState();
}

function start() {
  state.rawSegments = [];
  state.cleanSegments = [];
  el.rawOutput.replaceChildren();
  el.cleanOutput.replaceChildren();
  resetSessionOutput();

  const params = new URLSearchParams({
    source: el.sourceSelect.value,
    cleanup: el.cleanupToggle.checked ? "true" : "false",
  });
  const mic = selectedValue(el.micDevice);
  const system = selectedValue(el.systemDevice);
  if (mic !== null) params.set("mic_device", String(mic));
  if (system !== null) params.set("system_device", String(system));

  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  state.socket = new WebSocket(`${protocol}://${window.location.host}/ws/transcribe?${params}`);

  state.socket.addEventListener("open", () => {
    el.startButton.disabled = true;
    el.stopButton.disabled = false;
    setStatus("ضبط شروع شد. اولین خروجی بعد از چند ثانیه می‌آید.");
  });

  state.socket.addEventListener("message", (event) => {
    const data = JSON.parse(event.data);
    if (data.event === "status") setStatus(data.message);
    if (data.event === "partial") {
      setStatus(data.text || "در حال ترنسکریپت قطعه فعلی...");
      appendSegment(el.rawOutput, data);
    }
    if (data.event === "final") {
      state.rawSegments[data.index] = data;
      appendSegment(el.rawOutput, data);
      renderSessionOutput();
      setStatus("متن خام دریافت شد؛ ضبط ادامه دارد.");
    }
    if (data.event === "cleaning") {
      appendSegment(el.cleanOutput, data);
    }
    if (data.event === "cleaned") {
      state.cleanSegments[data.index] = data;
      appendSegment(el.cleanOutput, data);
      renderSessionOutput();
      if (data.cleanupPending) {
        setStatus("متن خام در پنل پاک‌سازی‌شده قرار گرفت؛ Ollama در پس‌زمینه در حال اصلاح است.");
      } else {
        setStatus(data.cleanupFailed ? "پاک‌سازی کامل نشد؛ متن خام نمایش داده شد." : "نسخه پاک‌سازی‌شده آماده شد.");
      }
    }
    if (data.event === "error") setStatus(data.message);
  });

  state.socket.addEventListener("close", () => {
    el.startButton.disabled = false;
    el.stopButton.disabled = true;
    state.socket = null;
  });
}

function stop() {
  if (!state.socket) return;
  state.stopped = true;
  renderSessionOutput();
  state.socket.send(JSON.stringify({ action: "stop" }));
  state.socket.close();
}

function collectText(kind) {
  const segments = kind === "clean" ? state.cleanSegments : state.rawSegments;
  return segments.filter(Boolean).map((segment) => segment.text).join("\n");
}

function srtFromSegments(segments) {
  return segments
    .filter(Boolean)
    .map((segment, index) =>
      [
        String(index + 1),
        `${segment.startLabel} --> ${segment.endLabel}`,
        segment.text,
        "",
      ].join("\n"),
    )
    .join("\n");
}

function download(name, type, content) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

async function summarizeSession() {
  const text = unifiedSessionText();
  if (!state.stopped || !text) {
    setStatus("برای ساخت خلاصه، اول ضبط را متوقف کنید و مطمئن شوید متن ثبت شده است.");
    return;
  }

  state.summarizing = true;
  state.summaryText = "";
  el.summaryOutput.textContent = "در حال ساخت خلاصه مفصل با Ollama...";
  setSessionActionState();
  setStatus("در حال ساخت خلاصه مفصل با Ollama...");

  try {
    const response = await fetch("/api/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || "خلاصه‌سازی انجام نشد.");
    }

    state.summaryText = String(data.summary || "").trim();
    el.summaryOutput.textContent = state.summaryText || "خلاصه‌ای تولید نشد.";
    setStatus(state.summaryText ? "خلاصه سشن آماده شد." : "خلاصه‌سازی خروجی خالی برگرداند.");
  } catch (error) {
    state.summaryText = "";
    el.summaryOutput.textContent = "";
    setStatus(`خطا در خلاصه‌سازی: ${error.message}`);
  } finally {
    state.summarizing = false;
    setSessionActionState();
  }
}

document.addEventListener("click", async (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;

  const copyTarget = target.dataset.copy;
  if (copyTarget) {
    const text = document.querySelector(`#${copyTarget}`)?.innerText || "";
    await navigator.clipboard.writeText(text);
    setStatus("متن کپی شد.");
  }

  const sessionCopy = target.dataset.sessionCopy;
  if (sessionCopy) {
    const text = sessionCopy === "summary" ? state.summaryText : unifiedSessionText();
    if (!text.trim()) {
      setStatus("متنی برای کپی وجود ندارد.");
      return;
    }
    await navigator.clipboard.writeText(text);
    setStatus(sessionCopy === "summary" ? "خلاصه کپی شد." : "متن یکپارچه کپی شد.");
  }

  const sessionDownload = target.dataset.sessionDownload;
  if (sessionDownload) {
    const text = sessionDownload === "summary" ? state.summaryText : unifiedSessionText();
    if (!text.trim()) {
      setStatus("متنی برای دانلود وجود ندارد.");
      return;
    }
    const fileName =
      sessionDownload === "summary" ? "transcript-fa-summary.txt" : "transcript-fa-unified.txt";
    download(fileName, "text/plain", text);
    setStatus(sessionDownload === "summary" ? "خلاصه دانلود شد." : "متن یکپارچه دانلود شد.");
  }

  if (target.id === "summarizeButton") {
    await summarizeSession();
  }

  const exportType = target.dataset.export;
  if (exportType) {
    const segments = state.cleanSegments.some(Boolean) ? state.cleanSegments : state.rawSegments;
    if (exportType === "txt") download("transcript-fa.txt", "text/plain", collectText("clean") || collectText("raw"));
    if (exportType === "json") download("transcript-fa.json", "application/json", JSON.stringify(segments.filter(Boolean), null, 2));
    if (exportType === "srt") download("transcript-fa.srt", "text/plain", srtFromSegments(segments));
  }
});

el.refreshButton.addEventListener("click", refresh);
el.startButton.addEventListener("click", start);
el.stopButton.addEventListener("click", stop);

refresh().catch((error) => setStatus(`خطا در بررسی وضعیت: ${error.message}`));
