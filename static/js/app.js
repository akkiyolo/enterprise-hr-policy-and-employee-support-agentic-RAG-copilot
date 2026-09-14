const chat = document.getElementById("chat");
const form = document.getElementById("chatForm");
const question = document.getElementById("question");
const trace = document.getElementById("trace");
const sourceUsed = document.getElementById("sourceUsed");
const traceStatus = document.getElementById("traceStatus");

const modal = document.getElementById("uploadModal");
const openUpload = document.getElementById("openUpload");
const closeUpload = document.getElementById("closeUpload");

const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
const adminKey = document.getElementById("adminKey");
const uploadStatus = document.getElementById("uploadStatus");


/* =========================================================
   UTILITIES
========================================================= */

function escapeHtml(value = "") {
  return String(value).replace(
    /[&<>'"]/g,
    (char) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      "'": "&#39;",
      '"': "&quot;",
    })[char]
  );
}


function formatText(value = "") {
  return escapeHtml(value).replace(/\n/g, "<br>");
}


/* =========================================================
   CHAT
========================================================= */

function addMessage(
  role,
  text,
  source = "",
  citations = []
) {
  const wrap = document.createElement("div");

  wrap.className = `message ${role}`;

  const avatar = role === "assistant" ? "AI" : "";

  const citationHtml =
    citations.length > 0
      ? `
        <div class="citations">
          <strong>Sources</strong>

          ${citations
            .map((citation) => {
              if (citation.url) {
                return `
                  <a
                    href="${escapeHtml(citation.url)}"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    ${escapeHtml(citation.title || citation.url)}
                  </a>
                `;
              }

              return escapeHtml(
                citation.title || "Source"
              );
            })
            .join("<br>")}
        </div>
      `
      : "";


  const sourceHtml = source
    ? `
      <div class="answer-source">
        Source: ${escapeHtml(source)}
      </div>
    `
    : "";


  wrap.innerHTML = `
    <div class="avatar">${avatar}</div>

    <div class="bubble">
      ${formatText(text)}

      ${sourceHtml}

      ${citationHtml}
    </div>
  `;


  chat.appendChild(wrap);

  requestAnimationFrame(() => {
    chat.scrollTo({
      top: chat.scrollHeight,
      behavior: "smooth",
    });
  });
}


/* =========================================================
   TRACE
========================================================= */

function renderTrace(items = []) {

  if (!items.length) {
    trace.innerHTML = `
      <div class="trace-empty">

        <div class="trace-empty-icon">
          <svg viewBox="0 0 24 24" fill="none">
            <path
              d="M12 3V7M12 17V21M3 12H7M17 12H21"
              stroke="currentColor"
              stroke-width="1.5"
              stroke-linecap="round"
            />
            <circle
              cx="12"
              cy="12"
              r="4"
              stroke="currentColor"
              stroke-width="1.5"
            />
          </svg>
        </div>

        <strong>No request yet</strong>

        <span>
          Agent execution steps will appear here.
        </span>

      </div>
    `;

    return;
  }


  trace.innerHTML = items
    .map(
      (item) => `
        <div class="trace-item">
          ${escapeHtml(item)}
        </div>
      `
    )
    .join("");
}


/* =========================================================
   ASK AGENT
========================================================= */

async function askAgent(q) {

  if (!q || !q.trim()) {
    return;
  }

  const cleanQuestion = q.trim();

  addMessage("user", cleanQuestion);

  question.value = "";

  autoResizeTextarea();

  renderTrace([
    "Initializing LangGraph workflow..."
  ]);

  sourceUsed.textContent = "Running";

  traceStatus.textContent = "RUNNING";

  const btn = form.querySelector(".send-btn");

  btn.disabled = true;

  const originalButton = btn.innerHTML;

  btn.innerHTML = `
    <span>Working</span>

    <svg
      class="loading-spinner"
      viewBox="0 0 24 24"
      fill="none"
    >
      <circle
        cx="12"
        cy="12"
        r="8"
        stroke="currentColor"
        stroke-width="1.7"
        stroke-linecap="round"
        stroke-dasharray="35 15"
      />
    </svg>
  `;


  try {

    const response = await fetch(
      "/api/chat",
      {
        method: "POST",

        headers: {
          "Content-Type": "application/json",
        },

        body: JSON.stringify({
          question: cleanQuestion,
        }),
      }
    );


    const data = await response.json();


    if (!response.ok) {
      throw new Error(
        data.detail || "Request failed"
      );
    }


    addMessage(
      "assistant",
      data.answer,
      data.source_used,
      data.citations || []
    );


    renderTrace(
      data.trace || []
    );


    sourceUsed.textContent =
      data.source_used || "Unknown";


    traceStatus.textContent = "COMPLETE";

  } catch (error) {

    addMessage(
      "assistant",
      `Something went wrong: ${error.message}`
    );


    renderTrace([
      "Request failed.",
      error.message
    ]);


    sourceUsed.textContent = "Error";

    traceStatus.textContent = "ERROR";

  } finally {

    btn.disabled = false;

    btn.innerHTML = originalButton;
  }
}


/* =========================================================
   FORM SUBMISSION
========================================================= */

form.addEventListener(
  "submit",
  (event) => {

    event.preventDefault();

    const q = question.value.trim();

    if (q) {
      askAgent(q);
    }
  }
);


/* =========================================================
   ENTER / SHIFT + ENTER
========================================================= */

question.addEventListener(
  "keydown",
  (event) => {

    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {

      event.preventDefault();

      form.requestSubmit();
    }
  }
);


/* =========================================================
   AUTO RESIZE TEXTAREA
========================================================= */

function autoResizeTextarea() {

  question.style.height = "auto";

  question.style.height =
    `${Math.min(question.scrollHeight, 140)}px`;
}


question.addEventListener(
  "input",
  autoResizeTextarea
);


/* =========================================================
   EXAMPLE QUESTIONS
========================================================= */

document
  .querySelectorAll(".example")
  .forEach((button) => {

    button.addEventListener(
      "click",
      () => {

        const text =
          button.textContent.trim();

        askAgent(text);
      }
    );
  });


/* =========================================================
   MODAL
========================================================= */

function openModal() {

  modal.classList.remove("hidden");

  document.body.style.overflow = "hidden";

  setTimeout(() => {
    adminKey.focus();
  }, 50);
}


function closeModal() {

  modal.classList.add("hidden");

  document.body.style.overflow = "";

  uploadStatus.textContent = "";
}


openUpload.addEventListener(
  "click",
  openModal
);


closeUpload.addEventListener(
  "click",
  closeModal
);


/* =========================================================
   CLOSE MODAL WHEN CLICKING BACKDROP
========================================================= */

modal.addEventListener(
  "click",
  (event) => {

    if (
      event.target.classList.contains(
        "modal-backdrop"
      )
    ) {
      closeModal();
    }
  }
);


/* =========================================================
   ESCAPE KEY
========================================================= */

document.addEventListener(
  "keydown",
  (event) => {

    if (
      event.key === "Escape" &&
      !modal.classList.contains("hidden")
    ) {
      closeModal();
    }
  }
);


/* =========================================================
   FILE SELECTION
========================================================= */

fileInput.addEventListener(
  "change",
  () => {

    const file = fileInput.files[0];

    if (!file) {
      return;
    }


    const fileDrop =
      document.querySelector(".file-drop");

    const fileText =
      fileDrop.querySelector(
        ".file-drop-text"
      );


    fileText.innerHTML = `
      <strong>
        ${escapeHtml(file.name)}
      </strong>

      <span>
        ${formatFileSize(file.size)}
      </span>
    `;
  }
);


/* =========================================================
   FILE SIZE
========================================================= */

function formatFileSize(bytes) {

  if (bytes < 1024) {
    return `${bytes} B`;
  }

  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }

  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}


/* =========================================================
   UPLOAD DOCUMENT
========================================================= */

uploadBtn.addEventListener(
  "click",
  async () => {

    const file =
      fileInput.files[0];

    const key =
      adminKey.value.trim();


    if (!file) {

      uploadStatus.textContent =
        "Choose a document first.";

      return;
    }


    if (!key) {

      uploadStatus.textContent =
        "Enter the admin API key.";

      return;
    }


    uploadBtn.disabled = true;


    const originalButton =
      uploadBtn.innerHTML;


    uploadBtn.innerHTML = `
      <span>Indexing...</span>

      <svg
        class="loading-spinner"
        viewBox="0 0 24 24"
        fill="none"
      >
        <circle
          cx="12"
          cy="12"
          r="8"
          stroke="currentColor"
          stroke-width="1.7"
          stroke-linecap="round"
          stroke-dasharray="35 15"
        />
      </svg>
    `;


    uploadStatus.textContent =
      "Uploading and indexing document...";


    const formData =
      new FormData();

    formData.append(
      "file",
      file
    );


    try {

      const response =
        await fetch(
          "/api/ingest",
          {
            method: "POST",

            headers: {
              "X-Admin-Key": key,
            },

            body: formData,
          }
        );


      const data =
        await response.json();


      if (!response.ok) {

        throw new Error(
          data.detail ||
          "Upload failed"
        );
      }


      uploadStatus.textContent =
        `Indexed ${data.file}: ${data.chunks} chunks.`;


      uploadStatus.style.color =
        "#70cf9c";


      uploadBtn.innerHTML = `
        <span>Document indexed</span>

        <svg
          viewBox="0 0 24 24"
          fill="none"
        >
          <path
            d="M5 12L10 17L19 7"
            stroke="currentColor"
            stroke-width="1.8"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
        </svg>
      `;


      setTimeout(() => {

        uploadBtn.innerHTML =
          originalButton;

        uploadBtn.disabled =
          false;

      }, 1800);


    } catch (error) {

      uploadStatus.textContent =
        `Error: ${error.message}`;

      uploadStatus.style.color =
        "#d78383";


      uploadBtn.innerHTML =
        originalButton;

      uploadBtn.disabled =
        false;
    }
  }
);


/* =========================================================
   SPINNER
========================================================= */

const spinnerStyles =
  document.createElement("style");

spinnerStyles.textContent = `
  .loading-spinner {
    width: 14px;
    height: 14px;

    animation:
      spin 800ms linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
`;

document.head.appendChild(
  spinnerStyles
);


/* =========================================================
   INITIALIZATION
========================================================= */

autoResizeTextarea();