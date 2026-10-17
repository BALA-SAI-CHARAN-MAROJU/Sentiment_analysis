/**
 * main.js - Client-Side Interactivity for Sentiment Analysis Dashboard
 * Handles Light/Dark theme switching, AJAX predictions, Chart.js visualizations,
 * bulk CSV processing, and live social media feed actions.
 */

document.addEventListener("DOMContentLoaded", function () {
    initThemeToggle();
    initPresetButtons();
    initAnalyzeForm();
    initLiveFeedClassifier();
});

// 0. Theme Toggle Handler (Dark & Light)
function initThemeToggle() {
    const toggleBtn = document.getElementById("themeToggleBtn");
    const themeIcon = document.getElementById("themeIcon");
    const themeLabel = document.getElementById("themeLabel");

    function applyTheme(theme) {
        document.documentElement.setAttribute("data-theme", theme);
        localStorage.setItem("sentiment_app_theme", theme);
        if (themeIcon) {
            themeIcon.className = theme === "light" ? "fa-solid fa-sun text-warning" : "fa-solid fa-moon text-info";
        }
        if (themeLabel) {
            themeLabel.innerText = theme === "light" ? "Light Mode" : "Dark Mode";
        }
        // Trigger chart theme refresh
        window.dispatchEvent(new Event("themeChanged"));
    }

    const savedTheme = localStorage.getItem("sentiment_app_theme") || "dark";
    applyTheme(savedTheme);

    if (toggleBtn) {
        toggleBtn.addEventListener("click", function () {
            const current = document.documentElement.getAttribute("data-theme") || "dark";
            const next = current === "dark" ? "light" : "dark";
            applyTheme(next);
        });
    }
}

// 1. Preset Demo Buttons Handler
function initPresetButtons() {
    const presetButtons = document.querySelectorAll(".preset-btn");
    const textArea = document.getElementById("postTextInput");

    presetButtons.forEach(btn => {
        btn.addEventListener("click", function () {
            if (textArea) {
                textArea.value = this.getAttribute("data-text");
                const form = document.getElementById("analyzeForm");
                if (form) {
                    form.dispatchEvent(new Event("submit"));
                }
            }
        });
    });
}

// 2. Real-time Single Text Analysis Form
function initAnalyzeForm() {
    const form = document.getElementById("analyzeForm");
    const textArea = document.getElementById("postTextInput");
    const resultBox = document.getElementById("predictionResultBox");
    const spinner = document.getElementById("analyzeSpinner");
    const btn = document.getElementById("analyzeBtn");

    if (!form || !textArea) return;

    form.addEventListener("submit", async function (e) {
        e.preventDefault();
        const text = textArea.value.trim();
        if (!text) {
            alert("Please enter some text or select a preset post.");
            return;
        }

        if (spinner) spinner.classList.remove("d-none");
        if (btn) btn.disabled = true;

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: text, source: "Web Form" }),
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || "Failed to analyze sentiment.");
            }

            const data = await response.json();
            displayPredictionResult(data);
        } catch (error) {
            console.error("Prediction error:", error);
            alert("Error: " + error.message);
        } finally {
            if (spinner) spinner.classList.add("d-none");
            if (btn) btn.disabled = false;
        }
    });
}

function displayPredictionResult(data) {
    const resultBox = document.getElementById("predictionResultBox");
    if (!resultBox) return;

    resultBox.classList.remove("d-none");

    const sentiment = data.sentiment;
    const confidence = (data.confidence * 100).toFixed(1);
    const cleanedText = data.cleaned_text || "N/A";
    const probabilities = data.probabilities || {};

    let badgeClass = "badge-neu";
    let icon = "fa-meh";
    if (sentiment.toLowerCase() === "positive") {
        badgeClass = "badge-pos";
        icon = "fa-smile";
    } else if (sentiment.toLowerCase() === "negative") {
        badgeClass = "badge-neg";
        icon = "fa-frown";
    }

    const badgeElem = document.getElementById("resSentimentBadge");
    if (badgeElem) {
        badgeElem.className = `badge fs-5 px-3 py-2 ${badgeClass}`;
        badgeElem.innerHTML = `<i class="far ${icon} me-2"></i> ${sentiment}`;
    }

    const confElem = document.getElementById("resConfidence");
    if (confElem) confElem.innerText = `${confidence}%`;

    const cleanedElem = document.getElementById("resCleanedText");
    if (cleanedElem) cleanedElem.innerText = cleanedText;

    // Probabilities Progress Bars & Value Labels
    const posProb = ((probabilities.positive || 0) * 100).toFixed(1);
    const neuProb = ((probabilities.neutral || 0) * 100).toFixed(1);
    const negProb = ((probabilities.negative || 0) * 100).toFixed(1);

    const posBar = document.getElementById("probBarPos");
    const posVal = document.getElementById("probValPos");
    if (posBar) {
        posBar.style.width = `${posProb}%`;
        posBar.innerText = "";
    }
    if (posVal) {
        posVal.innerText = `${posProb}%`;
    }

    const neuBar = document.getElementById("probBarNeu");
    const neuVal = document.getElementById("probValNeu");
    if (neuBar) {
        neuBar.style.width = `${neuProb}%`;
        neuBar.innerText = "";
    }
    if (neuVal) {
        neuVal.innerText = `${neuProb}%`;
    }

    const negBar = document.getElementById("probBarNeg");
    const negVal = document.getElementById("probValNeg");
    if (negBar) {
        negBar.style.width = `${negProb}%`;
        negBar.innerText = "";
    }
    if (negVal) {
        negVal.innerText = `${negProb}%`;
    }

    // Explainability Top Words
    const explainBox = document.getElementById("explainTokensBox");
    if (explainBox) {
        explainBox.innerHTML = "";
        const topWords = data.explainability?.top_words || [];
        if (topWords.length > 0) {
            topWords.forEach(item => {
                const pill = document.createElement("span");
                pill.className = "word-pill";
                pill.innerHTML = `<strong>${item.word}</strong> <span>(${item.score})</span>`;
                explainBox.appendChild(pill);
            });
        } else {
            explainBox.innerHTML = `<span class="text-muted small">No distinct high-weight vocabulary tokens detected.</span>`;
        }
    }

    resultBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// 3. Live Social Media Stream Classifier
function initLiveFeedClassifier() {
    const buttons = document.querySelectorAll(".classify-live-btn");
    buttons.forEach(btn => {
        btn.addEventListener("click", async function () {
            const card = this.closest(".social-post-card");
            const text = card.getAttribute("data-text");
            const source = card.getAttribute("data-source") || "Live Social API";
            const targetBadge = card.querySelector(".post-sentiment-badge");

            this.disabled = true;
            this.innerHTML = `<span class="spinner-border spinner-border-sm"></span>`;

            try {
                const response = await fetch("/api/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text: text, source: source }),
                });
                const data = await response.json();

                let badgeClass = "badge-neu";
                if (data.sentiment.toLowerCase() === "positive") badgeClass = "badge-pos";
                if (data.sentiment.toLowerCase() === "negative") badgeClass = "badge-neg";

                if (targetBadge) {
                    targetBadge.className = `badge ${badgeClass} px-2 py-1`;
                    targetBadge.innerText = `${data.sentiment} (${(data.confidence * 100).toFixed(0)}%)`;
                }
                this.classList.add("d-none");
            } catch (err) {
                console.error(err);
                this.disabled = false;
                this.innerText = "Retry";
            }
        });
    });
}
