// =========================================================
// AUTHENTINOTE FRONTEND
// =========================================================


// =========================================================
// API URL
// =========================================================
//
// If frontend is opened from:
// http://127.0.0.1:8000
//
// API becomes:
// http://127.0.0.1:5000
//
// If opened from another computer using:
// http://192.168.0.106:8000
//
// API becomes:
// http://192.168.0.106:5000
// =========================================================

const API_BASE =
    `${window.location.protocol}//${window.location.hostname}:5000`;

const DETECT_URL =
    `${API_BASE}/api/detect`;


// =========================================================
// ELEMENTS
// =========================================================

const imageInput =
    document.getElementById("imageInput");

const chooseButton =
    document.getElementById("chooseButton");

const cameraButton =
    document.getElementById("cameraButton");

const detectButton =
    document.getElementById("detectButton");

const previewArea =
    document.getElementById("previewArea");

const previewPlaceholder =
    document.getElementById("previewPlaceholder");

const previewImage =
    document.getElementById("previewImage");

const fileName =
    document.getElementById("fileName");

const loading =
    document.getElementById("loading");

const errorBox =
    document.getElementById("errorBox");

const uploadCard =
    document.getElementById("uploadCard");

const resultCard =
    document.getElementById("resultCard");

const resultImage =
    document.getElementById("resultImage");

const denominationResult =
    document.getElementById("denominationResult");

const denominationConfidence =
    document.getElementById(
        "denominationConfidence"
    );

const denominationProgress =
    document.getElementById(
        "denominationProgress"
    );

const authenticityResult =
    document.getElementById(
        "authenticityResult"
    );

const authenticityConfidence =
    document.getElementById(
        "authenticityConfidence"
    );

const authenticityProgress =
    document.getElementById(
        "authenticityProgress"
    );

const authSymbol =
    document.getElementById("authSymbol");

const resultMessage =
    document.getElementById("resultMessage");

const backButton =
    document.getElementById("backButton");

const speakButton =
    document.getElementById("speakButton");

const languageSelect =
    document.getElementById("languageSelect");


// =========================================================
// CURRENT IMAGE
// =========================================================

let selectedFile = null;

let lastResult = null;


// =========================================================
// CHOOSE IMAGE
// =========================================================

chooseButton.addEventListener(
    "click",
    () => {

        imageInput.removeAttribute(
            "capture"
        );

        imageInput.click();

    }
);


// =========================================================
// CAMERA
// =========================================================

cameraButton.addEventListener(
    "click",
    () => {

        imageInput.setAttribute(
            "capture",
            "environment"
        );

        imageInput.click();

    }
);


// =========================================================
// FILE SELECTED
// =========================================================

imageInput.addEventListener(
    "change",
    () => {

        const file =
            imageInput.files[0];

        if (!file) {
            return;
        }

        handleFile(file);

    }
);


// =========================================================
// HANDLE FILE
// =========================================================

function handleFile(file) {

    const allowedTypes = [
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp"
    ];

    if (
        !allowedTypes.includes(
            file.type
        )
    ) {

        showError(
            "Please select a JPG, PNG, WEBP or BMP image."
        );

        return;
    }


    selectedFile = file;


    // Show file name

    fileName.textContent =
        file.name;

    fileName.classList.remove(
        "hidden"
    );


    // Preview

    const objectURL =
        URL.createObjectURL(file);

    previewImage.src =
        objectURL;

    previewImage.classList.remove(
        "hidden"
    );

    previewPlaceholder.classList.add(
        "hidden"
    );


    // Enable detection

    detectButton.disabled =
        false;


    hideError();

}


// =========================================================
// DETECT
// =========================================================

detectButton.addEventListener(
    "click",
    detectCurrency
);


async function detectCurrency() {

    if (!selectedFile) {

        showError(
            "Please select a currency image first."
        );

        return;
    }


    hideError();


    // UI state

    detectButton.disabled =
        true;

    loading.classList.remove(
        "hidden"
    );


    const formData =
        new FormData();

    formData.append(
        "image",
        selectedFile
    );


    try {

        console.log(
            "Sending image to:",
            DETECT_URL
        );


        const response =
            await fetch(
                DETECT_URL,
                {
                    method: "POST",
                    body: formData
                }
            );


        const data =
            await response.json();


        console.log(
            "Backend response:",
            data
        );


        if (!response.ok) {

            throw new Error(
                data.message ||
                data.error ||
                "Detection failed."
            );
        }


        if (!data.success) {

            throw new Error(
                data.message ||
                "Currency could not be detected."
            );
        }


        lastResult = data;


        showResult(data);

    }

    catch (error) {

        console.error(
            "Detection error:",
            error
        );


        showError(
            error.message ||
            "Failed to connect to the detection server."
        );

    }

    finally {

        loading.classList.add(
            "hidden"
        );

        detectButton.disabled =
            false;

    }
}


// =========================================================
// SHOW RESULT
// =========================================================

function showResult(data) {

    uploadCard.classList.add(
        "hidden"
    );

    resultCard.classList.remove(
        "hidden"
    );


    // Image

    if (selectedFile) {

        resultImage.src =
            URL.createObjectURL(
                selectedFile
            );

    }


    // =====================================================
    // DENOMINATION
    // =====================================================

    if (data.denomination) {

        denominationResult.textContent =
            `₹${data.denomination}`;

    }

    else {

        denominationResult.textContent =
            "Not detected";

    }


    const denomConfidence =
        Number(
            data.denomination_confidence || 0
        );


    denominationConfidence.textContent =
        `${denomConfidence.toFixed(2)}%`;


    denominationProgress.style.width =
        `${Math.min(
            denomConfidence,
            100
        )}%`;


    // =====================================================
    // AUTHENTICITY
    // =====================================================

    const authenticity =
        String(
            data.authenticity || ""
        ).toUpperCase();


    if (
        authenticity === "REAL"
    ) {

        authenticityResult.textContent =
            "REAL";

        authenticityResult.style.color =
            "#16875a";

        authSymbol.textContent =
            "✓";

        authSymbol.style.color =
            "#16875a";

        authSymbol.style.background =
            "#e9f8f1";

    }

    else if (
        authenticity === "FAKE"
    ) {

        authenticityResult.textContent =
            "FAKE";

        authenticityResult.style.color =
            "#d64040";

        authSymbol.textContent =
            "!";

        authSymbol.style.color =
            "#d64040";

        authSymbol.style.background =
            "#fff0f0";

    }

    else if (
        authenticity === "UNSUPPORTED"
    ) {

        authenticityResult.textContent =
            "UNSUPPORTED";

        authenticityResult.style.color =
            "#a06b00";

        authSymbol.textContent =
            "—";

        authSymbol.style.color =
            "#a06b00";

        authSymbol.style.background =
            "#fff7df";

    }

    else {

        authenticityResult.textContent =
            "—";

    }


    // =====================================================
    // AUTHENTICITY CONFIDENCE
    // =====================================================

    const authConfidence =
        Number(
            data.authenticity_confidence || 0
        );


    authenticityConfidence.textContent =
        `${authConfidence.toFixed(2)}%`;


    authenticityProgress.style.width =
        `${Math.min(
            authConfidence,
            100
        )}%`;


    // =====================================================
    // MESSAGE
    // =====================================================

    resultMessage.textContent =
        data.message ||
        "Analysis completed.";


    // Scroll

    resultCard.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });

}


// =========================================================
// BACK
// =========================================================

backButton.addEventListener(
    "click",
    resetApplication
);


function resetApplication() {

    selectedFile = null;

    lastResult = null;

    imageInput.value = "";

    previewImage.src = "";

    previewImage.classList.add(
        "hidden"
    );

    previewPlaceholder.classList.remove(
        "hidden"
    );

    fileName.textContent = "";

    fileName.classList.add(
        "hidden"
    );

    detectButton.disabled =
        true;

    resultCard.classList.add(
        "hidden"
    );

    uploadCard.classList.remove(
        "hidden"
    );

    hideError();


    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });

}


// =========================================================
// ERROR
// =========================================================

function showError(message) {

    errorBox.textContent =
        message;

    errorBox.classList.remove(
        "hidden"
    );

}


function hideError() {

    errorBox.textContent = "";

    errorBox.classList.add(
        "hidden"
    );

}


// =========================================================
// SPEECH
// =========================================================

speakButton.addEventListener(
    "click",
    speakResult
);


function speakResult() {

    if (!lastResult) {

        return;
    }


    if (
        !("speechSynthesis" in window)
    ) {

        alert(
            "Speech synthesis is not supported by this browser."
        );

        return;
    }


    window.speechSynthesis.cancel();


    const language =
        languageSelect.value;


    const denomination =
        lastResult.denomination;


    const authenticity =
        lastResult.authenticity;


    let text;


    // =====================================================
    // ENGLISH
    // =====================================================

    if (language === "en-IN") {

        if (
            authenticity === "UNSUPPORTED"
        ) {

            text =
                `The detected denomination is ` +
                `2000 rupees. ` +
                `Authenticity verification ` +
                `is not available for this denomination.`;

        }

        else {

            text =
                `The detected denomination is ` +
                `${denomination} rupees. ` +
                `The currency note appears to be ` +
                `${authenticity}.`;

        }

    }


    // =====================================================
    // HINDI
    // =====================================================

    else if (language === "hi-IN") {

        text =
            `यह नोट ${denomination || ""} रुपये का है। ` +
            `यह नोट ${authenticity === "REAL"
                ? "असली"
                : authenticity === "FAKE"
                    ? "नकली"
                    : "सत्यापित नहीं"} है।`;

    }


    // =====================================================
    // TAMIL
    // =====================================================

    else if (language === "ta-IN") {

        text =
            `இது ${denomination || ""} ரூபாய் நோட்டு. ` +
            `இந்த நோட்டு ` +
            `${authenticity === "REAL"
                ? "உண்மையானது"
                : authenticity === "FAKE"
                    ? "போலியானது"
                    : "சரிபார்க்கப்படவில்லை"}.`;

    }


    // =====================================================
    // TELUGU
    // =====================================================

    else if (language === "te-IN") {

        text =
            `ఇది ${denomination || ""} రూపాయల నోటు. ` +
            `ఈ నోటు ` +
            `${authenticity === "REAL"
                ? "అసలైనది"
                : authenticity === "FAKE"
                    ? "నకిలీది"
                    : "ధృవీకరించబడలేదు"}.`;

    }


    // =====================================================
    // BENGALI
    // =====================================================

    else if (language === "bn-IN") {

        text =
            `এটি ${denomination || ""} টাকার নোট। ` +
            `নোটটি ` +
            `${authenticity === "REAL"
                ? "আসল"
                : authenticity === "FAKE"
                    ? "জাল"
                    : "যাচাই করা যায়নি"}.`;

    }


    // =====================================================
    // MARATHI
    // =====================================================

    else if (language === "mr-IN") {

        text =
            `ही ${denomination || ""} रुपयांची नोट आहे. ` +
            `ही नोट ` +
            `${authenticity === "REAL"
                ? "खरी"
                : authenticity === "FAKE"
                    ? "बनावट"
                    : "सत्यापित झालेली नाही"}.`;

    }


    // =====================================================
    // KANNADA
    // =====================================================

    else if (language === "kn-IN") {

        text =
            `ಇದು ${denomination || ""} ರೂಪಾಯಿ ನೋಟು. ` +
            `ಈ ನೋಟು ` +
            `${authenticity === "REAL"
                ? "ಅಸಲಿ"
                : authenticity === "FAKE"
                    ? "ನಕಲಿ"
                    : "ಪರಿಶೀಲಿಸಲಾಗಿಲ್ಲ"}.`;

    }


    // =====================================================
    // MALAYALAM
    // =====================================================

    else if (language === "ml-IN") {

        text =
            `ഇത് ${denomination || ""} രൂപയുടെ നോട്ടാണ്. ` +
            `ഈ നോട്ട് ` +
            `${authenticity === "REAL"
                ? "യഥാർത്ഥമാണ്"
                : authenticity === "FAKE"
                    ? "വ്യാജമാണ്"
                    : "പരിശോധിച്ചിട്ടില്ല"}.`;

    }


    const utterance =
        new SpeechSynthesisUtterance(
            text
        );


    utterance.lang =
        language;

    utterance.rate =
        0.85;

    utterance.pitch =
        1;


    window.speechSynthesis.speak(
        utterance
    );

}


// =========================================================
// STARTUP HEALTH CHECK
// =========================================================

async function checkBackend() {

    try {

        const response =
            await fetch(
                `${API_BASE}/api/health`
            );


        const data =
            await response.json();


        console.log(
            "Backend health:",
            data
        );

    }

    catch (error) {

        console.warn(
            "Backend health check failed.",
            error
        );

    }

}


checkBackend();