async function translateText() {
    const inputText = document.getElementById("inputText").value;
    const outputText = document.getElementById("outputText");
    const sourceText = document.getElementById("sourceText");

    if (!inputText.trim()) {
        outputText.innerText = "Please enter text to translate.";
        sourceText.innerText = "";
        return;
    }

    outputText.innerText = "Translating...";
    sourceText.innerText = "";

    const response = await fetch("/translate", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: inputText })
    });

    const data = await response.json();

    outputText.innerText = data.translation;
    sourceText.innerText = "Source: " + data.source;
}