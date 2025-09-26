async function msg() {
    // Collect form values
    const input = {
        Gender: document.getElementById("gender").value === "male" ? "Male" : "Female",
        Married: document.getElementById("married").value,
        Dependents: document.getElementById("dependents").value,
        Education: document.getElementById("education").value === "graduate" ? "Graduate" : "Not Graduate",
        Self_Employed: document.getElementById("self_employed").value === "yes" ? "Yes" : "No",
        ApplicantIncome: parseFloat(document.getElementById("applicant_income").value) || 0,
        CoapplicantIncome: parseFloat(document.getElementById("coapplicant_income").value) || 0,
        LoanAmount: parseFloat(document.getElementById("loan_amt").value) || 0,
        Loan_Amount_Term: parseFloat(document.getElementById("loan_amt_term").value) || 360,
        Credit_History: parseInt(document.getElementById("credit_history").value),
        Property_Area: document.getElementById("property_area").value === "urban" ? "Urban" : "Rural"
    };

    try {
        const response = await fetch("https://loan-backend.onrender.com/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(input)
        });

        // handle non-200 responses
        if (!response.ok) {
            const errBody = await response.json().catch(() => null);
            const errMsg = (errBody && errBody.detail) ? errBody.detail : `Server returned ${response.status}`;
            alert("Server error: " + errMsg);
            return;
        }

        const result = await response.json();

        // save both input and result so result page can show what was used
        localStorage.setItem("loanInput", JSON.stringify(input));
        localStorage.setItem("loanResult", JSON.stringify(result));

        // Redirect to result.html
        window.location.href = "result.html";

    } catch (error) {
        alert("Network or CORS error: " + error);
    }
}
