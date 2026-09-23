/**
 * CropAI Pathology Workbench — Landing Page Interactive Engine
 * Controls the live environmental risk sandbox and pipeline scrollspy.
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- 1. Interactive Micro-Climate Sandbox ---
    const humiditySlider = document.getElementById('humiditySlider');
    const tempSlider = document.getElementById('tempSlider');
    const severitySlider = document.getElementById('severitySlider');

    const humidityVal = document.getElementById('humidityVal');
    const tempVal = document.getElementById('tempVal');
    const severityVal = document.getElementById('severityVal');

    const riskScoreEl = document.getElementById('sandboxRiskScore');
    const riskBadgeEl = document.getElementById('sandboxRiskBadge');
    const driverEl = document.getElementById('sandboxDriver');

    function calculateSimulatedRisk() {
        if (!humiditySlider || !tempSlider || !severitySlider) return;

        const h = parseFloat(humiditySlider.value);
        const t = parseFloat(tempSlider.value);
        const s = parseFloat(severitySlider.value);

        // Update labels
        if (humidityVal) humidityVal.textContent = `${h}%`;
        if (tempVal) tempVal.textContent = `${t}°C`;
        if (severityVal) severityVal.textContent = `${s}%`;

        // Mathematical formulation mirroring expert_system/risk_calculator.py
        let baseRisk = s * 0.45;

        // Humidity factor (fungal pathogens thrive above 70% RH)
        let hFactor = 1.0;
        if (h > 80) hFactor = 1.6;
        else if (h > 70) hFactor = 1.3;
        else if (h < 45) hFactor = 0.6;

        // Thermal factor (optimal spore germination between 20°C - 30°C)
        let tFactor = 1.0;
        if (t >= 20 && t <= 30) tFactor = 1.3;
        else if (t < 14 || t > 35) tFactor = 0.7;

        let finalScore = Math.min(100, Math.max(0, (baseRisk * 0.5 + 15) * hFactor * tFactor));
        finalScore = Math.round(finalScore * 10) / 10;

        if (riskScoreEl) {
            riskScoreEl.textContent = finalScore.toFixed(1);
        }

        // Determine level & badge style
        let label = 'Low Risk';
        let color = 'var(--status-healthy)';
        let bg = 'var(--status-healthy-bg)';
        let border = 'var(--status-healthy-border)';
        let driver = 'Ambient conditions are suboptimal for aggressive pathogen escalation.';

        if (finalScore >= 75) {
            label = 'Critical Risk';
            color = 'var(--status-danger)';
            bg = 'var(--status-danger-bg)';
            border = 'var(--status-danger-border)';
            driver = 'Extreme humidity and favorable temperatures create high probability of rapid spore outbreak.';
        } else if (finalScore >= 50) {
            label = 'Elevated Risk';
            color = 'var(--accent)';
            bg = 'var(--accent-tint)';
            border = 'var(--accent-border)';
            driver = 'High relative humidity sustains foliar moisture, promoting fungal incubation.';
        } else if (finalScore >= 25) {
            label = 'Moderate Risk';
            color = 'var(--accent)';
            bg = 'var(--accent-tint)';
            border = 'var(--accent-border)';
            driver = 'Moderate infection observed. Standard preventive foliar scouting advised.';
        }

        if (riskScoreEl) {
            riskScoreEl.style.color = color;
        }

        if (riskBadgeEl) {
            riskBadgeEl.textContent = label;
            riskBadgeEl.style.color = color;
            riskBadgeEl.style.backgroundColor = bg;
            riskBadgeEl.style.borderColor = border;
        }

        if (driverEl) {
            driverEl.textContent = driver;
        }
    }

    if (humiditySlider && tempSlider && severitySlider) {
        humiditySlider.addEventListener('input', calculateSimulatedRisk);
        tempSlider.addEventListener('input', calculateSimulatedRisk);
        severitySlider.addEventListener('input', calculateSimulatedRisk);
        calculateSimulatedRisk();
    }

    // --- 2. Pipeline Stepper Scroll Spy ---
    const stepItems = document.querySelectorAll('.pipeline-stepper .step-item');
    const methodBlocks = document.querySelectorAll('.method-stack article');

    if (stepItems.length > 0 && methodBlocks.length > 0) {
        window.addEventListener('scroll', () => {
            const scrollPos = window.scrollY + 240;

            methodBlocks.forEach((block, index) => {
                const top = block.offsetTop;
                const height = block.offsetHeight;

                if (scrollPos >= top && scrollPos < top + height) {
                    stepItems.forEach(item => item.classList.remove('active'));
                    if (stepItems[index]) {
                        stepItems[index].classList.add('active');
                    }
                }
            });
        }, { passive: true });
    }
});
