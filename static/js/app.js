/**
 * CropAI Pathology Workbench — Diagnosis Console Controller
 * Manages specimen ingestion, pipeline checklist orchestration,
 * and diagnostic dossier rendering.
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References ---
    const dropChamber = document.getElementById('dropChamber');
    const fileInput = document.getElementById('fileInput');
    const dropPlaceholder = document.getElementById('dropPlaceholder');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const previewClearBtn = document.getElementById('previewClearBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const diagnosisForm = document.getElementById('diagnosisForm');
    const formErrorMessage = document.getElementById('formErrorMessage');

    const emptyState = document.getElementById('emptyState');
    const loadingCard = document.getElementById('loadingCard');
    const resultsDossier = document.getElementById('resultsDossier');

    // Pipeline Checklist Steps
    const stepClassification = document.getElementById('stepClassification');
    const stepGradcam = document.getElementById('stepGradcam');
    const stepSeverity = document.getElementById('stepSeverity');
    const stepExpert = document.getElementById('stepExpert');

    // Visual Dock Elements
    const dockMainImage = document.getElementById('dockMainImage');
    const dockTabs = document.querySelectorAll('.dock-tab-btn');
    const dockCaptionText = document.getElementById('dockCaptionText');

    let currentFile = null;
    let diagnosisResult = null;
    let visualViews = {
        overlay: '',
        heatmap: '',
        original: ''
    };

    // --- File Ingestion Handling ---
    if (dropChamber && fileInput) {
        dropChamber.addEventListener('click', (e) => {
            if (e.target !== previewClearBtn) {
                fileInput.click();
            }
        });

        dropChamber.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                fileInput.click();
            }
        });

        dropChamber.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropChamber.classList.add('drag-over');
        });

        dropChamber.addEventListener('dragleave', () => {
            dropChamber.classList.remove('drag-over');
        });

        dropChamber.addEventListener('drop', (e) => {
            e.preventDefault();
            dropChamber.classList.remove('drag-over');
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleSelectedFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files && fileInput.files.length > 0) {
                handleSelectedFile(fileInput.files[0]);
            }
        });
    }

    if (previewClearBtn) {
        previewClearBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            resetFileSelection();
        });
    }

    function handleSelectedFile(file) {
        if (!file.type.match('image.*')) {
            showError('Selected file is not an accepted image format (JPG, PNG, WebP).');
            return;
        }

        currentFile = file;
        hideError();

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            dropPlaceholder.style.display = 'none';
            previewContainer.classList.add('active');
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function resetFileSelection() {
        currentFile = null;
        fileInput.value = '';
        previewImage.src = '';
        previewContainer.classList.remove('active');
        dropPlaceholder.style.display = 'block';
        analyzeBtn.disabled = true;
        hideError();
    }

    // --- Sample Specimen Buttons ---
    const sampleButtons = document.querySelectorAll('.sample-btn');
    sampleButtons.forEach(btn => {
        btn.addEventListener('click', async () => {
            const crop = btn.getAttribute('data-crop');
            const cropSelect = document.getElementById('cropType');
            if (cropSelect && crop) {
                cropSelect.value = crop;
            }

            // Fetch the default hero specimen as a test file
            try {
                const response = await fetch('/static/images/hero-leaf.jpg');
                const blob = await response.blob();
                const file = new File([blob], `${crop.toLowerCase()}_sample_leaf.jpg`, { type: 'image/jpeg' });
                handleSelectedFile(file);
            } catch (err) {
                console.warn('Could not load sample file directly:', err);
            }
        });
    });

    // --- Visual Dock Tabs Controller ---
    dockTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            dockTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            const viewType = tab.getAttribute('data-view');
            updateDockView(viewType);
        });
    });

    function updateDockView(viewType) {
        if (!dockMainImage || !diagnosisResult) return;

        if (viewType === 'overlay' && visualViews.overlay) {
            dockMainImage.src = visualViews.overlay;
            dockCaptionText.textContent = 'Grad-CAM Overlay: Warm highlights indicate regions driving the disease diagnosis.';
        } else if (viewType === 'heatmap' && visualViews.heatmap) {
            dockMainImage.src = visualViews.heatmap;
            dockCaptionText.textContent = 'Raw Heatmap: Pure thermal gradient of convolutional activations.';
        } else if (viewType === 'original' && visualViews.original) {
            dockMainImage.src = visualViews.original;
            dockCaptionText.textContent = 'Original Specimen: Unprocessed RGB input tensor.';
        }
    }

    // --- Form Submission & Pipeline Execution ---
    if (diagnosisForm) {
        diagnosisForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            if (!currentFile) {
                showError('Please select or drop a leaf image to evaluate.');
                return;
            }

            hideError();
            setPipelineRunning(true);

            const formData = new FormData(diagnosisForm);
            formData.set('image', currentFile);

            // Animate pipeline checklist steps
            startChecklistAnimation();

            try {
                const response = await fetch('/api/diagnose', {
                    method: 'POST',
                    body: formData,
                });

                const data = await response.json();

                if (!response.ok || !data.success) {
                    throw new Error(data.error || 'Server rejected pathology analysis request.');
                }

                diagnosisResult = data;
                completeChecklist();
                setTimeout(() => {
                    renderDiagnosisDossier(data);
                    setPipelineRunning(false);
                }, 400);

            } catch (err) {
                setPipelineRunning(false);
                showError(`Pathology analysis failed: ${err.message}`);
                console.error('Diagnosis error:', err);
            }
        });
    }

    function setPipelineRunning(isRunning) {
        if (isRunning) {
            analyzeBtn.disabled = true;
            emptyState.style.display = 'none';
            resultsDossier.classList.remove('active');
            loadingCard.classList.add('active');
        } else {
            loadingCard.classList.remove('active');
            analyzeBtn.disabled = false;
        }
    }

    function startChecklistAnimation() {
        // Reset steps
        [stepClassification, stepGradcam, stepSeverity, stepExpert].forEach(step => {
            if (step) {
                step.className = 'pipeline-step';
            }
        });

        if (stepClassification) stepClassification.className = 'pipeline-step in-progress';

        setTimeout(() => {
            if (stepClassification) stepClassification.className = 'pipeline-step complete';
            if (stepGradcam) stepGradcam.className = 'pipeline-step in-progress';
        }, 500);

        setTimeout(() => {
            if (stepGradcam) stepGradcam.className = 'pipeline-step complete';
            if (stepSeverity) stepSeverity.className = 'pipeline-step in-progress';
        }, 900);

        setTimeout(() => {
            if (stepSeverity) stepSeverity.className = 'pipeline-step complete';
            if (stepExpert) stepExpert.className = 'pipeline-step in-progress';
        }, 1300);
    }

    function completeChecklist() {
        [stepClassification, stepGradcam, stepSeverity, stepExpert].forEach(step => {
            if (step) {
                step.className = 'pipeline-step complete';
            }
        });
    }

    function renderDiagnosisDossier(data) {
        const pred = data.prediction || {};
        const grad = data.gradcam || {};
        const sev = data.severity || {};
        const risk = data.risk_assessment || {};
        const details = data.disease_details || {};

        // 1. Primary Verdict Banner
        const verdictBanner = document.getElementById('verdictBanner');
        const verdictCropTag = document.getElementById('verdictCropTag');
        const verdictDiseaseName = document.getElementById('verdictDiseaseName');
        const verdictScientific = document.getElementById('verdictScientific');
        const verdictConfidence = document.getElementById('verdictConfidence');

        const isHealthy = pred.is_healthy;
        verdictBanner.className = `verdict-banner ${isHealthy ? 'status-healthy' : (risk.risk_level === 'high' || risk.risk_level === 'critical' ? 'status-danger' : 'status-warning')}`;

        if (verdictCropTag) {
            verdictCropTag.textContent = `${details.crop || 'SOLANACEAE'} PATHOLOGY VERDICT`;
        }
        if (verdictDiseaseName) {
            verdictDiseaseName.textContent = pred.disease_name || pred.disease_label;
        }
        if (verdictScientific) {
            verdictScientific.textContent = details.scientific_name ? `Pathogen: ${details.scientific_name} (${details.pathogen_type || 'Microorganism'})` : 'Baseline: Healthy foliage';
        }
        if (verdictConfidence) {
            verdictConfidence.textContent = `${pred.confidence_pct}%`;
        }

        // 2. Visual Attribution Dock
        visualViews.overlay = grad.overlay_url || data.image_url;
        visualViews.heatmap = grad.heatmap_url || data.image_url;
        visualViews.original = data.image_url;

        // Reset to overlay view
        dockTabs.forEach(t => t.classList.remove('active'));
        const defaultTab = document.querySelector('.dock-tab-btn[data-view="overlay"]');
        if (defaultTab) defaultTab.classList.add('active');
        updateDockView('overlay');

        // 3. Severity Quantification
        const severityPctVal = document.getElementById('severityPctVal');
        const severityLevelBadge = document.getElementById('severityLevelBadge');
        const severityBarFill = document.getElementById('severityBarFill');
        const diseasedPixelsVal = document.getElementById('diseasedPixelsVal');
        const healthyPixelsVal = document.getElementById('healthyPixelsVal');

        const sevPct = sev.severity_percentage || 0;
        if (severityPctVal) severityPctVal.textContent = `${sevPct.toFixed(1)}%`;
        if (severityLevelBadge) severityLevelBadge.textContent = (sev.severity_level || 'MILD').toUpperCase();
        if (severityBarFill) severityBarFill.style.width = `${Math.min(100, sevPct)}%`;

        if (sev.analysis) {
            if (diseasedPixelsVal) diseasedPixelsVal.textContent = `Infected: ${sev.analysis.diseased_pixels.toLocaleString()} px`;
            if (healthyPixelsVal) healthyPixelsVal.textContent = `Healthy: ${sev.analysis.healthy_pixels.toLocaleString()} px`;
        }

        // 4. Risk Assessment
        const riskScoreVal = document.getElementById('riskScoreVal');
        const riskLevelBadge = document.getElementById('riskLevelBadge');
        const riskBarFill = document.getElementById('riskBarFill');
        const actionUrgencyVal = document.getElementById('actionUrgencyVal');

        const rScore = risk.risk_score || 0;
        if (riskScoreVal) riskScoreVal.textContent = rScore.toFixed(1);
        if (riskLevelBadge) {
            riskLevelBadge.textContent = (risk.risk_level || 'LOW').toUpperCase();
            riskLevelBadge.style.color = risk.risk_color || 'var(--status-healthy)';
        }
        if (riskBarFill) {
            riskBarFill.style.width = `${Math.min(100, rScore)}%`;
            riskBarFill.style.backgroundColor = risk.risk_color || 'var(--accent)';
        }
        if (actionUrgencyVal) {
            actionUrgencyVal.textContent = `Urgency Protocol: ${(risk.action_urgency || 'Monitor').toUpperCase()}`;
        }

        // 5. Agronomic Recommendations
        const immediateList = document.getElementById('immediateActionsList');
        const organicList = document.getElementById('organicTreatmentList');
        const chemicalList = document.getElementById('chemicalTreatmentList');

        if (immediateList) immediateList.innerHTML = '';
        if (organicList) organicList.innerHTML = '';
        if (chemicalList) chemicalList.innerHTML = '';

        const recs = risk.recommendations || {};
        const immediateActions = recs.immediate_actions || ['Maintain regular field scouting protocols.', 'Ensure canopy aeration and optimal drainage.'];

        immediateActions.forEach(action => {
            const li = document.createElement('li');
            li.className = 'rec-action-item';
            li.innerHTML = `<span class="rec-action-bullet"></span><span>${action}</span>`;
            immediateList.appendChild(li);
        });

        // Treatments
        const treatments = recs.treatment || {};
        const organicItems = treatments.organic || ['Apply prophylactic neem extract solution (2%).', 'Dust with horticultural sulfur powder at onset of lesions.'];
        const chemicalItems = treatments.chemical || ['Apply protective Mancozeb foliar spray at recommended rates.', 'Rotate active fungicide modes of action to prevent pathogen resistance.'];

        organicItems.forEach(item => {
            const li = document.createElement('li');
            li.className = 'treatment-item';
            li.textContent = item;
            organicList.appendChild(li);
        });

        chemicalItems.forEach(item => {
            const li = document.createElement('li');
            li.className = 'treatment-item';
            li.textContent = item;
            chemicalList.appendChild(li);
        });

        // Show Dossier
        emptyState.style.display = 'none';
        resultsDossier.classList.add('active');

        // Smooth scroll to dossier on mobile
        if (window.innerWidth < 960) {
            resultsDossier.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    function showError(msg) {
        if (formErrorMessage) {
            formErrorMessage.textContent = msg;
            formErrorMessage.style.display = 'block';
        }
    }

    function hideError() {
        if (formErrorMessage) {
            formErrorMessage.textContent = '';
            formErrorMessage.style.display = 'none';
        }
    }
});
