/**
 * ARC AGI Captcha JavaScript - Progressive 3-Step Flow
 * Step 1: Study examples one by one
 * Step 2: Practice on a training example
 * Step 3: Solve the final challenge
 */

class ARCCaptcha {
    constructor() {
        this.currentChallenge = null;
        this.selectedColor = 0;
        this.userSolution = [];
        this.practiceSolution = [];
        this.isSubmitting = false;
        this.currentStep = 1;
        this.currentExampleIndex = 0;
        this.examplesShown = 0;
        this.failureCount = 0;
        this.maxFailures = 3;
        
        this.initializeElements();
        this.setupEventListeners();
        this.showInitialState();
    }
    
    showInitialState() {
        this.loadingEl.style.display = 'none';
        this.challengeContainer.style.display = 'none';
        this.successContainer.style.display = 'none';
        this.deathContainer.style.display = 'none';
        this.feedbackEl.style.display = 'none';
    }
    
    initializeElements() {
        // Main containers
        this.loadingEl = document.getElementById('loading');
        this.challengeContainer = document.getElementById('challenge-container');
        this.successContainer = document.getElementById('success-container');
        this.deathContainer = document.getElementById('death-container');
        
        // Step sections
        this.studySection = document.getElementById('study-section');
        this.practiceSection = document.getElementById('practice-section');
        this.solveSection = document.getElementById('solve-section');
        
        // Progress steps
        this.step1 = document.getElementById('step-1');
        this.step2 = document.getElementById('step-2');
        this.step3 = document.getElementById('step-3');
        
        // Study elements
        this.currentExample = document.getElementById('current-example');
        this.nextExampleBtn = document.getElementById('next-example-btn');
        this.readyPracticeBtn = document.getElementById('ready-practice-btn');
        this.patternHint = document.getElementById('pattern-hint');
        
        // Practice elements
        this.practiceInputGrid = document.getElementById('practice-input-grid');
        this.practiceOutputGrid = document.getElementById('practice-output-grid');
        this.practiceFeedback = document.getElementById('practice-feedback');
        this.backToStudyBtn = document.getElementById('back-to-study-btn');
        this.clearPracticeBtn = document.getElementById('clear-practice-btn');
        this.readySolveBtn = document.getElementById('ready-solve-btn');
        
        // Solve elements
        this.testInputGrid = document.getElementById('test-input-grid');
        this.testOutputGrid = document.getElementById('test-output-grid');
        this.solveFeedback = document.getElementById('solve-feedback');
        this.backToPracticeBtn = document.getElementById('back-to-practice-btn');
        this.clearSolveBtn = document.getElementById('clear-solve-btn');
        this.verifyFinalBtn = document.getElementById('verify-final-btn');
        
        // Shared elements
        this.colorPalette = document.getElementById('color-palette');
        this.colorPicker = document.getElementById('color-picker');
        this.difficultySelect = document.getElementById('difficulty-select');
        this.newChallengeBtn = document.getElementById('new-challenge-btn');
        
        // Captcha elements
        this.humanCheckbox = document.getElementById('human-checkbox');
        this.solveTime = document.getElementById('solve-time');
        this.restartBtn = document.getElementById('restart-btn');
        this.copyTokenBtn = document.getElementById('copy-token-btn');
        
        // Death page elements
        this.tryAgainBtn = document.getElementById('try-again-btn');
        this.failureCountEl = document.getElementById('failure-count');
        
        // Feedback
        this.feedbackEl = document.getElementById('feedback');
        this.verificationTokenEl = document.getElementById('verification-token');
        
        // Track timing
        this.startTime = null;
    }
    
    setupEventListeners() {
        // Captcha checkbox
        this.humanCheckbox.addEventListener('change', (e) => {
            if (e.target.checked) {
                this.startChallenge();
            } else {
                this.resetChallenge();
            }
        });
        
        // Study step buttons
        if (this.nextExampleBtn) this.nextExampleBtn.addEventListener('click', () => this.showNextExample());
        if (this.readyPracticeBtn) this.readyPracticeBtn.addEventListener('click', () => this.goToStep(2));
        
        // Practice step buttons
        if (this.backToStudyBtn) this.backToStudyBtn.addEventListener('click', () => this.goToStep(1));
        if (this.clearPracticeBtn) this.clearPracticeBtn.addEventListener('click', () => this.clearPracticeGrid());
        if (this.readySolveBtn) this.readySolveBtn.addEventListener('click', () => this.goToStep(3));
        
        // Solve step buttons
        if (this.backToPracticeBtn) this.backToPracticeBtn.addEventListener('click', () => this.goToStep(2));
        if (this.clearSolveBtn) this.clearSolveBtn.addEventListener('click', () => this.clearSolveGrid());
        if (this.verifyFinalBtn) this.verifyFinalBtn.addEventListener('click', () => this.submitSolution());
        
        // Challenge controls
        if (this.newChallengeBtn) this.newChallengeBtn.addEventListener('click', () => this.loadNewChallenge());
        
        // Success actions
        if (this.restartBtn) this.restartBtn.addEventListener('click', () => this.restart());
        if (this.copyTokenBtn) this.copyTokenBtn.addEventListener('click', () => this.copyToken());
        
        // Death page actions
        if (this.tryAgainBtn) this.tryAgainBtn.addEventListener('click', () => this.restart());
        
        // Difficulty change
        if (this.difficultySelect) this.difficultySelect.addEventListener('change', () => this.loadNewChallenge());
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key >= '0' && e.key <= '9') {
                this.selectColor(parseInt(e.key));
            } else if (e.key === 'r' || e.key === 'R') {
                if (this.currentStep === 2) this.clearPracticeGrid();
                else if (this.currentStep === 3) this.clearSolveGrid();
            } else if (e.key === 'Enter') {
                if (this.currentStep === 3) this.submitSolution();
            }
        });
    }
    
    async startChallenge() {
        this.humanCheckbox.disabled = true;
        await this.loadChallenge();
    }
    
    async loadChallenge() {
        try {
            this.showLoading();
            const difficulty = this.difficultySelect ? this.difficultySelect.value : 'easy';
            
            const response = await fetch(`/api/challenge?difficulty=${difficulty}`);
            if (!response.ok) {
                throw new Error('Failed to load challenge');
            }
            
            this.currentChallenge = await response.json();
            this.startTime = new Date();
            this.currentStep = 1;
            this.currentExampleIndex = 0;
            this.examplesShown = 0;
            
            this.setupColorPicker();
            this.showChallenge();
            this.goToStep(1);
            
        } catch (error) {
            console.error('Error loading challenge:', error);
            this.showError('Failed to load challenge. Please try again.');
        }
    }
    
    async loadNewChallenge() {
        this.currentChallenge = null;
        await this.loadChallenge();
    }
    
    goToStep(step) {
        this.currentStep = step;
        this.updateProgressBar();
        
        // Hide all sections
        if (this.studySection) this.studySection.style.display = 'none';
        if (this.practiceSection) this.practiceSection.style.display = 'none';
        if (this.solveSection) this.solveSection.style.display = 'none';
        
        // Remove active classes
        [this.studySection, this.practiceSection, this.solveSection].forEach(section => {
            if (section) section.classList.remove('active');
        });
        
        switch (step) {
            case 1:
                this.showStudyStep();
                break;
            case 2:
                this.showPracticeStep();
                break;
            case 3:
                this.showSolveStep();
                break;
        }
        
        // Show color palette for interactive steps
        if (this.colorPalette) {
            this.colorPalette.style.display = step >= 2 ? 'block' : 'none';
        }
    }
    
    updateProgressBar() {
        // Update progress bar visual state
        [this.step1, this.step2, this.step3].forEach((step, index) => {
            if (step) {
                step.classList.remove('active', 'completed');
                if (index + 1 === this.currentStep) {
                    step.classList.add('active');
                } else if (index + 1 < this.currentStep) {
                    step.classList.add('completed');
                }
            }
        });
    }
    
    showStudyStep() {
        if (this.studySection) {
            this.studySection.style.display = 'block';
            this.studySection.classList.add('active');
        }
        this.renderCurrentExample();
        this.updateStudyButtons();
    }
    
    renderCurrentExample() {
        if (!this.currentChallenge || !this.currentExample) return;
        
        const example = this.currentChallenge.train[this.currentExampleIndex];
        if (!example) return;
        
        this.currentExample.innerHTML = `
            <div class="example-display">
                <div class="example-side">
                    <h4>📥 Input</h4>
                    <div class="example-input-grid"></div>
                </div>
                <div class="example-arrow">→<br/><span style="font-size: 0.8rem;">Transform</span></div>
                <div class="example-side">
                    <h4>📤 Output</h4>
                    <div class="example-output-grid"></div>
                </div>
            </div>
        `;
        
        const inputContainer = this.currentExample.querySelector('.example-input-grid');
        const outputContainer = this.currentExample.querySelector('.example-output-grid');
        
        this.renderGrid(example.input, inputContainer, false);
        this.renderGrid(example.output, outputContainer, false);
    }
    
    updateStudyButtons() {
        const hasMoreExamples = this.currentExampleIndex < this.currentChallenge.train.length - 1;
        
        if (this.nextExampleBtn) {
            this.nextExampleBtn.textContent = hasMoreExamples ? 
                `👁️ Show Another Example (${this.examplesShown + 1}/${this.currentChallenge.train.length})` : 
                '👁️ Show First Example Again';
        }
    }
    
    showNextExample() {
        this.currentExampleIndex = (this.currentExampleIndex + 1) % this.currentChallenge.train.length;
        this.examplesShown = Math.max(this.examplesShown, this.currentExampleIndex + 1);
        this.renderCurrentExample();
        this.updateStudyButtons();
    }
    
    showHint() {
        if (this.currentChallenge && this.currentChallenge.hint && this.patternHint) {
            this.patternHint.textContent = this.currentChallenge.hint;
            this.patternHint.style.display = 'block';
            
            setTimeout(() => {
                if (this.patternHint) {
                    this.patternHint.style.display = 'none';
                }
            }, 8000);
        }
    }
    
    showPracticeStep() {
        if (this.practiceSection) {
            this.practiceSection.style.display = 'block';
            this.practiceSection.classList.add('active');
        }
        
        // Use first training example for practice
        const practiceExample = this.currentChallenge.train[0];
        this.renderGrid(practiceExample.input, this.practiceInputGrid, false);
        
        // Initialize practice solution
        this.practiceSolution = practiceExample.input.map(row => [...row]);
        this.renderGrid(this.practiceSolution, this.practiceOutputGrid, true);
        
        this.updatePracticeFeedback('Try to recreate the pattern you learned. Click cells to paint them!');
    }
    
    showSolveStep() {
        if (this.solveSection) {
            this.solveSection.style.display = 'block';
            this.solveSection.classList.add('active');
        }
        
        // Use test case
        const testCase = this.currentChallenge.test[0];
        this.renderGrid(testCase.input, this.testInputGrid, false);
        
        // Initialize user solution
        this.userSolution = testCase.input.map(row => [...row]);
        this.renderGrid(this.userSolution, this.testOutputGrid, true);
        
        this.updateSolveFeedback('Apply the pattern to solve this final challenge!');
    }
    
    renderGrid(gridData, container, interactive = false) {
        if (!container || !gridData) return;
        
        const rows = gridData.length;
        const cols = gridData[0].length;
        
        container.className = 'grid' + (interactive ? ' interactive' : '');
        container.style.gridTemplateRows = `repeat(${rows}, 1fr)`;
        container.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
        container.innerHTML = '';
        
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const cell = document.createElement('div');
                cell.className = `grid-cell color-${gridData[i][j]}`;
                cell.dataset.row = i;
                cell.dataset.col = j;
                cell.dataset.color = gridData[i][j];
                
                if (interactive) {
                    cell.addEventListener('click', () => this.handleCellClick(i, j, container));
                }
                
                container.appendChild(cell);
            }
        }
    }
    
    handleCellClick(row, col, container) {
        if (this.isSubmitting) return;
        
        let targetSolution;
        if (container === this.practiceOutputGrid) {
            targetSolution = this.practiceSolution;
        } else if (container === this.testOutputGrid) {
            targetSolution = this.userSolution;
        } else {
            return;
        }
        
        // Update solution
        targetSolution[row][col] = this.selectedColor;
        
        // Update visual
        const cell = container.querySelector(`[data-row="${row}"][data-col="${col}"]`);
        if (cell) {
            cell.className = cell.className.replace(/color-\d+/g, '');
            cell.classList.add(`color-${this.selectedColor}`);
            cell.dataset.color = this.selectedColor;
            
            // Paint effect
            cell.style.transform = 'scale(1.15)';
            setTimeout(() => {
                cell.style.transform = 'scale(1)';
            }, 200);
        }
        
        // Update feedback
        if (this.currentStep === 2) {
            this.updatePracticeFeedback(`Painted cell (${row + 1}, ${col + 1}) with color ${this.selectedColor}`);
        } else if (this.currentStep === 3) {
            this.updateSolveFeedback(`Painted cell (${row + 1}, ${col + 1}) with color ${this.selectedColor}`);
        }
    }
    
    setupColorPicker() {
        if (!this.colorPicker || !this.currentChallenge) return;
        
        // Get unique colors from the challenge
        const allColors = new Set();
        
        this.currentChallenge.train.forEach(example => {
            [...example.input, ...example.output].forEach(row => {
                row.forEach(color => allColors.add(color));
            });
        });
        
        this.currentChallenge.test[0].input.forEach(row => {
            row.forEach(color => allColors.add(color));
        });
        
        // Create color options
        const sortedColors = Array.from(allColors).sort((a, b) => a - b);
        this.colorPicker.innerHTML = '';
        
        sortedColors.forEach(color => {
            const colorOption = document.createElement('div');
            colorOption.className = `color-option color-${color}`;
            colorOption.dataset.color = color;
            colorOption.setAttribute('data-color', color);
            colorOption.title = `Color ${color} (Press ${color} key)`;
            
            if (color === this.selectedColor) {
                colorOption.classList.add('selected');
            }
            
            colorOption.addEventListener('click', () => this.selectColor(color));
            this.colorPicker.appendChild(colorOption);
        });
    }
    
    selectColor(color) {
        this.selectedColor = color;
        
        if (this.colorPicker) {
            this.colorPicker.querySelectorAll('.color-option').forEach(option => {
                option.classList.remove('selected');
            });
            
            const selectedOption = this.colorPicker.querySelector(`[data-color="${color}"]`);
            if (selectedOption) {
                selectedOption.classList.add('selected');
            }
        }
    }
    
    clearPracticeGrid() {
        if (this.currentChallenge) {
            const practiceExample = this.currentChallenge.train[0];
            this.practiceSolution = practiceExample.input.map(row => [...row]);
            this.renderGrid(this.practiceSolution, this.practiceOutputGrid, true);
            this.updatePracticeFeedback('Practice grid cleared!');
        }
    }
    
    clearSolveGrid() {
        if (this.currentChallenge) {
            const testCase = this.currentChallenge.test[0];
            this.userSolution = testCase.input.map(row => [...row]);
            this.renderGrid(this.userSolution, this.testOutputGrid, true);
            this.updateSolveFeedback('Grid cleared!');
        }
    }
    
    async submitSolution() {
        if (this.isSubmitting || !this.currentChallenge) return;
        
        this.isSubmitting = true;
        if (this.verifyFinalBtn) {
            this.verifyFinalBtn.disabled = true;
            this.verifyFinalBtn.innerHTML = '<span>⏳</span> Verifying...';
        }
        
        try {
            const response = await fetch('/api/verify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    solution: this.userSolution
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showSuccess(result.token);
            } else {
                this.failureCount++;
                if (this.failureCount >= this.maxFailures) {
                    this.showDeathPage();
                } else {
                    this.updateSolveFeedback(`❌ Wrong! ${this.maxFailures - this.failureCount} attempts remaining!`, 'error');
                }
            }
            
        } catch (error) {
            console.error('Error submitting solution:', error);
            this.updateSolveFeedback('❌ Failed to submit. Please try again.', 'error');
        } finally {
            this.isSubmitting = false;
            if (this.verifyFinalBtn) {
                this.verifyFinalBtn.disabled = false;
                this.verifyFinalBtn.innerHTML = '<span>✅</span> Verify I\'m Human';
            }
        }
    }
    
    updatePracticeFeedback(message, type = 'info') {
        if (this.practiceFeedback) {
            this.practiceFeedback.textContent = message;
            this.practiceFeedback.className = `live-feedback ${type}`;
        }
    }
    
    updateSolveFeedback(message, type = 'info') {
        if (this.solveFeedback) {
            this.solveFeedback.textContent = message;
            this.solveFeedback.className = `live-feedback ${type}`;
        }
    }
    
    showLoading() {
        if (this.loadingEl) this.loadingEl.style.display = 'block';
        if (this.challengeContainer) this.challengeContainer.style.display = 'none';
        if (this.successContainer) this.successContainer.style.display = 'none';
        if (this.feedbackEl) this.feedbackEl.style.display = 'none';
    }
    
    showChallenge() {
        if (this.loadingEl) this.loadingEl.style.display = 'none';
        if (this.challengeContainer) {
            this.challengeContainer.style.display = 'block';
        }
        if (this.successContainer) this.successContainer.style.display = 'none';
        if (this.feedbackEl) this.feedbackEl.style.display = 'none';
        
        // Hide hero section and show main content with transition
        const heroSection = document.getElementById('hero-section');
        const mainContent = document.querySelector('.main-content');
        
        if (heroSection) {
            heroSection.style.transform = 'translateY(-20px)';
            heroSection.style.opacity = '0';
            heroSection.style.transition = 'all 0.5s ease';
            setTimeout(() => {
                heroSection.style.display = 'none';
            }, 500);
        }
        
        if (mainContent) {
            setTimeout(() => {
                mainContent.classList.add('active');
            }, 300);
        }
    }
    
    showSuccess(token) {
        if (this.loadingEl) this.loadingEl.style.display = 'none';
        if (this.challengeContainer) this.challengeContainer.style.display = 'none';
        if (this.successContainer) this.successContainer.style.display = 'block';
        if (this.feedbackEl) this.feedbackEl.style.display = 'none';
        if (this.deathContainer) this.deathContainer.style.display = 'none';
        
        // Check the checkbox
        if (this.humanCheckbox) {
            this.humanCheckbox.checked = true;
            this.humanCheckbox.disabled = true;
        }
        
        // Calculate solve time
        if (this.startTime && this.solveTime) {
            const endTime = new Date();
            const solveSeconds = Math.round((endTime - this.startTime) / 1000);
            this.solveTime.textContent = `${solveSeconds}s`;
        }
        
        if (this.verificationTokenEl) {
            this.verificationTokenEl.value = token;
        }
    }

    showDeathPage() {
        if (this.loadingEl) this.loadingEl.style.display = 'none';
        if (this.challengeContainer) this.challengeContainer.style.display = 'none';
        if (this.successContainer) this.successContainer.style.display = 'none';
        if (this.feedbackEl) this.feedbackEl.style.display = 'none';
        if (this.deathContainer) this.deathContainer.style.display = 'block';
        
        // Update failure count display
        if (this.failureCountEl) {
            this.failureCountEl.textContent = `${this.failureCount}/${this.maxFailures}`;
        }
        
        // Uncheck and disable checkbox
        if (this.humanCheckbox) {
            this.humanCheckbox.checked = false;
            this.humanCheckbox.disabled = true;
        }
    }
    
    showError(message) {
        if (this.feedbackEl) {
            this.feedbackEl.textContent = message;
            this.feedbackEl.className = 'feedback error';
            this.feedbackEl.style.display = 'block';
            
            setTimeout(() => {
                this.feedbackEl.style.display = 'none';
            }, 5000);
        }
    }
    
    resetChallenge() {
        this.currentChallenge = null;
        this.currentStep = 1;
        this.currentExampleIndex = 0;
        this.examplesShown = 0;
        this.showInitialState();
    }
    
    restart() {
        this.currentChallenge = null;
        this.selectedColor = 0;
        this.userSolution = [];
        this.practiceSolution = [];
        this.startTime = null;
        this.currentStep = 1;
        this.currentExampleIndex = 0;
        this.examplesShown = 0;
        this.failureCount = 0;
        
        if (this.humanCheckbox) {
            this.humanCheckbox.checked = false;
            this.humanCheckbox.disabled = false;
        }
        
        if (this.patternHint) this.patternHint.style.display = 'none';
        if (this.colorPalette) this.colorPalette.style.display = 'none';
        
        // Show hero section again
        const heroSection = document.getElementById('hero-section');
        const mainContent = document.querySelector('.main-content');
        
        if (mainContent) {
            mainContent.classList.remove('active');
        }
        
        if (heroSection) {
            setTimeout(() => {
                heroSection.style.display = 'flex';
                heroSection.style.transform = 'translateY(0)';
                heroSection.style.opacity = '1';
            }, 300);
        }
        
        this.showInitialState();
    }
    
    copyToken() {
        if (this.verificationTokenEl) {
            this.verificationTokenEl.select();
            this.verificationTokenEl.setSelectionRange(0, 99999);
            
            try {
                document.execCommand('copy');
                if (this.copyTokenBtn) {
                    const original = this.copyTokenBtn.textContent;
                    this.copyTokenBtn.textContent = 'Copied!';
                    setTimeout(() => {
                        this.copyTokenBtn.textContent = original;
                    }, 2000);
                }
            } catch (err) {
                console.error('Failed to copy token:', err);
            }
        }
    }
}

// Initialize the captcha when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new ARCCaptcha();
}); 