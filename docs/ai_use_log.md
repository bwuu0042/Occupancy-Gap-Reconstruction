# AI Use Reflection

> **Draft for me to personalise before submission.** Replace the bracketed prompts with an honest estimate and your own examples. Keep only statements that describe what you actually did and can explain in the presentation.

## Tools and how I used them

I used Claude (Anthropic) during the project to help plan the approach, draft and debug code, interpret results, and review documentation. I worked in Google Colab and often copied Claude's suggested code into the notebook or project files, then ran it and made changes as I developed the analysis.

I also used OpenAI Codex to review the project brief and GitHub repository, explain the project structure and saved results, and draft parts of the written report, consolidated results summary, README update, and this reflection. Google Colab was my coding and execution environment; it is not an AI tool.

## Approximate level of AI contribution

I would describe AI's overall contribution as **substantial assistance with coding and writing**. Claude supplied or revised code and helped me reason through parts of the analysis. Codex drafted substantial report and documentation text. I copied and adapted AI suggestions rather than treating them as verified answers.

**My estimate of the approximate contribution is [add an honest estimate, such as low/moderate/high or a percentage, and explain what the estimate refers to].** I should distinguish code that AI drafted from code or explanations that I changed, tested, and can explain myself.

## Why I used AI

I used AI to help me make progress on unfamiliar coding and analysis tasks, generate or improve a starting point for code, troubleshoot errors, and organise technical explanations. It helped me work through an open-ended project with several methods and outputs.

**[Add a sentence about the main reason AI was useful to you personally, and any parts where using it was less helpful.]**

## How I checked AI-assisted work

I ran the project in Colab using the occupancy dataset supplied for the unit. I ran the automated test suite; the current HTML test report records **29 passed and 0 failed**. The tests cover known outputs, invalid inputs, decay-method analytical behaviour, and checks that hidden gap values do not affect predictions or features.

I also reviewed the saved accuracy, confidence-interval, sensitivity, and performance outputs against the project code and notebook. The README records that fixed-seed reruns reproduced the reported G.20 and G.25 results. These checks support the implementation and reproducibility claims, but they do not prove that sensor headcounts are independent ground truth or that simulated outages represent every real outage.

**[Add any checks you personally performed beyond running the tests, such as checking a calculation by hand, inspecting a plot or comparing a result with an expected value.]**

## Decisions and responsibility

The final engineering decisions and submitted claims remain my responsibility. I am responsible for understanding the problem, the input and output definitions, why the methods were compared, the meaning of the metrics, and the limitations of using simulated gaps and sensor-derived reference values.

The project uses a chronological train/validation/test split, and the learned model settings are documented as selected using validation data. I need to be able to explain these choices and the results without AI assistance during the presentation and questions.

**[Add one or two examples of decisions you personally made, such as which results mattered most to your interpretation or what trade-off you considered when comparing a simple method with a learned model.]**

## Errors and limitations I encountered

AI suggestions were not always correct. During the project, I caught and corrected a feature-matrix shape bug, an overstated statistical test, a misreading of the G.25 results, and a depth-4 versus depth-5 error when interpreting a decision-tree validation result. I had to compare suggestions with the code, test outputs, and saved results rather than assume a confident explanation was correct.

AI also cannot know whether a result is appropriate for this engineering context unless I provide and check the relevant data and assumptions. Copying code without understanding it would make it harder for me to detect errors or answer questions about the work.

**[Check that these examples match your experience. Add another concrete example if there was an AI suggestion you rejected or changed.]**

## Effect on my learning and workflow

AI helped me get a working starting point and resolve coding or writing problems more quickly. It also showed me that passing tests and producing plausible plots are not enough by themselves: I need to understand what each function uses, what the test verifies, and what the result cannot establish.

**[Finish this paragraph in your own words: What can you now explain or do that you could not do at the start? What do you still need to practise? How did copying and adapting code affect your understanding?]**

I remain responsible for all code, results, claims, references, and engineering decisions in my submission. I will revise this reflection so it accurately describes my own contribution and understanding.
