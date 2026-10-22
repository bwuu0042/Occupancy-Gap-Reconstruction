# AI Use Reflection

## Tools and how I used them

I used Claude (Anthropic) during the project to help plan the approach, draft and debug code, interpret results, and review documentation. I worked in Google Colab and often copied suggested code into the notebook or project files, then ran it and adapted it as the analysis developed.

I also used OpenAI Codex to review the project brief and GitHub repository, explain the project structure and saved results, and help draft parts of the report, results summary, README, and this reflection. Google Colab was my coding and execution environment; it is not an AI tool.

## Approximate level of AI contribution

I would describe my use of AI as **moderate**. AI contributed to planning, coding support, debugging, interpretation, and writing, but I ran the analysis in Colab, adapted suggestions, and checked the work against the project code, tests, and saved outputs. This describes AI as a regular part of my workflow rather than the sole source of the project.

## Why I used AI

I used AI to make the process easier and more efficient, while putting theory learned in class into practical situations, code, experiments, and tests. It helped me get a starting point for unfamiliar tasks and work through errors and explanations more quickly. I still needed to understand what the code did and judge whether the results supported the claims.

## How I checked AI-assisted work

I ran the project in Google Colab and used the automated test suite. The committed HTML test report records **29 passed and 0 failed**. The tests cover expected outputs, invalid inputs, the decay method's analytical behaviour, and checks that hidden gap values do not affect predictions or features.

I also compared the saved accuracy, confidence-interval, sensitivity, and performance outputs with the project code and notebook. The project records fixed-seed reruns that reproduced the reported G.20 and G.25 results. These checks support the implementation and reproducibility claims, but they do not prove that sensor headcounts are independent ground truth or that simulated outages represent every real outage.

## Decisions and responsibility

The final engineering decisions and submitted claims remain my responsibility. I am responsible for understanding the problem, the input and output definitions, why the methods were compared, the meaning of the metrics, and the limitations of simulated gaps and sensor-derived reference values.

The project uses a chronological train, validation, and test split. The learned model settings were selected using validation data and then applied to G.25 without room-specific retuning. I must be able to explain these choices and the results in my presentation and answers.

## Errors and limitations I encountered

AI suggestions were not always correct. During the project, a feature-matrix shape bug, an overstated statistical test, a misreading of the G.25 results, and a depth-4 versus depth-5 error in interpreting a decision-tree validation result were identified and corrected. This reinforced the need to compare suggestions with the code, test outputs, and saved results instead of treating a confident explanation as proof.

AI also cannot decide whether a result is appropriate for this engineering context unless I check the data and assumptions. Copying code without understanding it would make it harder to detect errors or answer questions about the work.

## Effect on my learning and workflow

AI made it easier to move from class concepts to practical code and tests. Working through the notebook helped me connect interpolation, weekly profiles, MAE and RMSE, chronological splits, and validation choices with actual occupancy data. Copying a suggestion gave me a starting point, while running and adapting it showed me where I needed to understand the details and check the outcome.

AI made the workflow more efficient and helped me apply theory from class to practical situations and tests. I remain responsible for the code, results, claims, references, and engineering decisions in my submission.
