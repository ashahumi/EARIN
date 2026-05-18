# Exercise 7: Logic and Inference - Variant 2

This repository contains the Prolog implementation for Variant 2 of Lab 7. The program calculates and returns the number of days $N$ between two given dates under the assumption that the year is 2024 (a leap year).

## Implementation Overview
* **Predicate Name:** `interval`
* **Input Format:** Two string arguments formatted as `"DDMM"` (e.g., `"2205"` for May 22nd)
* **Output:** An integer representing the absolute difference in days printed to the console.

## How to Run the Code

You can execute and test this code easily using either a local environment or the online SWI-Prolog editor.

### Option 1: Using SWI-Prolog Web Editor (SWISH)
1. Open your web browser and go to [SWISH SWI-Prolog](https://swish.swi-prolog.org/).
2. Copy the entire contents of the `.pl` code file from this repository.
3. Paste the code into the large text area on the left-hand side of the screen.
4. Use the query box in the bottom right corner to run queries.

### Option 2: Using Local SWI-Prolog Installation
1. Open your terminal or command prompt in the directory containing the code file.
2. Load the file into the interpreter by running:
   ```prolog
   swipl -s filename.pl