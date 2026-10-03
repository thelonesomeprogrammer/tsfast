import os

with open("src/sliding.rs", "r") as f:
    text = f.read()

text = text.replace("let column_results: Result<Vec<Vec<Vec<f32>>>, String> = self.states[..n_cols]", "let column_results = self.states[..n_cols]")

with open("src/sliding.rs", "w") as f:
    f.write(text)
