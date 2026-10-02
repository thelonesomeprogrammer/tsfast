import pandas as pd

def update_readme():
    # Read the latest rows from benchmarks.csv
    df = pd.read_csv('.jules/benchmarks.csv')

    # Generate the Markdown table for the README
    latest_date = df['Date'].max()
    latest_df = df[df['Date'] == latest_date]

    md_table = latest_df.to_markdown(index=False)

    # Read the existing README
    with open('README.md', 'r') as f:
        readme_content = f.read()

    # Find the "## Latest Results" section and replace it
    start_index = readme_content.find('## Latest Results')
    if start_index != -1:
        new_content = readme_content[:start_index] + "## Latest Results\n" + md_table + "\n"
        with open('README.md', 'w') as f:
            f.write(new_content)
        print("README.md updated.")
    else:
        print("Could not find '## Latest Results' in README.md")

if __name__ == '__main__':
    update_readme()
