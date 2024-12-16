import os
import sys
import nbformat
import uuid

def add_links_to_notebook(notebook_path):
    # Read repository details from environment variables
    repo_owner = os.getenv("REPO_OWNER", "default-owner")
    repo_name = os.getenv("REPO_NAME", "default-repo")
    branch_name = os.getenv("BRANCH_NAME", "main")

    # Templates for Colab and nbviewer links
    colab_template = (
        '<a href="https://colab.research.google.com/github/{repo_owner}/{repo_name}/blob/{branch}/{file_path}" target="_parent">'
        '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
    )
    nbviewer_template = (
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
        '<a href="https://nbviewer.org/github/{repo_owner}/{repo_name}/blob/{branch}/{file_path}" target="_parent">'
        '<img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>'
    )

    # Compute the relative file path from the repository root
    file_path = os.path.relpath(notebook_path).replace("\\", "/")
    colab_link = colab_template.format(
        repo_owner=repo_owner, repo_name=repo_name, branch=branch_name, file_path=file_path
    )
    nbviewer_link = nbviewer_template.format(
        repo_owner=repo_owner, repo_name=repo_name, branch=branch_name, file_path=file_path
    )
    full_links = f"{colab_link} {nbviewer_link}"

    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    modified = False  # Track if the notebook has been modified

    # Step 1: Check if the first cell is a Colab auto-added cell
    if notebook["cells"] and notebook["cells"][0]["cell_type"] == "markdown":
        first_cell = notebook["cells"][0]
        if (
            first_cell.get("metadata", {}).get("colab_type") == "text"
            and first_cell.get("metadata", {}).get("id") == "view-in-github"
        ):
            # Remove the Colab auto-added cell
            notebook["cells"].pop(0)
            modified = True
            print(f"Removed Colab auto-added cell from: {notebook_path}")

    # Step 2: Check the next cell (new first cell after removal)
    if notebook["cells"]:
        second_cell = notebook["cells"][0]
        if second_cell["cell_type"] == "markdown":
            source = second_cell["source"]

            # Check if the cell contains both Colab and nbviewer links
            has_colab_link = "colab" in source
            has_nbviewer_link = "nbviewer" in source

            if has_colab_link and has_nbviewer_link:
                # Validate the links
                if colab_link not in source or nbviewer_link not in source:
                    # Rewrite the cell with correct links
                    second_cell["source"] = full_links
                    modified = True
                    print(f"Updated links in: {notebook_path}")
            else:
                # Add a new markdown cell if links are missing
                links_cell = nbformat.v4.new_markdown_cell(source=full_links)
                links_cell["id"] = str(uuid.uuid4())  # Add a unique ID to the new cell
                notebook["cells"].insert(0, links_cell)
                modified = True
                print(f"Added links to: {notebook_path}")
        else:
            # If the second cell is not markdown, add the links as a new cell at the top
            links_cell = nbformat.v4.new_markdown_cell(source=full_links)
            links_cell["id"] = str(uuid.uuid4())  # Add a unique ID to the new cell
            notebook["cells"].insert(0, links_cell)
            modified = True
            print(f"Added links to: {notebook_path}")

    else:
        # If there are no cells, add a new markdown cell with the correct links
        links_cell = nbformat.v4.new_markdown_cell(source=full_links)
        links_cell["id"] = str(uuid.uuid4())  # Add a unique ID to the new cell
        notebook["cells"].insert(0, links_cell)
        modified = True
        print(f"Added links to: {notebook_path}")

    # Save the notebook only if it was modified
    if modified:
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(notebook, f)
        print(f"Saved changes to: {notebook_path}")
    else:
        print(f"No changes needed: {notebook_path}")


def main(file_list_path):
    with open(file_list_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        status, file_path = line.strip().split("\t", 1)
        if status in {"A", "M"}:
            add_links_to_notebook(file_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python add_links_to_notebooks.py <file_list>")
        sys.exit(1)

    main(sys.argv[1])
