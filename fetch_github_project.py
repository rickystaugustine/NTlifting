import requests
import json
import pandas as pd
import os
import matplotlib.pyplot as plt

# ✅ Define Repository Information
GITHUB_TOKEN = os.getenv("GH_PAT")
OWNER = "rickystaugustine"
REPO = "NTlifting"
PROJECT_NUMBER = 1
GRAPHQL_URL = "https://api.github.com/graphql"

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

# ✅ Optimized GraphQL Query to Fetch Issues & Kanban Board
query = """
{
  user(login: "rickystaugustine") {
    projectV2(number: 1) {
      title
      url
      items(first: 50) {
        nodes {
          id
          fieldValues(first: 10) {
            nodes {
              ... on ProjectV2ItemFieldTextValue {
                text
              }
              ... on ProjectV2ItemFieldSingleSelectValue {
                name
                field {
                  ... on ProjectV2FieldCommon {
                    name
                  }
                }
              }
              ... on ProjectV2ItemFieldDateValue {
                date
                field {
                  name
                }
              }
              ... on ProjectV2ItemFieldNumberValue {
                number
                field {
                  name
                }
              }
            }
          }
          content {
            ... on Issue {
              number
              title
              url
              state
              labels(first: 10) {
                nodes {
                  name
                }
              }
            }
          }
        }
      }
    }
  }
}
"""

# ✅ Send GraphQL Request
response = requests.post(GRAPHQL_URL, json={"query": query}, headers=HEADERS)
data = response.json()

# ✅ Extract Project Data
if "data" in data and "user" in data["data"] and "projectV2" in data["data"]["user"]:
    project = data["data"]["user"]["projectV2"]
    print(f"\n📌 **Project Title:** {project['title']}")
    print(f"🔗 **Project URL:** {project['url']}\n")

    # ✅ Extract Issue Data
    issues_list = []
    kanban_board_data = []

    for item in project["items"]["nodes"]:
        if "content" in item and item["content"] is not None:
            issue_data = {
                "Issue Number": item["content"].get("number", "N/A"),
                "Title": item["content"].get("title", "Unknown"),
                "Status": item["content"].get("state", "Unknown"),
                "Labels": ", ".join([label["name"] for label in item["content"].get("labels", {}).get("nodes", [])]),
                "URL": item["content"].get("url", "No URL"),
                "Priority": "Unassigned",
                "Size": "Unassigned",
                "Estimation": "Unassigned",
                "Start Date": "Unassigned",
                "Due Date": "Unassigned",
            }

            if "fieldValues" in item and "nodes" in item["fieldValues"]:
                for field in item["fieldValues"]["nodes"]:
                    if "name" in field and "field" in field:
                        field_name = field["field"]["name"]

                        if "Priority" in field_name:
                            issue_data["Priority"] = field["name"]
                        elif "Size" in field_name:
                            issue_data["Size"] = field["name"]
                        elif "Estimation" in field_name:
                            issue_data["Estimation"] = field["number"]
                        elif "Start Date" in field_name:
                            issue_data["Start Date"] = field["date"]
                        elif "Due Date" in field_name:
                            issue_data["Due Date"] = field["date"]

            issues_list.append(issue_data)

        # ✅ Extract Kanban Board Data
        if "fieldValues" in item and "nodes" in item["fieldValues"]:
            kanban_column = None
            issue_title = item["content"].get("title", "Unknown") if "content" in item else "Unknown"

            for field in item["fieldValues"]["nodes"]:
                if "name" in field and "field" in field and "name" in field["field"]:
                    if "Status" in field["field"]["name"]:
                        kanban_column = field["name"]

            if kanban_column:
                kanban_board_data.append({"Column": kanban_column, "Issue": issue_title})

    # ✅ Save Issue Data
    df_issues = pd.DataFrame(issues_list)
    df_issues.to_csv("issues_export.csv", index=False)
    df_issues.to_excel("issues_export.xlsx", index=False)

    # ✅ Save Kanban Board Data
    if kanban_board_data:
        df_kanban = pd.DataFrame(kanban_board_data)
        df_kanban.to_csv("kanban_board.csv", index=False)
        df_kanban.to_excel("kanban_board.xlsx", index=False)

    # ✅ Generate Charts
    charts = {
        "Issue Status Breakdown": df_issues["Status"].value_counts(),
        "Issue Priority Distribution": df_issues["Priority"].value_counts(),
        "Issue Size Distribution": df_issues["Size"].value_counts(),
        "Estimation vs. Size": df_issues.groupby("Size")["Estimation"].mean(),
        "Label Usage Distribution": df_issues["Labels"].str.split(", ").explode().value_counts(),
    }

    chart_filenames = {
        "Issue Status Breakdown": "issue_status_chart.png",
        "Issue Priority Distribution": "priority_chart.png",
        "Issue Size Distribution": "size_chart.png",
        "Estimation vs. Size": "estimation_vs_size_chart.png",
        "Label Usage Distribution": "labels_chart.png",
    }

    for title, data in charts.items():
        if not data.empty:
            plt.figure(figsize=(10, 6))
            data.plot(kind="bar", color="blue")
            plt.title(title)
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.tight_layout()
            plt.savefig(chart_filenames[title])
