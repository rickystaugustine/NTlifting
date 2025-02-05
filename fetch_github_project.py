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

# ✅ Define GraphQL Query
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
                  ... on ProjectV2FieldCommon {
                    name
                  }
                }
              }
              ... on ProjectV2ItemFieldNumberValue {
                number
                field {
                  ... on ProjectV2FieldCommon {
                    name
                  }
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

# ✅ Debug: Verify API Authentication
if not GITHUB_TOKEN:
    print("🚨 ERROR: `GH_PAT` is missing. Ensure your GitHub token is set correctly in repository secrets.")
    exit(1)

# ✅ Send GraphQL Request
response = requests.post(GRAPHQL_URL, json={"query": query}, headers=HEADERS)
data = response.json()

# ✅ Debug API Response
print("\n🔍 Debugging: Full API Response from GitHub:")
print(json.dumps(data, indent=2))

# ✅ Check for API Errors
if "message" in data and "status" in data and data["status"] == 401:
    print("\n🚨 ERROR: Authentication failed (401 Unauthorized). Check your GH_PAT token permissions.")
    exit(1)

# ✅ Check if the API returned data
if "data" not in data or "user" not in data["data"] or "projectV2" not in data["data"]["user"]:
    print("\n🚨 ERROR: 'data' field not found in response. Check API permissions and query syntax.")
    exit(1)

# ✅ Extract Project Data
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
try:
    df_issues = pd.DataFrame(issues_list)
    df_issues.to_csv("issues_export.csv", index=False)
    df_issues.to_excel("issues_export.xlsx", index=False)
    print("✅ Issues exported successfully.")
except Exception as e:
    print(f"❌ ERROR saving issues files: {e}")

# ✅ Save Kanban Board Data
try:
    df_kanban = pd.DataFrame(kanban_board_data)
    df_kanban.to_csv("kanban_board.csv", index=False)
    df_kanban.to_excel("kanban_board.xlsx", index=False)
    print("✅ Kanban Board exported successfully.")
except Exception as e:
    print(f"❌ ERROR saving Kanban board files: {e}")

# ✅ Generate Charts
df_issues["Estimation"] = pd.to_numeric(df_issues["Estimation"], errors="coerce")
print("\n🔍 Debugging: Checking 'Estimation' Column Before Plotting")
print(df_issues[["Size", "Estimation"]].dropna())

if df_issues["Estimation"].dtype != "float64":
    print("\n🚨 ERROR: 'Estimation' column is not numeric after conversion.")
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
        print(f"✅ {title} Chart saved as {chart_filenames[title]}")
